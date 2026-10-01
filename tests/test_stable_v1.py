"""Acceptance failures and train/validation/test independence for the stable path."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from mcdrp.experiments.stable_v1 import (
    check_inputs,
    input_path,
    inventory,
    scientific_config_sha256,
    sha256,
    verify_run,
    write_json,
)
from mcdrp.features.expression import fit_expression_pipeline, transform_expression
from mcdrp.models.stable_core import replay, train_case


def test_expression_projection_is_stable_across_blas_thread_profiles():
    from threadpoolctl import threadpool_limits

    rng = np.random.default_rng(42)
    expression = pd.DataFrame(rng.normal(size=(717, 1024)))
    projected = []
    for threads in (8, 16):
        with threadpool_limits(limits=threads, user_api="blas"):
            pipeline = fit_expression_pipeline(
                expression.iloc[:616], n_components=64, random_state=42
            )
            assert pipeline.blas_threads == 8
            projected.append(transform_expression(pipeline, expression))
    np.testing.assert_allclose(projected[0], projected[1], rtol=0, atol=1e-7)


def test_verified_inputs_reject_changes_and_path_escape(tmp_path):
    source = tmp_path / "raw.csv"
    source.write_text("frozen raw data")
    config = {"inputs": {"source": {"path": "raw.csv", "sha256": sha256(source)}}}
    assert check_inputs(config, tmp_path)["source"] == str(source)
    source.write_text("updated raw data")
    with pytest.raises(ValueError, match="checksum"):
        check_inputs(config, tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        input_path(tmp_path, "../other.csv")
    with pytest.raises(ValueError, match="escapes"):
        input_path(tmp_path, str(source.resolve()))


def test_release_reference_checksum_does_not_change_recipe_but_parameters_do():
    config = {
        "seed": 42,
        "preprocessing_version": "expression_float64_v1",
        "release_reference_sha256": "initial",
    }
    expected = scientific_config_sha256(config)
    config["release_reference_sha256"] = "pinned"
    assert scientific_config_sha256(config) == expected
    config["seed"] = 43
    assert scientific_config_sha256(config) != expected


@pytest.mark.parametrize("change", ["missing", "modified", "extra"])
def test_release_verification_rejects_incomplete_or_changed_artifacts(tmp_path, change):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"verified fitted model")
    write_json(
        tmp_path / "manifest.json",
        {
            "status": "complete",
            "config_sha256": "frozen",
            "artifacts": inventory(tmp_path),
        },
    )
    assert verify_run(tmp_path)["files"] == 1
    if change == "missing":
        artifact.unlink()
    elif change == "modified":
        artifact.write_bytes(b"stale model")
    else:
        (tmp_path / "unrecorded.csv").write_text("extra predictions")
    with pytest.raises(ValueError, match="missing, added or changed"):
        verify_run(tmp_path)


def test_stable_saved_models_and_selection_ignore_test_labels(tmp_path):
    rng = np.random.default_rng(7)
    cohort = pd.DataFrame(
        {
            "pair_id": [f"p{k}" for k in range(36)],
            "drug_id": np.repeat([1, 2, 3, 4, 5, 6], 6),
            "depmap_id": [f"c{k % 6}" for k in range(36)],
            "ln_ic50": rng.normal(size=36),
        }
    )
    assignments = pd.DataFrame(
        {
            "pair_id": cohort.pair_id,
            "split": ["train"] * 24 + ["validation"] * 6 + ["test"] * 6,
        }
    )
    splits = tmp_path / "splits"
    splits.mkdir()
    assignments.to_csv(splits / "cold_drug.csv", index=False)
    features = {
        "cells": {f"c{k}": rng.normal(size=3).astype(np.float32) for k in range(6)},
        **{
            name: {str(k): rng.normal(size=4).astype(np.float32) for k in range(1, 7)}
            for name in ("targets", "morgan", "embeddings")
        },
    }
    joblib.dump(features, tmp_path / "features.joblib")
    outputs = []
    for version in range(2):
        cp = tmp_path / f"cohort_{version}.csv"
        modified = cohort.copy()
        if version:
            modified.loc[assignments.split.eq("test"), "ln_ic50"] += 1000
        modified.to_csv(cp, index=False)
        case_path = tmp_path / f"case_{version}.json"
        output = tmp_path / f"fits_{version}"
        write_json(
            case_path,
            {
                "name": "synthetic",
                "cohort": str(cp),
                "assignments": str(splits / "cold_drug.csv"),
                "target": "ln_ic50",
                "features": str(tmp_path / "features.joblib"),
                "output": str(output),
                "split_name": "cold_drug",
                "rf_trees": 8,
                "xgb_params": {
                    "n_estimators": 12,
                    "max_depth": 2,
                    "learning_rate": 0.1,
                    "early_stopping_rounds": 3,
                    "tree_method": "hist",
                    "n_jobs": 1,
                    "random_state": 42,
                    "device": "cpu",
                },
            },
        )
        train_case(str(case_path), ("B0", "RF", "B6", "B7"))
        outputs.append(output)
        with pytest.raises(ValueError, match="never overwrites"):
            train_case(str(case_path), ("B0",))
    for first in outputs[0].glob("*/fit.json"):
        second = outputs[1] / first.parent.name / "fit.json"
        fit1, fit2 = json.loads(first.read_text()), json.loads(second.read_text())
        assert fit1["candidates"] == fit2["candidates"]
        assert fit1["best_iteration"] == fit2["best_iteration"]
        saved1 = joblib.load(first.parent / "model.joblib")
        saved2 = joblib.load(second.parent / "model.joblib")
        np.testing.assert_array_equal(
            replay(saved1, cohort, features), replay(saved2, cohort, features)
        )
