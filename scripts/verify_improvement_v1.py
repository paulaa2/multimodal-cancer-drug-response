"""Independently verify complete predictions, validation selection and saved fits."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd

from mcdrp.experiments.stable_v1 import inventory, sha256, verify_run, write_json
from mcdrp.models.improvement_core import (
    apply_adapter,
    cell_partition,
    fit_adapter,
    select_ensemble,
)
from mcdrp.models.residual_neural import feature_tables, predict, restore
from mcdrp.results.metrics_report import build_metrics_report
from mcdrp.results.uncertainty import conformal_radius, interval_diagnostics


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-9, equal_nan=True)


def main() -> None:
    config = json.loads(Path("configs/improvement_v1.json").read_text())
    baseline, output = Path(config["baseline_root"]), Path(config["output_root"])
    original = verify_run(baseline)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert sha256(output / "config.json") == manifest["config_sha256"]
    assert inventory(output) == manifest["files"]
    cases, restored_count, maximum_delta = [], 0, 0.0
    all_pairs = {"ensemble": [], "neural": []}
    for name in config["cases"]:
        case = json.loads((baseline / "cases" / f"{name}.json").read_text())
        cohort = pd.read_csv(case["cohort"], low_memory=False)
        assignments = pd.read_csv(case["assignments"])
        rows = cohort.merge(assignments, on="pair_id", validate="one_to_one")
        train = rows.loc[rows.split.eq("train")]
        val = rows.loc[rows.split.eq("validation")]
        test = rows.loc[rows.split.eq("test")]
        val_predictions = []
        for member in config["ensemble_members"]:
            p = pd.read_csv(baseline / "fits" / name / member / "predictions.csv")
            val_predictions.append(
                p.set_index("pair_id").reindex(val.pair_id).y_pred.to_numpy()
            )
        selection = select_ensemble(
            val[case["target"]].to_numpy(),
            np.column_stack(val_predictions),
            config["ensemble_grid_step"],
        )
        saved = json.loads((output / "ensemble" / name / "selection.json").read_text())
        close(selection["weights"], saved["weights"])
        for family in ["ensemble", "neural"]:
            path = output / family / name
            # Exact float round trips preserve near-tied neural predictions and ranks.
            predictions = pd.read_csv(path / "predictions.csv", float_precision="round_trip")
            recalculated = build_metrics_report(
                predictions, cohort, Path(case["assignments"]).parent, target=case["target"]
            )
            recorded = pd.read_csv(path / "metrics.csv")
            numeric = recalculated.select_dtypes(include="number").columns
            recorded = recorded.set_index("subset").reindex(recalculated.subset)
            close(recalculated[numeric].to_numpy(), recorded[numeric].to_numpy())
            assert predictions[predictions.subset.eq("test")].shape[0] == len(test)
            if name.startswith("cold_"):
                all_pairs[family].extend(test.pair_id.tolist())
            if family == "ensemble":
                matrix = np.column_stack(
                    [
                        pd.read_csv(baseline / "fits" / name / member / "predictions.csv")
                        .set_index("pair_id")
                        .reindex(predictions.pair_id)
                        .y_pred.to_numpy()
                        for member in config["ensemble_members"]
                    ]
                )
                close(matrix @ np.array(saved["weights"]), predictions.y_pred.to_numpy())
        features = joblib.load(case["features"])
        neural = output / "neural" / name
        chosen = json.loads((neural / "selection.json").read_text())
        candidate_scores = []
        for i, candidate in enumerate(config["neural"]["candidates"]):
            seed_predictions = []
            for seed in config["neural"]["seeds"]:
                artifact = joblib.load(neural / f"candidate_{i}_seed_{seed}.joblib")
                tables = feature_tables(
                    rows, features, candidate["blocks"], train, artifact["scalers"]
                )
                prediction = predict(
                    restore(artifact, "cuda"),
                    val,
                    tables,
                    device="cuda",
                    mean=artifact["mean"],
                    scale=artifact["scale"],
                )
                rmse = np.sqrt(np.mean((prediction - val[case["target"]].to_numpy()) ** 2))
                close(rmse, artifact["validation_rmse"])
                seed_predictions.append(prediction)
                restored_count += 1
                if i == chosen["selected_candidate"]:
                    actual = predict(
                        restore(artifact, "cuda"),
                        test,
                        tables,
                        device="cuda",
                        mean=artifact["mean"],
                        scale=artifact["scale"],
                    )
                    stored = pd.read_csv(neural / f"selected_seed_{seed}_test.csv")
                    expected = (
                        stored.set_index("pair_id").reindex(test.pair_id).y_pred.to_numpy()
                    )
                    close(actual, expected)
                    maximum_delta = max(maximum_delta, float(np.max(np.abs(actual - expected))))
            average = np.mean(seed_predictions, axis=0)
            if case["target"] == "auc":
                average = np.clip(average, 0, 1)
            candidate_scores.append(
                np.sqrt(np.mean((average - val[case["target"]].to_numpy()) ** 2))
            )
        assert np.argmin(candidate_scores) == chosen["selected_candidate"]
        cases.append({"case": name, "test_pairs": len(test), "neural_candidates": 4})
    for pairs in all_pairs.values():
        assert len(pairs) == len(set(pairs)) == 149364
    setup = config["adaptation"]
    case = json.loads((baseline / "cases" / f"{setup['case']}.json").read_text())
    rows = pd.read_csv(case["cohort"], low_memory=False).merge(
        pd.read_csv(case["assignments"]), on="pair_id", validate="one_to_one"
    )
    prism = rows.loc[rows.split.eq("test")].drop(columns="split")
    labels = cell_partition(
        prism, seed=setup["seed"], fractions=setup["fractions"], roles=setup["roles"]
    )
    recorded = pd.read_csv(output / "adaptation/assignments.csv")
    assert (
        labels.set_index("pair_id")["split"]
        .sort_index()
        .equals(recorded.set_index("pair_id")["split"].sort_index())
    )
    assert recorded.groupby("depmap_id")["split"].nunique().max() == 1
    predictions = pd.read_csv(output / "adaptation/predictions.csv")
    metrics = pd.read_csv(output / "adaptation/metrics.csv")
    selection = json.loads((output / "adaptation/selection.json").read_text())
    target_rows = prism.merge(labels[["pair_id", "split"]], on="pair_id", validate="one_to_one")
    source_paths = {
        "RF_targets": baseline
        / "fits"
        / setup["case"]
        / "RF_targets_expression_pca/predictions.csv",
        "RF_morgan": baseline
        / "fits"
        / setup["case"]
        / "RF_morgan_expression_pca/predictions.csv",
        "ensemble": output / "ensemble" / setup["case"] / "predictions.csv",
        "neural": output / "neural" / setup["case"] / "predictions.csv",
    }
    for model, path in source_paths.items():
        source = (
            pd.read_csv(path)
            .set_index("pair_id")
            .reindex(target_rows.pair_id)
            .y_pred.to_numpy()
        )
        y = target_rows[case["target"]].to_numpy()
        chemistry = target_rows.chemical_id.to_numpy()
        train = target_rows.split.eq("train").to_numpy()
        val = target_rows.split.eq("validation").to_numpy()
        development = target_rows.split.isin(["train", "validation"]).to_numpy()
        candidates = []
        for shrinkage in [None, *setup["shrinkage"]]:
            adapter = fit_adapter(y[train], source[train], chemistry[train], shrinkage)
            prediction = apply_adapter(adapter, source[val], chemistry[val])
            candidates.append(np.sqrt(np.mean((prediction - y[val]) ** 2)))
        best = int(np.argmin(candidates))
        close(candidates, [c["validation_rmse"] for c in selection[model]["candidates"]])
        expected_shrinkage = [None, *setup["shrinkage"]][best]
        assert expected_shrinkage == selection[model]["selected"]["shrinkage"]
        expected_adapter = fit_adapter(
            y[development], source[development], chemistry[development], expected_shrinkage
        )
        recorded_adapter = json.loads(
            (output / "adaptation" / f"{model}_adapter.json").read_text()
        )
        close(
            apply_adapter(expected_adapter, source, chemistry),
            apply_adapter(recorded_adapter, source, chemistry),
        )
    for (model, method), data in predictions.groupby(["model", "method"]):
        cal, test = [data.loc[data.subset.eq(s)] for s in ["calibration", "test"]]
        assert set(cal.pair_id) == set(
            recorded.loc[recorded.split.eq("calibration"), "pair_id"]
        )
        assert set(test.pair_id) == set(recorded.loc[recorded.split.eq("test"), "pair_id"])
        radius = conformal_radius(
            (cal.y_true - cal.y_pred).to_numpy(), cal.depmap_id.to_numpy(), setup["alpha"]
        )
        stats = interval_diagnostics(
            test.y_true.to_numpy(),
            test.y_pred.to_numpy(),
            test.depmap_id.to_numpy(),
            radius,
            (0, 1),
        )
        saved = metrics.loc[metrics.model.eq(model) & metrics.method.eq(method)].iloc[0]
        for key, value in stats.items():
            close(value, saved[key])
        if method == "prism_supervised_adaptation":
            adapter = json.loads((output / "adaptation" / f"{model}_adapter.json").read_text())
            source = (
                predictions.loc[
                    predictions.model.eq(model)
                    & predictions.method.eq("source_only_point_predictor")
                    & predictions.subset.eq("test")
                ]
                .set_index("pair_id")
                .reindex(test.pair_id)
            )
            close(
                apply_adapter(adapter, source.y_pred.to_numpy(), test.chemical_id.to_numpy()),
                test.y_pred,
            )
    result = {
        "status": "verified",
        "stable_reference": original,
        "cases": cases,
        "neural_models_reloaded": restored_count,
        "selected_test_reload_max_delta": maximum_delta,
        "cold_test_pairs_per_model": 149364,
        "adaptation_cells_disjoint": True,
        "adaptation_test_cells": int(
            recorded.loc[recorded.split.eq("test"), "depmap_id"].nunique()
        ),
        "manifest_sha256": sha256(output / "manifest.json"),
    }
    write_json(Path("data/reports/improvement_v1_validation.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
