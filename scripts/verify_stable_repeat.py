"""Independently refit two release models from raw expression and drug features.

Run in a second locked environment. This check repeats fixed validation-selected
RF settings and XGBoost early stopping; it never adjusts settings using test scores.
It is a targeted numerical diagnostic, not a repeat of all 60 release fits.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/stable_v1"))
    parser.add_argument("--inputs-root", type=Path, default=Path("."))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--blas-threads", type=int, default=16)
    args = parser.parse_args()
    root, report_path = args.output.resolve(), args.report.resolve()
    if report_path.is_relative_to(root) or report_path.exists():
        raise ValueError("Choose a new diagnostic report outside the frozen run.")
    if args.blas_threads < 1:
        parser.error("--blas-threads must be positive")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(args.blas_threads)

    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    from mcdrp.experiments.stable_v1 import check_environment, check_inputs, write_json
    from mcdrp.features.expression import (
        fit_expression_pipeline,
        load_expression_matrix,
        transform_expression,
    )
    from mcdrp.features.fingerprints import build_fingerprint_matrix
    from mcdrp.features.targets import build_target_matrix
    from mcdrp.models.baseline_b2 import fit_xgboost, predict_xgboost
    from mcdrp.models.stable_core import assemble, replay

    config = json.loads((root / "config.json").read_text())
    environment = check_environment(config)
    inputs = check_inputs(config, args.inputs_root.resolve())
    case = json.loads((root / "cases/cold_drug_fold_0.json").read_text())
    cohort = pd.read_csv(case["cohort"], low_memory=False)
    rows = cohort.merge(pd.read_csv(case["assignments"]), on="pair_id", validate="one_to_one")
    train = rows.loc[rows["split"].eq("train")]
    validation = rows.loc[rows["split"].eq("validation")]
    test = rows.loc[rows["split"].eq("test")]
    expression = load_expression_matrix(inputs["expression"], set(cohort["depmap_id"]))
    pipeline = fit_expression_pipeline(
        expression.loc[expression.index.isin(set(train["depmap_id"]))],
        n_components=256,
        random_state=config["seed"],
    )
    matrix = transform_expression(pipeline, expression)
    catalog = cohort.drop_duplicates("drug_id")
    _, target_matrix = build_target_matrix(catalog, set(train["drug_id"].astype(str)))
    keys = catalog["drug_id"].astype(str).tolist()
    features = {
        "cells": dict(zip(expression.index, matrix, strict=True)),
        "targets": dict(zip(keys, target_matrix, strict=True)),
        "morgan": dict(
            zip(keys, build_fingerprint_matrix(catalog["canonical_smiles"]), strict=True)
        ),
    }
    original_features = joblib.load(case["features"])
    original_matrix = np.vstack([original_features["cells"][k] for k in expression.index])
    feature_delta = float(np.max(np.abs(matrix - original_matrix)))
    if feature_delta > config["projection_absolute_tolerance"]:
        raise ValueError(f"Independent preprocessing exceeds the tolerance: {feature_delta}")
    for block in ("targets", "morgan"):
        for key in keys:
            np.testing.assert_array_equal(features[block][key], original_features[block][key])
    checks = []
    for stage, name, block in (
        ("RF", "morgan_expression_pca", "morgan"),
        ("B6", "targets_expression_pca", "targets"),
    ):
        directory = Path(case["output"]) / f"{stage}_{name}"
        fit = json.loads((directory / "fit.json").read_text())
        x_train, x_val = assemble(train, features, block), assemble(validation, features, block)
        if stage == "RF":
            keep = np.ptp(x_train, axis=0) > 0
            model = RandomForestRegressor(**fit["params"])
            model.fit(x_train[:, keep], train[case["target"]].to_numpy())
            prediction = model.predict(assemble(test, features, block)[:, keep])
            best_iteration = None
        else:
            model, actual_device = fit_xgboost(
                config["xgb_params"],
                x_train,
                train[case["target"]].to_numpy(),
                x_val,
                validation[case["target"]].to_numpy(),
                device="cuda",
            )
            if actual_device != "cuda":
                raise ValueError("The independent repeat did not use CUDA.")
            best_iteration = model.best_iteration
            if best_iteration != fit["best_iteration"]:
                raise ValueError("The repeated early-stopping iteration differs.")
            prediction = predict_xgboost(model, assemble(test, features, block))
        reference = replay(joblib.load(directory / "model.joblib"), test, original_features)
        delta = float(np.max(np.abs(prediction - reference)))
        np.testing.assert_allclose(prediction, reference, rtol=0, atol=1e-7)
        checks.append(
            {
                "model": f"{stage}_{name}",
                "test_pairs": len(test),
                "best_iteration": best_iteration,
                "max_prediction_delta": delta,
                "status": "verified",
            }
        )
        print(json.dumps(checks[-1]), flush=True)
    write_json(
        report_path,
        {
            "status": "verified",
            "purpose": "Independent environment, raw preprocessing and fixed model recipe",
            "case": case["name"],
            "blas_threads": args.blas_threads,
            "max_feature_delta": feature_delta,
            "projection_tolerance": config["projection_absolute_tolerance"],
            "prediction_tolerance": 1e-7,
            "models": checks,
            "environment": environment,
        },
    )


if __name__ == "__main__":
    main()
