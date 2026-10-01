"""Bounded exploratory improvements using the immutable stable-v1 partitions.

Run stages separately with --stage ensemble|neural|adaptation|report|all.
No output in results/stable_v1 is modified. Interrupted neural runs resume at
completed cases, never by changing candidates in response to test outcomes.
"""

from __future__ import annotations

# ruff: noqa: E402 -- CUDA determinism and BLAS limits must precede numerical imports.
import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "8")

import joblib
import numpy as np
import pandas as pd

from mcdrp.experiments.stable_v1 import inventory, sha256, verify_run, write_json
from mcdrp.metrics import regression_metrics
from mcdrp.models.improvement_core import (
    apply_adapter,
    cell_partition,
    fit_adapter,
    select_ensemble,
)
from mcdrp.results.metrics_report import build_metrics_report
from mcdrp.results.predictions import prediction_frame, write_predictions
from mcdrp.results.uncertainty import conformal_radius, interval_diagnostics, paired_bootstrap
from mcdrp.splits.validation import selection_fingerprint, validate_assignments


def case_inputs(baseline: Path, name: str) -> tuple[dict, pd.DataFrame]:
    config = json.loads((baseline / "cases" / f"{name}.json").read_text())
    cohort = pd.read_csv(config["cohort"], low_memory=False)
    labels = pd.read_csv(config["assignments"])
    validate_assignments(cohort, labels)
    rows = cohort.merge(labels, on="pair_id", validate="one_to_one")
    if config["split_name"] == "cold_drug":
        sets = [set(g["chemical_id"]) for _, g in rows.groupby("split")]
        if any(a & b for i, a in enumerate(sets) for b in sets[i + 1 :]):
            raise ValueError("Chemical identities overlap across subsets.")
    elif (
        rows.loc[rows["split"].isin(["train", "validation"]), "study"].ne("gdsc2").any()
        or rows.loc[rows["split"].eq("test"), "study"].ne("prism").any()
    ):
        raise ValueError("Source-only training or target-only evaluation violated.")
    return config, rows


def aligned_prediction(path: Path, rows: pd.DataFrame) -> np.ndarray:
    prediction = pd.read_csv(path)
    if prediction["pair_id"].duplicated().any() or set(prediction["pair_id"]) != set(
        rows.loc[rows["split"].isin(["validation", "test"]), "pair_id"]
    ):
        raise ValueError("Prediction coverage does not match validation and test.")
    aligned = rows[["pair_id", "split"]].merge(
        prediction, on="pair_id", how="left", validate="one_to_one"
    )
    scored = aligned["split"].isin(["validation", "test"])
    if not aligned.loc[scored, "subset"].eq(aligned.loc[scored, "split"]).all():
        raise ValueError("Prediction subsets do not match assignments.")
    if not np.isfinite(aligned.loc[scored, ["y_true", "y_pred"]].to_numpy()).all():
        raise ValueError("Evaluation predictions must be finite.")
    return aligned["y_pred"].to_numpy()


def score_case(output: Path, rows: pd.DataFrame, case: dict, frames: list) -> None:
    predictions = write_predictions(frames, output / "predictions.csv")
    metrics = build_metrics_report(
        predictions,
        rows.drop(columns="split"),
        Path(case["assignments"]).parent,
        target=case["target"],
    )
    metrics.to_csv(output / "metrics.csv", index=False)


def ensemble_stage(config: dict, baseline: Path, output: Path) -> None:
    for name in config["cases"]:
        destination = output / "ensemble" / name
        if (destination / "selection.json").exists():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        case, rows = case_inputs(baseline, name)
        matrix = np.column_stack(
            [
                aligned_prediction(baseline / "fits" / name / member / "predictions.csv", rows)
                for member in config["ensemble_members"]
            ]
        )
        val = rows["split"].eq("validation").to_numpy()
        selected = select_ensemble(
            rows.loc[val, case["target"]].to_numpy(), matrix[val], config["ensemble_grid_step"]
        )
        selected["members"] = config["ensemble_members"]
        selected["selection_fingerprint"] = selection_fingerprint(
            rows.drop(columns="split"),
            rows[["pair_id", "split"]],
            target=case["target"],
            context={
                "members": config["ensemble_members"],
                "step": config["ensemble_grid_step"],
            },
        )
        frames = []
        for subset in ["validation", "test"]:
            mask = rows["split"].eq(subset).to_numpy()
            prediction = matrix[mask] @ np.array(selected["weights"])
            frames.append(
                prediction_frame(
                    rows.loc[mask],
                    rows.loc[mask, case["target"]].to_numpy(),
                    prediction,
                    stage="E1",
                    model="convex_ensemble",
                    split_name=case["split_name"],
                    subset=subset,
                )
            )
        score_case(destination, rows, case, frames)
        write_json(destination / "selection.json", selected)
        print(f"Ensemble {name}: {selected['weights']}", flush=True)


def neural_stage(config: dict, baseline: Path, output: Path) -> None:
    from mcdrp.models.residual_neural import feature_tables, fit, predict, restore

    for name in config["cases"]:
        destination = output / "neural" / name
        if (destination / "selection.json").exists():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        case, rows = case_inputs(baseline, name)
        features = joblib.load(case["features"])
        train, val = (rows.loc[rows["split"].eq(s)] for s in ["train", "validation"])
        candidates = []
        for candidate_id, candidate in enumerate(config["neural"]["candidates"]):
            tables = feature_tables(rows, features, candidate["blocks"], train)
            predictions = []
            for seed in config["neural"]["seeds"]:
                artifact_path = destination / f"candidate_{candidate_id}_seed_{seed}.joblib"
                if artifact_path.exists():
                    artifact = joblib.load(artifact_path)
                else:
                    print(f"Neural {name}, candidate {candidate_id}, seed {seed}", flush=True)
                    artifact, history = fit(
                        train,
                        val,
                        tables,
                        target=case["target"],
                        candidate=candidate,
                        config=config["neural"],
                        seed=seed,
                    )
                    pd.DataFrame(history).to_csv(
                        artifact_path.with_suffix(".history.csv"), index=False
                    )
                    joblib.dump(artifact, artifact_path, compress=3)
                model = restore(artifact, config["neural"]["device"])
                prediction = predict(
                    model,
                    val,
                    tables,
                    device=config["neural"]["device"],
                    mean=artifact["mean"],
                    scale=artifact["scale"],
                )
                restored = restore(joblib.load(artifact_path), config["neural"]["device"])
                repeated = predict(
                    restored,
                    val,
                    tables,
                    device=config["neural"]["device"],
                    mean=artifact["mean"],
                    scale=artifact["scale"],
                )
                np.testing.assert_array_equal(prediction, repeated)
                predictions.append(prediction)
                print(
                    f"  best epoch {artifact['best_epoch']}, validation RMSE "
                    f"{artifact['validation_rmse']:.5f}; reload delta 0",
                    flush=True,
                )
            average = np.mean(predictions, axis=0)
            if case["target"] == "auc":
                average = np.clip(average, 0, 1)
            candidates.append(
                {
                    "candidate": candidate_id,
                    "validation_rmse": float(
                        np.sqrt(np.mean((average - val[case["target"]].to_numpy()) ** 2))
                    ),
                }
            )
        best = min(candidates, key=lambda c: c["validation_rmse"])["candidate"]
        candidate = config["neural"]["candidates"][best]
        frames = []
        for subset in ["validation", "test"]:
            data = rows.loc[rows["split"].eq(subset)]
            predictions = []
            for seed in config["neural"]["seeds"]:
                artifact = joblib.load(destination / f"candidate_{best}_seed_{seed}.joblib")
                tables = feature_tables(
                    rows, features, candidate["blocks"], train, artifact["scalers"]
                )
                model = restore(artifact, config["neural"]["device"])
                prediction = predict(
                    model,
                    data,
                    tables,
                    device=config["neural"]["device"],
                    mean=artifact["mean"],
                    scale=artifact["scale"],
                )
                repeated = predict(
                    restore(artifact, config["neural"]["device"]),
                    data,
                    tables,
                    device=config["neural"]["device"],
                    mean=artifact["mean"],
                    scale=artifact["scale"],
                )
                np.testing.assert_array_equal(prediction, repeated)
                predictions.append(prediction)
                pd.DataFrame({"pair_id": data["pair_id"], "y_pred": prediction}).to_csv(
                    destination / f"selected_seed_{seed}_{subset}.csv", index=False
                )
            average = np.mean(predictions, axis=0)
            if case["target"] == "auc":
                average = np.clip(average, 0, 1)
            frames.append(
                prediction_frame(
                    data,
                    data[case["target"]].to_numpy(),
                    average,
                    stage="N1",
                    model="residual_towers_seed_ensemble",
                    split_name=case["split_name"],
                    subset=subset,
                )
            )
        score_case(destination, rows, case, frames)
        write_json(
            destination / "selection.json",
            {
                "selected_candidate": best,
                "candidates": candidates,
                "seeds": config["neural"]["seeds"],
                "selection_fingerprint": selection_fingerprint(
                    rows.drop(columns="split"),
                    rows[["pair_id", "split"]],
                    target=case["target"],
                    context=config["neural"],
                ),
                "reload_max_delta": 0.0,
            },
        )


def adaptation_stage(config: dict, baseline: Path, output: Path) -> None:
    setup = config["adaptation"]
    case, full_rows = case_inputs(baseline, setup["case"])
    rows = full_rows.loc[full_rows["split"].eq("test")].drop(columns="split").copy()
    assignments = cell_partition(
        rows, seed=setup["seed"], fractions=setup["fractions"], roles=setup["roles"]
    )
    rows = rows.merge(assignments[["pair_id", "split"]], on="pair_id", validate="one_to_one")
    destination = output / "adaptation"
    destination.mkdir(exist_ok=True)
    assignments.to_csv(destination / "assignments.csv", index=False)
    models = {
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
    reports, frames, selection = [], [], {}
    for name, path in models.items():
        if not path.exists():
            raise ValueError(f"Missing completed upstream model: {path}")
        prediction = (
            pd.read_csv(path).set_index("pair_id")["y_pred"].reindex(rows["pair_id"]).to_numpy()
        )
        train, val, dev, cal, test = [
            rows["split"].isin(labels).to_numpy()
            for labels in [
                ["train"],
                ["validation"],
                ["train", "validation"],
                ["calibration"],
                ["test"],
            ]
        ]
        y, chemistry = (
            rows[case["target"]].to_numpy(),
            rows["chemical_id"].astype(str).to_numpy(),
        )
        candidates = []
        for shrinkage in [None, *setup["shrinkage"]]:
            adapter = fit_adapter(y[train], prediction[train], chemistry[train], shrinkage)
            p = apply_adapter(adapter, prediction[val], chemistry[val])
            candidates.append(
                {
                    "shrinkage": shrinkage,
                    "validation_rmse": float(np.sqrt(np.mean((p - y[val]) ** 2))),
                }
            )
        best = min(candidates, key=lambda c: c["validation_rmse"])
        adapter = fit_adapter(y[dev], prediction[dev], chemistry[dev], best["shrinkage"])
        write_json(destination / f"{name}_adapter.json", adapter)
        restored = json.loads((destination / f"{name}_adapter.json").read_text())
        adapted = apply_adapter(adapter, prediction, chemistry)
        np.testing.assert_array_equal(adapted, apply_adapter(restored, prediction, chemistry))
        selection[name] = {
            "candidates": candidates,
            "selected": best,
            "reload_max_delta": 0.0,
            "selection_fingerprint": selection_fingerprint(
                rows,
                assignments[["pair_id", "split"]],
                target=case["target"],
                context={"model": name, "shrinkage": setup["shrinkage"]},
            ),
        }
        for method, p in [
            ("source_only_point_predictor", prediction),
            ("prism_supervised_adaptation", adapted),
        ]:
            # Both interval variants use held-out PRISM labels, not zero-shot calibration.
            radius = conformal_radius(
                y[cal] - p[cal], rows.loc[cal, "depmap_id"].to_numpy(), setup["alpha"]
            )
            metrics = regression_metrics(
                y[test],
                p[test],
                cell_keys=rows.loc[test, "depmap_id"].tolist(),
                drug_keys=rows.loc[test, "drug_id"].tolist(),
            )
            reports.append(
                {
                    "model": name,
                    "method": method,
                    **metrics,
                    "interval_calibration_domain": "PRISM",
                    "alpha": setup["alpha"],
                    "radius": radius,
                    **interval_diagnostics(
                        y[test], p[test], rows.loc[test, "depmap_id"].to_numpy(), radius, (0, 1)
                    ),
                    "n_calibration_cells": int(rows.loc[cal, "depmap_id"].nunique()),
                }
            )
            for subset, mask in [("calibration", cal), ("test", test)]:
                frames.append(
                    pd.DataFrame(
                        {
                            "pair_id": rows.loc[mask, "pair_id"],
                            "depmap_id": rows.loc[mask, "depmap_id"],
                            "chemical_id": chemistry[mask],
                            "model": name,
                            "method": method,
                            "subset": subset,
                            "y_true": y[mask],
                            "y_pred": p[mask],
                        }
                    )
                )
        delta = paired_bootstrap(
            y[test], adapted[test], prediction[test], rows.loc[test, "depmap_id"].to_numpy()
        )
        selection[name]["paired_cell_bootstrap"] = delta
    pd.DataFrame(reports).to_csv(destination / "metrics.csv", index=False)
    pd.concat(frames).to_csv(destination / "predictions.csv", index=False)
    write_json(destination / "selection.json", selection)


def report_stage(config: dict, baseline: Path, output: Path) -> None:
    metrics, diagnostics, bootstrap = [], [], []
    for name in config["cases"]:
        case, rows = case_inputs(baseline, name)
        sources = {
            "B0": baseline / "fits" / name / "B0_mean_effects",
            "RF_targets": baseline / "fits" / name / "RF_targets_expression_pca",
            "RF_morgan": baseline / "fits" / name / "RF_morgan_expression_pca",
            "ensemble": output / "ensemble" / name,
            "neural": output / "neural" / name,
        }
        test = rows.loc[rows["split"].eq("test")]
        prediction_map = {}
        for model, path in sources.items():
            report = pd.read_csv(path / "metrics.csv")
            scored = report.loc[report["subset"].eq("test")].to_dict("records")
            if len(scored) != 1:
                raise ValueError("Expected one complete test metric row.")
            metrics.append({"case": name, "label": model, **scored[0]})
            prediction = pd.read_csv(path / "predictions.csv").set_index("pair_id")["y_pred"]
            prediction_map[model] = prediction.reindex(test["pair_id"]).to_numpy()
        groups = test["chemical_id"] if name.startswith("cold_") else test["depmap_id"]
        for model in ["ensemble", "neural"]:
            bootstrap.append(
                {
                    "case": name,
                    "label": model,
                    "reference": "RF_targets",
                    "group": "chemical_id" if name.startswith("cold_") else "depmap_id",
                    **paired_bootstrap(
                        test[case["target"]].to_numpy(),
                        prediction_map[model],
                        prediction_map["RF_targets"],
                        groups.to_numpy(),
                    ),
                }
            )
        if name.startswith("cold_"):
            features = joblib.load(case["features"])
            drugs = sorted(test["drug_id"].astype(str).unique())
            training_drugs = sorted(
                rows.loc[rows["split"].eq("train"), "drug_id"].astype(str).unique()
            )
            fp = np.vstack([features["morgan"][d] for d in drugs]).astype(float)
            training_fp = np.vstack([features["morgan"][d] for d in training_drugs]).astype(
                float
            )
            intersection = fp @ training_fp.T
            union = fp.sum(axis=1)[:, None] + training_fp.sum(axis=1)[None, :] - intersection
            similarity = np.divide(
                intersection, union, out=np.zeros_like(union), where=union > 0
            ).max(axis=1)
            details = test.copy()
            details["error"] = prediction_map["RF_targets"] - test[case["target"]].to_numpy()
            for drug, sim in zip(drugs, similarity, strict=True):
                data = details.loc[details["drug_id"].astype(str).eq(drug)]
                diagnostics.append(
                    {
                        "case": name,
                        "drug_id": drug,
                        "name": data["gdsc_drug_name"].iloc[0],
                        "chemical_id": data["chemical_id"].iloc[0],
                        "n_pairs": len(data),
                        "nearest_train_tanimoto": float(sim),
                        "known_target_tokens": int(np.sum(features["targets"][drug])),
                        "rmse": float(np.sqrt(np.mean(data["error"] ** 2))),
                        "bias": float(data["error"].mean()),
                        "centered_rmse": float(data["error"].std(ddof=0)),
                    }
                )
    snapshot = {
        "scope": config["scope"],
        "config": config,
        "metrics": metrics,
        "drug_diagnostics": diagnostics,
        "bootstrap": bootstrap,
        "adaptation": pd.read_csv(output / "adaptation/metrics.csv").to_dict("records"),
        "adaptation_selection": json.loads((output / "adaptation/selection.json").read_text()),
        "environment": json.loads((output / "environment.json").read_text()),
        "ensemble_selections": {
            name: json.loads((output / "ensemble" / name / "selection.json").read_text())
            for name in config["cases"]
        },
        "neural_selections": {
            name: json.loads((output / "neural" / name / "selection.json").read_text())
            for name in config["cases"]
        },
    }

    # Correlation metrics can legitimately be undefined. JSON uses null, never NaN.
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    snapshot = clean(snapshot)
    write_json(output / "summary.json", snapshot)
    write_json(Path("data/reports/improvement_v1_summary.json"), snapshot)
    write_json(
        output / "manifest.json",
        {
            "status": "complete",
            "config_sha256": sha256(output / "config.json"),
            "files": inventory(output),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/improvement_v1.json")
    parser.add_argument(
        "--stage", choices=["ensemble", "neural", "adaptation", "report", "all"], default="all"
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    baseline, output = Path(config["baseline_root"]), Path(config["output_root"])
    if not output.exists():
        # Validate the frozen reference before any experimental fit.
        verify_run(baseline)
        output.mkdir(parents=True)
        write_json(output / "config.json", config)
        versions = {
            p: importlib.metadata.version(p)
            for p in ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "torch"]
        }
        write_json(
            output / "environment.json",
            {
                "python": platform.python_version(),
                "packages": versions,
                "stable_environment": False,
            },
        )
    elif json.loads((output / "config.json").read_text()) != config:
        raise ValueError("Existing experiment configuration differs; choose a new output root.")
    source = [
        Path(__file__),
        Path("src/mcdrp/models/improvement_core.py"),
        Path("src/mcdrp/models/residual_neural.py"),
    ]
    hashes = {str(p): sha256(p) for p in source}
    record = output / f"{args.stage}_source.json"
    if record.exists() and json.loads(record.read_text()) != hashes:
        raise ValueError("Source changed during a stage; start a separate experimental run.")
    write_json(record, hashes)
    for stage, function in [
        ("ensemble", ensemble_stage),
        ("neural", neural_stage),
        ("adaptation", adaptation_stage),
        ("report", report_stage),
    ]:
        if args.stage in [stage, "all"]:
            function(config, baseline, output)


if __name__ == "__main__":
    main()
