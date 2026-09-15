"""Validation-based hyperparameter tuning for B2 XGBoost baseline."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from mcdrp.features.expression import build_cell_features
from mcdrp.features.fingerprints import build_fingerprint_matrix
from mcdrp.metrics import regression_metrics
from mcdrp.models.baseline_b1 import DEFAULT_SPLITS, EXPRESSION_PATH, assemble_features
from mcdrp.models.baseline_b2 import fit_xgboost, predict_xgboost, xgb_device_params

logger = logging.getLogger(__name__)


DEFAULT_GRID = {
    "max_depth": (4, 6),
    "learning_rate": (0.03, 0.05),
    "subsample": (0.8, 1.0),
    "colsample_bytree": (0.8, 1.0),
}


def iter_param_grid(grid: dict[str, tuple[Any, ...]]) -> list[dict[str, Any]]:
    """Expand a small parameter grid."""

    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(grid[k] for k in keys))]


def subset_features(
    cohort: pd.DataFrame,
    data: pd.DataFrame,
    fp_matrix: np.ndarray,
    cell_feature_map: dict[str, np.ndarray],
    subset: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build features for one subset."""

    pair_id_to_idx = {pid: idx for idx, pid in enumerate(cohort["pair_id"])}
    rows = data.loc[data["split"].eq(subset)].copy()
    indices = np.array([pair_id_to_idx[pid] for pid in rows["pair_id"]])
    return rows, assemble_features(rows, fp_matrix, cell_feature_map, row_indices=indices)


def metric_row(
    params: dict[str, Any],
    split_name: str,
    subset: str,
    rows: pd.DataFrame,
    predictions: np.ndarray,
    target: str,
    best_iteration: int,
    fit_seconds: float | None = None,
) -> dict[str, Any]:
    """Create one metrics row."""

    metrics = regression_metrics(rows[target].to_numpy(), predictions)
    row = {
        "model": "xgboost",
        "params": json.dumps(params, sort_keys=True),
        "split_name": split_name,
        "subset": subset,
        "n_rows": int(len(rows)),
        "n_cell_lines": int(rows["depmap_id"].nunique()),
        "n_drugs": int(rows["drug_id"].nunique()),
        "best_iteration": int(best_iteration),
        **metrics,
    }
    if fit_seconds is not None:
        row["fit_seconds"] = fit_seconds
    return row


def tune_b2_for_split(
    cohort: pd.DataFrame,
    assignments: pd.DataFrame,
    fp_matrix: np.ndarray,
    cell_feature_map: dict[str, np.ndarray],
    *,
    split_name: str,
    target: str,
    n_estimators: int,
    early_stopping_rounds: int,
    device: str,
) -> list[dict[str, Any]]:
    """Tune XGBoost on validation RMSE and evaluate best setting on test."""

    data = cohort.merge(assignments, on="pair_id", how="inner", validate="one_to_one")
    train_rows, x_train = subset_features(cohort, data, fp_matrix, cell_feature_map, "train")
    val_rows, x_val = subset_features(cohort, data, fp_matrix, cell_feature_map, "validation")
    test_rows, x_test = subset_features(cohort, data, fp_matrix, cell_feature_map, "test")
    y_train = train_rows[target].to_numpy()
    y_val = val_rows[target].to_numpy()

    rows: list[dict[str, Any]] = []
    fitted: list[tuple[dict[str, Any], xgb.XGBRegressor, float, int, str]] = []
    for params in iter_param_grid(DEFAULT_GRID):
        xgb_params = {
            **params,
            "n_estimators": n_estimators,
            "early_stopping_rounds": early_stopping_rounds,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        }
        xgb_params.update(xgb_device_params(device))
        t0 = time.time()
        model, actual_device = fit_xgboost(
            xgb_params,
            x_train,
            y_train,
            x_val,
            y_val,
            device=device,
        )
        elapsed = time.time() - t0
        predictions = predict_xgboost(model, x_val)
        row = metric_row(
            params,
            split_name,
            "validation",
            val_rows,
            predictions,
            target,
            model.best_iteration,
            fit_seconds=elapsed,
        )
        row["device"] = actual_device
        rows.append(row)
        fitted.append((params, model, row["rmse"], model.best_iteration, actual_device))
        logger.info(
            "XGBoost %s validation RMSE %.3f fitted in %.1fs",
            params,
            row["rmse"],
            elapsed,
        )

    best_params, best_model, _rmse, best_iteration, best_device = min(
        fitted,
        key=lambda item: item[2],
    )
    test_predictions = predict_xgboost(best_model, x_test)
    test_row = metric_row(
        best_params,
        split_name,
        "test",
        test_rows,
        test_predictions,
        target,
        best_iteration,
    )
    test_row["device"] = best_device
    test_row["selected_by_validation"] = True
    rows.append(test_row)
    return rows


def tune_b2(
    cohort_path: str | Path = "data/processed/cohort_pairs.csv",
    split_dir: str | Path = "data/processed/splits",
    expression_path: str | Path = EXPRESSION_PATH,
    output: str | Path = "results/baselines/b2_tuning_metrics.csv",
    summary: str | Path = "data/reports/b2_tuning_summary.json",
    *,
    target: str = "ln_ic50",
    n_components: int = 256,
    split_names: tuple[str, ...] = DEFAULT_SPLITS,
    n_estimators: int = 700,
    early_stopping_rounds: int = 40,
    device: str = "auto",
) -> pd.DataFrame:
    """Tune B2 hyperparameters for all requested splits."""

    cohort = pd.read_csv(cohort_path)
    split_dir = Path(split_dir)

    logger.info("Building Morgan fingerprint matrix for %d rows ...", len(cohort))
    fp_matrix = build_fingerprint_matrix(cohort["canonical_smiles"])
    logger.info("Fingerprint matrix shape: %s", fp_matrix.shape)

    all_rows: list[dict[str, Any]] = []
    for split_name in split_names:
        logger.info("=== Tuning B2 split: %s ===", split_name)
        assignments = pd.read_csv(split_dir / f"{split_name}.csv")
        merged = cohort.merge(assignments, on="pair_id", how="inner")
        train_ids = set(merged.loc[merged["split"].eq("train"), "depmap_id"].unique())
        pipeline, cell_feature_map = build_cell_features(
            str(expression_path),
            cohort,
            train_ids,
            n_components=n_components,
        )
        logger.info(
            "PCA explained variance: %.1f%%",
            pipeline.explained_variance_ratio_sum * 100,
        )
        all_rows.extend(
            tune_b2_for_split(
                cohort,
                assignments,
                fp_matrix,
                cell_feature_map,
                split_name=split_name,
                target=target,
                n_estimators=n_estimators,
                early_stopping_rounds=early_stopping_rounds,
                device=device,
            )
        )

    metrics = pd.DataFrame(all_rows)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)

    summary_data = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_path": str(cohort_path),
        "expression_path": str(expression_path),
        "output": str(output),
        "target": target,
        "n_pca_components": n_components,
        "n_estimators": n_estimators,
        "early_stopping_rounds": early_stopping_rounds,
        "xgb_device_requested": device,
        "grid": {key: list(value) for key, value in DEFAULT_GRID.items()},
        "splits": list(split_names),
        "best_validation": summarize_best_validation(metrics),
    }
    summary_path = Path(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_data, handle, indent=2)
    return metrics


def summarize_best_validation(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize best validation setting per split."""

    rows: list[dict[str, Any]] = []
    validation = metrics.loc[metrics["subset"].eq("validation")]
    for split_name, group in validation.groupby("split_name"):
        best = group.sort_values("rmse").iloc[0]
        rows.append(
            {
                "split_name": split_name,
                "params": best["params"],
                "validation_rmse": float(best["rmse"]),
                "validation_mae": float(best["mae"]),
                "best_iteration": int(best["best_iteration"]),
                "device": best["device"],
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description="Tune B2 XGBoost baseline.")
    parser.add_argument("--cohort", default="data/processed/cohort_pairs.csv")
    parser.add_argument("--split-dir", default="data/processed/splits")
    parser.add_argument("--expression", default=EXPRESSION_PATH)
    parser.add_argument("--output", default="results/baselines/b2_tuning_metrics.csv")
    parser.add_argument("--summary", default="data/reports/b2_tuning_summary.json")
    parser.add_argument("--target", default="ln_ic50")
    parser.add_argument("--n-components", type=int, default=256)
    parser.add_argument("--n-estimators", type=int, default=700)
    parser.add_argument("--early-stopping", type=int, default=40)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Training device. auto tries CUDA and falls back to CPU.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=list(DEFAULT_SPLITS),
    )
    return parser


def main() -> None:
    """Command-line entry point."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = build_parser().parse_args()
    metrics = tune_b2(
        cohort_path=args.cohort,
        split_dir=args.split_dir,
        expression_path=args.expression,
        output=args.output,
        summary=args.summary,
        target=args.target,
        n_components=args.n_components,
        split_names=tuple(args.splits),
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping,
        device=args.device,
    )
    print(f"Wrote B2 tuning metrics to {args.output}")
    print(metrics.sort_values(["split_name", "subset", "rmse"]).to_string())


if __name__ == "__main__":
    main()
