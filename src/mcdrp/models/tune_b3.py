"""Validation-based hyperparameter tuning for B3 MLP baseline."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mcdrp.features.expression import build_cell_features
from mcdrp.features.fingerprints import build_fingerprint_matrix
from mcdrp.metrics import regression_metrics
from mcdrp.models.baseline_b1 import DEFAULT_SPLITS, EXPRESSION_PATH
from mcdrp.models.baseline_b3 import (
    MLPFit,
    fit_mlp_regressor,
    fit_scaler,
    predict_mlp,
    prepare_split_features,
)

logger = logging.getLogger(__name__)


DEFAULT_GRID = {
    "hidden_layer_sizes": ((256,), (512, 128)),
    "alpha": (0.0001, 0.001),
    "learning_rate_init": (0.001, 0.0005),
}


def iter_param_grid(grid: dict[str, tuple[Any, ...]]) -> list[dict[str, Any]]:
    """Expand a small parameter grid."""

    keys = list(grid)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(grid[key] for key in keys))
    ]


def serializable_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert tuple-valued parameters into JSON-friendly lists."""

    output = dict(params)
    output["hidden_layer_sizes"] = list(output["hidden_layer_sizes"])
    return output


def metric_row(
    params: dict[str, Any],
    split_name: str,
    subset: str,
    rows: pd.DataFrame,
    predictions: Any,
    target: str,
    fit: MLPFit,
    fit_seconds: float | None = None,
) -> dict[str, Any]:
    """Create one metrics row for a tuned MLP candidate."""

    metrics = regression_metrics(
        rows[target].to_numpy(),
        predictions,
        cell_keys=rows["depmap_id"].tolist(),
        drug_keys=rows["drug_id"].tolist(),
    )
    row = {
        "model": "mlp",
        "params": json.dumps(serializable_params(params), sort_keys=True),
        "split_name": split_name,
        "subset": subset,
        "n_rows": int(len(rows)),
        "n_cell_lines": int(rows["depmap_id"].nunique()),
        "n_drugs": int(rows["drug_id"].nunique()),
        "backend": fit.backend,
        "device": fit.device,
        "n_iter": fit.n_iter,
        "loss": fit.loss,
        **metrics,
    }
    if fit_seconds is not None:
        row["fit_seconds"] = fit_seconds
    return row


def tune_b3_for_split(
    cohort: pd.DataFrame,
    assignments: pd.DataFrame,
    fp_matrix: Any,
    cell_feature_map: dict[str, Any],
    *,
    split_name: str,
    target: str,
    batch_size: int,
    max_iter: int,
    early_stopping: bool,
    validation_fraction: float,
    random_state: int,
    device: str,
) -> list[dict[str, Any]]:
    """Tune B3 on validation RMSE and evaluate the selected config on test."""

    subsets = fit_scaler(
        prepare_split_features(cohort, assignments, fp_matrix, cell_feature_map)
    )
    train_rows, x_train = subsets["train"]
    val_rows, x_val = subsets["validation"]
    test_rows, x_test = subsets["test"]
    y_train = train_rows[target].to_numpy()

    rows: list[dict[str, Any]] = []
    fitted: list[tuple[dict[str, Any], MLPFit, float]] = []
    for params in iter_param_grid(DEFAULT_GRID):
        t0 = time.time()
        fit = fit_mlp_regressor(
            x_train,
            y_train,
            hidden_layer_sizes=params["hidden_layer_sizes"],
            alpha=params["alpha"],
            batch_size=batch_size,
            learning_rate_init=params["learning_rate_init"],
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            random_state=random_state,
            device=device,
        )
        elapsed = time.time() - t0

        val_predictions = predict_mlp(fit, x_val)
        row = metric_row(
            params,
            split_name,
            "validation",
            val_rows,
            val_predictions,
            target,
            fit,
            fit_seconds=elapsed,
        )
        rows.append(row)
        fitted.append((params, fit, row["rmse"]))
        logger.info(
            "MLP %s validation RMSE %.3f fitted in %.1fs on %s/%s",
            serializable_params(params),
            row["rmse"],
            elapsed,
            fit.backend,
            fit.device,
        )

    best_params, best_fit, _rmse = min(fitted, key=lambda item: item[2])
    test_predictions = predict_mlp(best_fit, x_test)
    test_row = metric_row(
        best_params,
        split_name,
        "test",
        test_rows,
        test_predictions,
        target,
        best_fit,
    )
    test_row["selected_by_validation"] = True
    rows.append(test_row)
    return rows


def tune_b3(
    cohort_path: str | Path = "data/processed/cohort_pairs.csv",
    split_dir: str | Path = "data/processed/splits",
    expression_path: str | Path = EXPRESSION_PATH,
    output: str | Path = "results/baselines/b3_tuning_metrics.csv",
    summary: str | Path = "data/reports/b3_tuning_summary.json",
    *,
    target: str = "ln_ic50",
    n_components: int = 256,
    split_names: tuple[str, ...] = DEFAULT_SPLITS,
    batch_size: int = 512,
    max_iter: int = 120,
    early_stopping: bool = True,
    validation_fraction: float = 0.1,
    random_state: int = 42,
    device: str = "auto",
) -> pd.DataFrame:
    """Tune B3 hyperparameters for all requested splits."""

    cohort = pd.read_csv(cohort_path)
    split_dir = Path(split_dir)

    logger.info("Building Morgan fingerprint matrix for %d rows ...", len(cohort))
    fp_matrix = build_fingerprint_matrix(cohort["canonical_smiles"])
    logger.info("Fingerprint matrix shape: %s", fp_matrix.shape)

    all_rows: list[dict[str, Any]] = []
    for split_name in split_names:
        logger.info("=== Tuning B3 split: %s ===", split_name)
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
            tune_b3_for_split(
                cohort,
                assignments,
                fp_matrix,
                cell_feature_map,
                split_name=split_name,
                target=target,
                batch_size=batch_size,
                max_iter=max_iter,
                early_stopping=early_stopping,
                validation_fraction=validation_fraction,
                random_state=random_state,
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
        "batch_size": batch_size,
        "max_iter": max_iter,
        "early_stopping": early_stopping,
        "validation_fraction": validation_fraction,
        "device_requested": device,
        "grid": {
            key: [list(value) if isinstance(value, tuple) else value for value in values]
            for key, values in DEFAULT_GRID.items()
        },
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
                "backend": best["backend"],
                "device": best["device"],
                "n_iter": int(best["n_iter"]),
                "loss": float(best["loss"]),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description="Tune B3 MLP baseline.")
    parser.add_argument("--cohort", default="data/processed/cohort_pairs.csv")
    parser.add_argument("--split-dir", default="data/processed/splits")
    parser.add_argument("--expression", default=EXPRESSION_PATH)
    parser.add_argument("--output", default="results/baselines/b3_tuning_metrics.csv")
    parser.add_argument("--summary", default="data/reports/b3_tuning_summary.json")
    parser.add_argument("--target", default="ln_ic50")
    parser.add_argument("--n-components", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable sklearn internal early stopping.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Training device. auto uses CUDA when PyTorch can see a GPU.",
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
    metrics = tune_b3(
        cohort_path=args.cohort,
        split_dir=args.split_dir,
        expression_path=args.expression,
        output=args.output,
        summary=args.summary,
        target=args.target,
        n_components=args.n_components,
        split_names=tuple(args.splits),
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        early_stopping=not args.no_early_stopping,
        validation_fraction=args.validation_fraction,
        random_state=args.random_state,
        device=args.device,
    )
    print(f"Wrote B3 tuning metrics to {args.output}")
    print(metrics.sort_values(["split_name", "subset", "rmse"]).to_string())


if __name__ == "__main__":
    main()
