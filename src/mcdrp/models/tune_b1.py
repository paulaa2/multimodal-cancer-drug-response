"""Validation-based hyperparameter tuning for B1 linear baselines."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge

from mcdrp.features.expression import build_cell_features
from mcdrp.features.fingerprints import build_fingerprint_matrix
from mcdrp.metrics import regression_metrics
from mcdrp.models.baseline_b1 import DEFAULT_SPLITS, EXPRESSION_PATH, assemble_features

logger = logging.getLogger(__name__)


RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
ELASTIC_ALPHAS = (0.005, 0.01, 0.02, 0.05)
ELASTIC_L1_RATIOS = (0.3, 0.7, 0.9)


def build_subset_features(
    cohort: pd.DataFrame,
    data: pd.DataFrame,
    fp_matrix: np.ndarray,
    cell_feature_map: dict[str, np.ndarray],
    subset: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build features for one train/validation/test subset."""

    pair_id_to_idx = {pid: idx for idx, pid in enumerate(cohort["pair_id"])}
    rows = data.loc[data["split"].eq(subset)].copy()
    indices = np.array([pair_id_to_idx[pid] for pid in rows["pair_id"]])
    features = assemble_features(rows, fp_matrix, cell_feature_map, row_indices=indices)
    return rows, features


def metric_row(
    model_name: str,
    params: dict[str, Any],
    split_name: str,
    subset: str,
    rows: pd.DataFrame,
    predictions: np.ndarray,
    target: str,
) -> dict[str, Any]:
    """Create one metrics row for a tuned model."""

    metrics = regression_metrics(rows[target].to_numpy(), predictions)
    return {
        "model": model_name,
        "params": json.dumps(params, sort_keys=True),
        "split_name": split_name,
        "subset": subset,
        "n_rows": int(len(rows)),
        "n_cell_lines": int(rows["depmap_id"].nunique()),
        "n_drugs": int(rows["drug_id"].nunique()),
        **metrics,
    }


def tune_b1_for_split(
    cohort: pd.DataFrame,
    assignments: pd.DataFrame,
    fp_matrix: np.ndarray,
    cell_feature_map: dict[str, np.ndarray],
    *,
    split_name: str,
    target: str,
) -> list[dict[str, Any]]:
    """Tune B1 models on validation RMSE and evaluate selected configs on test."""

    data = cohort.merge(assignments, on="pair_id", how="inner", validate="one_to_one")
    train_rows, x_train = build_subset_features(
        cohort, data, fp_matrix, cell_feature_map, "train"
    )
    val_rows, x_val = build_subset_features(
        cohort, data, fp_matrix, cell_feature_map, "validation"
    )
    test_rows, x_test = build_subset_features(
        cohort, data, fp_matrix, cell_feature_map, "test"
    )
    y_train = train_rows[target].to_numpy()

    candidates: list[tuple[str, dict[str, Any], Any]] = []
    for alpha in RIDGE_ALPHAS:
        candidates.append(("ridge", {"alpha": alpha}, Ridge(alpha=alpha)))
    for alpha in ELASTIC_ALPHAS:
        for l1_ratio in ELASTIC_L1_RATIOS:
            candidates.append(
                (
                    "elastic_net",
                    {"alpha": alpha, "l1_ratio": l1_ratio},
                    ElasticNet(
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        max_iter=2000,
                        random_state=42,
                        selection="random",
                        copy_X=False,
                    ),
                )
            )

    rows: list[dict[str, Any]] = []
    fitted: list[tuple[str, dict[str, Any], Any, float]] = []
    for model_name, params, model in candidates:
        t0 = time.time()
        model.fit(x_train, y_train)
        elapsed = time.time() - t0
        val_predictions = model.predict(x_val)
        val_row = metric_row(
            model_name,
            params,
            split_name,
            "validation",
            val_rows,
            val_predictions,
            target,
        )
        val_row["fit_seconds"] = elapsed
        rows.append(val_row)
        fitted.append((model_name, params, model, val_row["rmse"]))
        logger.info(
            "%s %s validation RMSE %.3f fitted in %.1fs",
            model_name,
            params,
            val_row["rmse"],
            elapsed,
        )

    for model_name in ("ridge", "elastic_net"):
        best_name, best_params, best_model, _rmse = min(
            (item for item in fitted if item[0] == model_name),
            key=lambda item: item[3],
        )
        test_predictions = best_model.predict(x_test)
        test_row = metric_row(
            best_name,
            best_params,
            split_name,
            "test",
            test_rows,
            test_predictions,
            target,
        )
        test_row["selected_by_validation"] = True
        rows.append(test_row)

    return rows


def tune_b1(
    cohort_path: str | Path = "data/processed/cohort_pairs.csv",
    split_dir: str | Path = "data/processed/splits",
    expression_path: str | Path = EXPRESSION_PATH,
    output: str | Path = "results/baselines/b1_tuning_metrics.csv",
    summary: str | Path = "data/reports/b1_tuning_summary.json",
    *,
    target: str = "ln_ic50",
    n_components: int = 256,
    split_names: tuple[str, ...] = DEFAULT_SPLITS,
) -> pd.DataFrame:
    """Tune B1 hyperparameters for all requested splits."""

    cohort = pd.read_csv(cohort_path)
    split_dir = Path(split_dir)

    logger.info("Building Morgan fingerprint matrix for %d rows ...", len(cohort))
    fp_matrix = build_fingerprint_matrix(cohort["canonical_smiles"])
    logger.info("Fingerprint matrix shape: %s", fp_matrix.shape)

    all_rows: list[dict[str, Any]] = []
    for split_name in split_names:
        logger.info("=== Tuning B1 split: %s ===", split_name)
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
            tune_b1_for_split(
                cohort,
                assignments,
                fp_matrix,
                cell_feature_map,
                split_name=split_name,
                target=target,
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
        "ridge_alphas": list(RIDGE_ALPHAS),
        "elastic_alphas": list(ELASTIC_ALPHAS),
        "elastic_l1_ratios": list(ELASTIC_L1_RATIOS),
        "splits": list(split_names),
        "best_validation": summarize_best_validation(metrics),
    }
    summary_path = Path(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_data, handle, indent=2)
    return metrics


def summarize_best_validation(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize best validation setting per split/model."""

    rows: list[dict[str, Any]] = []
    validation = metrics.loc[metrics["subset"].eq("validation")]
    for (split_name, model), group in validation.groupby(["split_name", "model"]):
        best = group.sort_values("rmse").iloc[0]
        rows.append(
            {
                "split_name": split_name,
                "model": model,
                "params": best["params"],
                "validation_rmse": float(best["rmse"]),
                "validation_mae": float(best["mae"]),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description="Tune B1 Ridge/ElasticNet baselines.")
    parser.add_argument("--cohort", default="data/processed/cohort_pairs.csv")
    parser.add_argument("--split-dir", default="data/processed/splits")
    parser.add_argument("--expression", default=EXPRESSION_PATH)
    parser.add_argument("--output", default="results/baselines/b1_tuning_metrics.csv")
    parser.add_argument("--summary", default="data/reports/b1_tuning_summary.json")
    parser.add_argument("--target", default="ln_ic50")
    parser.add_argument("--n-components", type=int, default=256)
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
    metrics = tune_b1(
        cohort_path=args.cohort,
        split_dir=args.split_dir,
        expression_path=args.expression,
        output=args.output,
        summary=args.summary,
        target=args.target,
        n_components=args.n_components,
        split_names=tuple(args.splits),
    )
    print(f"Wrote B1 tuning metrics to {args.output}")
    print(metrics.sort_values(["split_name", "subset", "model", "rmse"]).to_string())


if __name__ == "__main__":
    main()
