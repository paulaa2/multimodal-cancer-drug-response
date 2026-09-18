"""Validation-based tuning for B7 pretrained drug embedding models."""

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

from mcdrp.features.drug_embeddings import DrugEmbeddingTable, load_drug_embeddings
from mcdrp.features.expression import build_cell_features
from mcdrp.features.pathways import build_pathway_features
from mcdrp.metrics import regression_metrics
from mcdrp.models.baseline_b2 import fit_xgboost, predict_xgboost, xgb_device_params
from mcdrp.models.pretrained_b7 import (
    DEFAULT_B7_MODELS,
    EXPRESSION_PATH,
    assemble_b7_features,
)
from mcdrp.splits.make_splits import DEFAULT_SPLITS

logger = logging.getLogger(__name__)

DEFAULT_GRID = {
    "max_depth": (3, 5),
    "learning_rate": (0.01, 0.025),
    "subsample": (0.8, 1.0),
    "colsample_bytree": (0.8, 1.0),
}


def iter_param_grid(grid: dict[str, tuple[Any, ...]]) -> list[dict[str, Any]]:
    """Expand a small parameter grid."""

    keys = list(grid)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(grid[key] for key in keys))
    ]


def metric_row(
    model_name: str,
    params: dict[str, Any],
    split_name: str,
    subset: str,
    rows: pd.DataFrame,
    predictions: np.ndarray,
    target: str,
    *,
    n_features: int,
    best_iteration: int,
    fit_seconds: float | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Create one B7 tuning metrics row."""

    metrics = regression_metrics(
        rows[target].to_numpy(),
        predictions,
        cell_keys=rows["depmap_id"].tolist(),
        drug_keys=rows["drug_id"].tolist(),
    )
    row = {
        "model": model_name,
        "params": json.dumps(params, sort_keys=True),
        "split_name": split_name,
        "subset": subset,
        "n_rows": int(len(rows)),
        "n_cell_lines": int(rows["depmap_id"].nunique()),
        "n_drugs": int(rows["drug_id"].nunique()),
        "n_features": int(n_features),
        "best_iteration": int(best_iteration),
        **metrics,
    }
    if fit_seconds is not None:
        row["fit_seconds"] = float(fit_seconds)
    if device is not None:
        row["device"] = device
    return row


def build_variant_features(
    subsets: dict[str, pd.DataFrame],
    *,
    model_name: str,
    embedding_table: DrugEmbeddingTable,
    pca_feature_map: dict[str, np.ndarray],
    pathway_feature_map: dict[str, np.ndarray],
    cohort_key_column: str,
    missing_embeddings: str,
) -> dict[str, np.ndarray]:
    """Build train/validation/test matrices once for one B7 variant."""

    return {
        subset_name: assemble_b7_features(
            rows,
            model_name=model_name,
            drug_embedding_map=embedding_table.embedding_map,
            pca_feature_map=pca_feature_map,
            pathway_feature_map=pathway_feature_map,
            cohort_key_column=cohort_key_column,
            missing_embeddings=missing_embeddings,
        )
        for subset_name, rows in subsets.items()
    }


def tune_b7_variant_for_split(
    subsets: dict[str, pd.DataFrame],
    features: dict[str, np.ndarray],
    *,
    model_name: str,
    split_name: str,
    target: str,
    n_estimators: int,
    early_stopping_rounds: int,
    device: str,
) -> list[dict[str, Any]]:
    """Tune one B7 feature variant for one split."""

    x_train = features["train"]
    x_val = features["validation"]
    x_test = features["test"]
    y_train = subsets["train"][target].to_numpy()
    y_val = subsets["validation"][target].to_numpy()

    rows: list[dict[str, Any]] = []
    fitted: list[tuple[dict[str, Any], xgb.XGBRegressor, float, int, str, float]] = []
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
        best_iteration = getattr(model, "best_iteration", n_estimators - 1)
        predictions = predict_xgboost(model, x_val)
        row = metric_row(
            model_name,
            params,
            split_name,
            "validation",
            subsets["validation"],
            predictions,
            target,
            n_features=x_val.shape[1],
            best_iteration=best_iteration,
            fit_seconds=elapsed,
            device=actual_device,
        )
        rows.append(row)
        fitted.append(
            (params, model, row["rmse"], int(best_iteration), actual_device, elapsed)
        )
        logger.info(
            "B7 %s / %s params=%s validation RMSE %.3f fitted in %.1fs",
            split_name,
            model_name,
            params,
            row["rmse"],
            elapsed,
        )

    best_params, best_model, _rmse, best_iteration, best_device, _elapsed = min(
        fitted,
        key=lambda item: item[2],
    )
    test_predictions = predict_xgboost(best_model, x_test)
    test_row = metric_row(
        model_name,
        best_params,
        split_name,
        "test",
        subsets["test"],
        test_predictions,
        target,
        n_features=x_test.shape[1],
        best_iteration=best_iteration,
        device=best_device,
    )
    test_row["selected_by_validation"] = True
    rows.append(test_row)
    return rows


def tune_b7_for_split(
    cohort: pd.DataFrame,
    assignments: pd.DataFrame,
    embedding_table: DrugEmbeddingTable,
    pca_feature_map: dict[str, np.ndarray],
    pathway_feature_map: dict[str, np.ndarray],
    *,
    split_name: str,
    target: str,
    model_names: tuple[str, ...],
    n_estimators: int,
    early_stopping_rounds: int,
    device: str,
    cohort_key_column: str,
    missing_embeddings: str,
) -> list[dict[str, Any]]:
    """Tune all requested B7 feature variants for one split."""

    data = cohort.merge(assignments, on="pair_id", how="inner", validate="one_to_one")
    subsets = {
        subset: data.loc[data["split"].eq(subset)].copy()
        for subset in ("train", "validation", "test")
    }

    all_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        features = build_variant_features(
            subsets,
            model_name=model_name,
            embedding_table=embedding_table,
            pca_feature_map=pca_feature_map,
            pathway_feature_map=pathway_feature_map,
            cohort_key_column=cohort_key_column,
            missing_embeddings=missing_embeddings,
        )
        logger.info(
            "Tuning B7 %s / %s — train=%s, validation=%s",
            split_name,
            model_name,
            features["train"].shape,
            features["validation"].shape,
        )
        all_rows.extend(
            tune_b7_variant_for_split(
                subsets,
                features,
                model_name=model_name,
                split_name=split_name,
                target=target,
                n_estimators=n_estimators,
                early_stopping_rounds=early_stopping_rounds,
                device=device,
            )
        )
    return all_rows


def tune_b7(
    cohort_path: str | Path = "data/processed/cohort_pairs.csv",
    split_dir: str | Path = "data/processed/splits",
    expression_path: str | Path = EXPRESSION_PATH,
    drug_embeddings: str | Path = "data/external/drug_embeddings.csv",
    output: str | Path = "results/models/b7_tuning_metrics.csv",
    summary: str | Path = "data/reports/b7_tuning_summary.json",
    *,
    target: str = "ln_ic50",
    n_components: int = 256,
    gene_sets: str = "builtin_cancer_core",
    min_pathway_genes: int = 3,
    split_names: tuple[str, ...] = DEFAULT_SPLITS,
    model_names: tuple[str, ...] = DEFAULT_B7_MODELS,
    embedding_key_column: str = "drug_id",
    cohort_key_column: str = "drug_id",
    embedding_prefix: str | None = None,
    missing_embeddings: str = "error",
    n_estimators: int = 900,
    early_stopping_rounds: int = 60,
    device: str = "auto",
) -> pd.DataFrame:
    """Tune B7 hyperparameters by validation RMSE."""

    cohort = pd.read_csv(cohort_path)
    split_dir = Path(split_dir)
    embedding_table = load_drug_embeddings(
        drug_embeddings,
        key_column=embedding_key_column,
        embedding_prefix=embedding_prefix,
    )
    logger.info(
        "Loaded %d drug embeddings with dimension %d from %s.",
        embedding_table.n_drugs,
        embedding_table.embedding_dim,
        embedding_table.source_path,
    )

    all_rows: list[dict[str, Any]] = []
    pathway_names: list[str] = []
    pca_variance_by_split: dict[str, float] = {}
    for split_name in split_names:
        logger.info("=== Tuning B7 split: %s ===", split_name)
        assignments = pd.read_csv(split_dir / f"{split_name}.csv")
        merged = cohort.merge(assignments, on="pair_id", how="inner")
        train_ids = set(merged.loc[merged["split"].eq("train"), "depmap_id"].unique())

        pca_pipeline, pca_feature_map = build_cell_features(
            str(expression_path),
            cohort,
            train_ids,
            n_components=n_components,
        )
        pathway_pipeline, pathway_feature_map = build_pathway_features(
            str(expression_path),
            cohort,
            train_ids,
            gene_sets=gene_sets,
            min_genes=min_pathway_genes,
        )
        pathway_names = pathway_pipeline.pathway_names
        pca_variance_by_split[split_name] = float(
            pca_pipeline.explained_variance_ratio_sum
        )
        all_rows.extend(
            tune_b7_for_split(
                cohort,
                assignments,
                embedding_table,
                pca_feature_map,
                pathway_feature_map,
                split_name=split_name,
                target=target,
                model_names=model_names,
                n_estimators=n_estimators,
                early_stopping_rounds=early_stopping_rounds,
                device=device,
                cohort_key_column=cohort_key_column,
                missing_embeddings=missing_embeddings,
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
        "split_dir": str(split_dir),
        "drug_embeddings": embedding_table.source_path,
        "output": str(output),
        "target": target,
        "embedding_key_column": embedding_key_column,
        "cohort_key_column": cohort_key_column,
        "embedding_dim": embedding_table.embedding_dim,
        "n_embedding_drugs": embedding_table.n_drugs,
        "missing_embeddings": missing_embeddings,
        "n_pca_components": n_components,
        "pca_variance_by_split": pca_variance_by_split,
        "gene_sets": str(gene_sets),
        "min_pathway_genes": min_pathway_genes,
        "pathways": pathway_names,
        "model_variants": list(model_names),
        "grid": {key: list(value) for key, value in DEFAULT_GRID.items()},
        "n_estimators": n_estimators,
        "early_stopping_rounds": early_stopping_rounds,
        "device_requested": device,
        "splits": list(split_names),
        "best_validation": summarize_best_validation(metrics),
    }
    summary_path = Path(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_data, handle, indent=2)
    return metrics


def summarize_best_validation(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize best validation setting per split and B7 variant."""

    rows: list[dict[str, Any]] = []
    validation = metrics.loc[metrics["subset"].eq("validation")]
    for (split_name, model_name), group in validation.groupby(["split_name", "model"]):
        best = group.sort_values("rmse").iloc[0]
        rows.append(
            {
                "split_name": split_name,
                "model": model_name,
                "params": best["params"],
                "validation_rmse": float(best["rmse"]),
                "validation_mae": float(best["mae"]),
                "best_iteration": int(best["best_iteration"]),
                "device": best.get("device"),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description="Tune B7 pretrained models.")
    parser.add_argument("--cohort", default="data/processed/cohort_pairs.csv")
    parser.add_argument("--split-dir", default="data/processed/splits")
    parser.add_argument("--expression", default=EXPRESSION_PATH)
    parser.add_argument("--drug-embeddings", default="data/external/drug_embeddings.csv")
    parser.add_argument("--output", default="results/models/b7_tuning_metrics.csv")
    parser.add_argument("--summary", default="data/reports/b7_tuning_summary.json")
    parser.add_argument("--target", default="ln_ic50")
    parser.add_argument("--n-components", type=int, default=256)
    parser.add_argument("--gene-sets", default="builtin_cancer_core")
    parser.add_argument("--min-pathway-genes", type=int, default=3)
    parser.add_argument("--embedding-key-column", default="drug_id")
    parser.add_argument("--cohort-key-column", default="drug_id")
    parser.add_argument("--embedding-prefix", default=None)
    parser.add_argument(
        "--missing-embeddings",
        choices=["error", "zero"],
        default="error",
    )
    parser.add_argument("--n-estimators", type=int, default=900)
    parser.add_argument("--early-stopping-rounds", type=int, default=60)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=list(DEFAULT_SPLITS),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_B7_MODELS),
        choices=list(DEFAULT_B7_MODELS),
    )
    return parser


def main() -> None:
    """Command-line entry point."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = build_parser().parse_args()
    metrics = tune_b7(
        cohort_path=args.cohort,
        split_dir=args.split_dir,
        expression_path=args.expression,
        drug_embeddings=args.drug_embeddings,
        output=args.output,
        summary=args.summary,
        target=args.target,
        n_components=args.n_components,
        gene_sets=args.gene_sets,
        min_pathway_genes=args.min_pathway_genes,
        split_names=tuple(args.splits),
        model_names=tuple(args.models),
        embedding_key_column=args.embedding_key_column,
        cohort_key_column=args.cohort_key_column,
        embedding_prefix=args.embedding_prefix,
        missing_embeddings=args.missing_embeddings,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        device=args.device,
    )
    print(f"Wrote B7 tuning metrics to {args.output}")
    print(metrics.sort_values(["split_name", "subset", "model", "rmse"]).to_string())


if __name__ == "__main__":
    main()
