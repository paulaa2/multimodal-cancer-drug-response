"""B6 modality ablation study.

This script compares feature blocks under the same model family so the project
can answer a stronger question than "which model wins?": which information
source is responsible for each performance gain?

Default ablations:

- expression_pca
- pathways
- morgan
- morgan_expression_pca
- morgan_pathways
- morgan_expression_pca_pathways
"""

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
import xgboost as xgb

from mcdrp.features.expression import build_cell_features
from mcdrp.features.fingerprints import build_fingerprint_matrix
from mcdrp.features.pathways import build_pathway_features
from mcdrp.metrics import regression_metrics
from mcdrp.models.baseline_b1 import assemble_features
from mcdrp.models.baseline_b2 import fit_xgboost, predict_xgboost, xgb_device_params
from mcdrp.splits.make_splits import DEFAULT_SPLITS

logger = logging.getLogger(__name__)

EXPRESSION_PATH = "data/raw/OmicsExpressionTPMLogp1HumanProteinCodingGenesStranded.csv"
DEFAULT_ABLATIONS = (
    "expression_pca",
    "pathways",
    "morgan",
    "morgan_expression_pca",
    "morgan_pathways",
    "morgan_expression_pca_pathways",
)


def feature_matrix_from_cell_map(
    rows: pd.DataFrame,
    cell_feature_map: dict[str, np.ndarray],
) -> np.ndarray:
    """Build a dense cell feature matrix aligned to response rows."""

    n_features = next(iter(cell_feature_map.values())).shape[0]
    features = np.zeros((len(rows), n_features), dtype=np.float32)
    for idx, depmap_id in enumerate(rows["depmap_id"]):
        value = cell_feature_map.get(str(depmap_id))
        if value is not None:
            features[idx] = value
    return features


def assemble_ablation_features(
    rows: pd.DataFrame,
    *,
    ablation: str,
    cohort: pd.DataFrame,
    fp_matrix: np.ndarray,
    pca_feature_map: dict[str, np.ndarray],
    pathway_feature_map: dict[str, np.ndarray],
) -> np.ndarray:
    """Assemble the requested feature blocks."""

    pair_id_to_idx = {pid: idx for idx, pid in enumerate(cohort["pair_id"])}
    row_indices = np.array([pair_id_to_idx[pid] for pid in rows["pair_id"]])

    blocks: list[np.ndarray] = []
    if "morgan" in ablation:
        blocks.append(fp_matrix[row_indices].astype(np.float32))
    if "expression_pca" in ablation:
        blocks.append(feature_matrix_from_cell_map(rows, pca_feature_map))
    if "pathways" in ablation:
        blocks.append(feature_matrix_from_cell_map(rows, pathway_feature_map))

    if not blocks:
        raise ValueError(f"Unknown or empty ablation: {ablation}")
    return np.hstack(blocks).astype(np.float32)


def metric_row(
    model_name: str,
    split_name: str,
    subset: str,
    rows: pd.DataFrame,
    predictions: np.ndarray,
    target: str,
    *,
    n_features: int,
    fit_seconds: float,
    best_iteration: int,
    device: str,
) -> dict[str, Any]:
    """Create one ablation metrics row."""

    metrics = regression_metrics(rows[target].to_numpy(), predictions)
    return {
        "model": model_name,
        "split_name": split_name,
        "subset": subset,
        "n_rows": int(len(rows)),
        "n_cell_lines": int(rows["depmap_id"].nunique()),
        "n_drugs": int(rows["drug_id"].nunique()),
        "n_features": n_features,
        "fit_seconds": fit_seconds,
        "best_iteration": int(best_iteration),
        "device": device,
        **metrics,
    }


def run_ablation_for_split(
    cohort: pd.DataFrame,
    assignments: pd.DataFrame,
    fp_matrix: np.ndarray,
    pca_feature_map: dict[str, np.ndarray],
    pathway_feature_map: dict[str, np.ndarray],
    *,
    split_name: str,
    target: str,
    ablations: tuple[str, ...],
    xgb_params: dict[str, Any],
    device: str,
) -> list[dict[str, Any]]:
    """Run every ablation for one split."""

    data = cohort.merge(assignments, on="pair_id", how="inner", validate="one_to_one")
    subsets = {
        subset: data.loc[data["split"].eq(subset)].copy()
        for subset in ("train", "validation", "test")
    }

    results: list[dict[str, Any]] = []
    for ablation in ablations:
        x_train = assemble_ablation_features(
            subsets["train"],
            ablation=ablation,
            cohort=cohort,
            fp_matrix=fp_matrix,
            pca_feature_map=pca_feature_map,
            pathway_feature_map=pathway_feature_map,
        )
        x_val = assemble_ablation_features(
            subsets["validation"],
            ablation=ablation,
            cohort=cohort,
            fp_matrix=fp_matrix,
            pca_feature_map=pca_feature_map,
            pathway_feature_map=pathway_feature_map,
        )
        y_train = subsets["train"][target].to_numpy()
        y_val = subsets["validation"][target].to_numpy()

        logger.info(
            "B6 %s / %s — train=%s, validation=%s",
            split_name,
            ablation,
            x_train.shape,
            x_val.shape,
        )
        t0 = time.time()
        model, actual_device = fit_xgboost(
            xgb_params,
            x_train,
            y_train,
            x_val,
            y_val,
            device=device,
        )
        fit_seconds = time.time() - t0
        best_iteration = getattr(model, "best_iteration", xgb_params["n_estimators"] - 1)

        for subset in ("validation", "test"):
            rows = subsets[subset]
            x_subset = assemble_ablation_features(
                rows,
                ablation=ablation,
                cohort=cohort,
                fp_matrix=fp_matrix,
                pca_feature_map=pca_feature_map,
                pathway_feature_map=pathway_feature_map,
            )
            predictions = predict_xgboost(model, x_subset)
            results.append(
                metric_row(
                    ablation,
                    split_name,
                    subset,
                    rows,
                    predictions,
                    target,
                    n_features=x_subset.shape[1],
                    fit_seconds=fit_seconds,
                    best_iteration=best_iteration,
                    device=actual_device,
                )
            )
    return results


def run_b6_ablation(
    cohort_path: str | Path = "data/processed/cohort_pairs.csv",
    split_dir: str | Path = "data/processed/splits",
    expression_path: str | Path = EXPRESSION_PATH,
    output: str | Path = "results/ablations/b6_modality_ablation_metrics.csv",
    summary: str | Path = "data/reports/b6_modality_ablation_summary.json",
    *,
    target: str = "ln_ic50",
    n_components: int = 256,
    gene_sets: str = "builtin_cancer_core",
    min_pathway_genes: int = 3,
    split_names: tuple[str, ...] = DEFAULT_SPLITS,
    ablations: tuple[str, ...] = DEFAULT_ABLATIONS,
    n_estimators: int = 600,
    learning_rate: float = 0.03,
    max_depth: int = 5,
    early_stopping_rounds: int = 40,
    device: str = "auto",
) -> pd.DataFrame:
    """Run the B6 feature ablation study."""

    cohort = pd.read_csv(cohort_path)
    split_dir = Path(split_dir)
    logger.info("Building Morgan fingerprint matrix for %d rows ...", len(cohort))
    fp_matrix = build_fingerprint_matrix(cohort["canonical_smiles"])

    xgb_params: dict[str, Any] = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "early_stopping_rounds": early_stopping_rounds,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }
    xgb_params.update(xgb_device_params(device))

    all_results: list[dict[str, Any]] = []
    pathway_names: list[str] = []
    for split_name in split_names:
        logger.info("=== Processing B6 ablations split: %s ===", split_name)
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
        logger.info(
            "B6 %s — PCA variance %.1f%%, pathways=%d",
            split_name,
            pca_pipeline.explained_variance_ratio_sum * 100,
            len(pathway_names),
        )

        all_results.extend(
            run_ablation_for_split(
                cohort,
                assignments,
                fp_matrix,
                pca_feature_map,
                pathway_feature_map,
                split_name=split_name,
                target=target,
                ablations=ablations,
                xgb_params=xgb_params,
                device=device,
            )
        )

    metrics = pd.DataFrame(all_results)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)

    summary_data = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_path": str(cohort_path),
        "expression_path": str(expression_path),
        "split_dir": str(split_dir),
        "output": str(output),
        "target": target,
        "n_pca_components": n_components,
        "gene_sets": str(gene_sets),
        "min_pathway_genes": min_pathway_genes,
        "pathways": pathway_names,
        "ablations": list(ablations),
        "xgb_params": {k: v for k, v in xgb_params.items() if k != "n_jobs"},
        "device_requested": device,
        "splits": list(split_names),
        "best_by_split_subset": summarize_best(metrics),
    }
    summary_path = Path(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_data, handle, indent=2)
    return metrics


def summarize_best(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize the best ablation for each split/subset."""

    rows: list[dict[str, Any]] = []
    for (split_name, subset), group in metrics.groupby(["split_name", "subset"]):
        best = group.sort_values("rmse").iloc[0]
        rows.append(
            {
                "split_name": split_name,
                "subset": subset,
                "best_model": best["model"],
                "rmse": float(best["rmse"]),
                "mae": float(best["mae"]),
                "pearson": float(best["pearson"])
                if pd.notna(best["pearson"])
                else None,
                "spearman": float(best["spearman"])
                if pd.notna(best["spearman"])
                else None,
                "r2": float(best["r2"]) if pd.notna(best["r2"]) else None,
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description="Run B6 modality ablations.")
    parser.add_argument("--cohort", default="data/processed/cohort_pairs.csv")
    parser.add_argument("--split-dir", default="data/processed/splits")
    parser.add_argument("--expression", default=EXPRESSION_PATH)
    parser.add_argument(
        "--output",
        default="results/ablations/b6_modality_ablation_metrics.csv",
    )
    parser.add_argument(
        "--summary",
        default="data/reports/b6_modality_ablation_summary.json",
    )
    parser.add_argument("--target", default="ln_ic50")
    parser.add_argument("--n-components", type=int, default=256)
    parser.add_argument("--gene-sets", default="builtin_cancer_core")
    parser.add_argument("--min-pathway-genes", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=list(DEFAULT_SPLITS),
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=list(DEFAULT_ABLATIONS),
        choices=list(DEFAULT_ABLATIONS),
    )
    return parser


def main() -> None:
    """Command-line entry point."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = build_parser().parse_args()
    metrics = run_b6_ablation(
        cohort_path=args.cohort,
        split_dir=args.split_dir,
        expression_path=args.expression,
        output=args.output,
        summary=args.summary,
        target=args.target,
        n_components=args.n_components,
        gene_sets=args.gene_sets,
        min_pathway_genes=args.min_pathway_genes,
        split_names=tuple(args.splits),
        ablations=tuple(args.ablations),
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        early_stopping_rounds=args.early_stopping_rounds,
        device=args.device,
    )
    print(f"\nWrote B6 ablation metrics to {args.output}")
    print("=" * 72)
    for _, row in metrics.sort_values(["split_name", "subset", "rmse"]).iterrows():
        print(
            f"  {row['split_name']:>14s} / {row['subset']:>10s} / "
            f"{row['model']:>30s}: RMSE={row['rmse']:.3f}  "
            f"MAE={row['mae']:.3f}  Pearson={row['pearson']:.3f}  "
            f"R²={row['r2']:.3f}"
        )


if __name__ == "__main__":
    main()
