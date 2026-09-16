"""B1 classical baselines: Ridge and Elastic Net with Morgan FP + PCA features.

This baseline concatenates Morgan fingerprints (drug representation) with
PCA-reduced gene expression (cell-line representation) and fits Ridge and
Elastic Net regressors.  All preprocessing is fit on training data only.
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
from sklearn.linear_model import ElasticNet, RidgeCV

from mcdrp.features.expression import build_cell_features
from mcdrp.features.fingerprints import build_fingerprint_matrix
from mcdrp.metrics import regression_metrics
from mcdrp.splits.make_splits import DEFAULT_SPLITS

logger = logging.getLogger(__name__)

EXPRESSION_PATH = "data/raw/OmicsExpressionTPMLogp1HumanProteinCodingGenesStranded.csv"


def assemble_features(
    rows: pd.DataFrame,
    fp_matrix: np.ndarray,
    cell_feature_map: dict[str, np.ndarray],
    *,
    row_indices: np.ndarray,
) -> np.ndarray:
    """Concatenate drug fingerprints and cell PCA features for given rows.

    Parameters
    ----------
    rows:
        Subset of the cohort DataFrame (train, validation, or test).
    fp_matrix:
        Full fingerprint matrix aligned with the complete cohort.
    cell_feature_map:
        Mapping from ``depmap_id`` to PCA feature vector.
    row_indices:
        Integer indices into ``fp_matrix`` for the rows in ``rows``.

    Returns
    -------
    Feature matrix of shape ``(len(rows), n_fp_bits + n_pca_components)``.
    """
    fps = fp_matrix[row_indices]

    n_pca = next(iter(cell_feature_map.values())).shape[0]
    cell_feats = np.zeros((len(rows), n_pca), dtype=np.float32)
    for local_idx, depmap_id in enumerate(rows["depmap_id"]):
        if depmap_id in cell_feature_map:
            cell_feats[local_idx] = cell_feature_map[depmap_id]

    return np.hstack([fps.astype(np.float32), cell_feats])


def evaluate_model(
    model: Any,
    model_name: str,
    X: np.ndarray,
    rows: pd.DataFrame,
    *,
    target: str,
    split_name: str,
    subset: str,
) -> dict[str, Any]:
    """Predict and compute regression metrics for one model on one subset."""
    predictions = model.predict(X)
    metrics = regression_metrics(rows[target].to_numpy(), predictions)
    return {
        "model": model_name,
        "split_name": split_name,
        "subset": subset,
        "n_rows": int(len(rows)),
        "n_cell_lines": int(rows["depmap_id"].nunique()),
        "n_drugs": int(rows["drug_id"].nunique()),
        **metrics,
    }


def run_b1_for_split(
    cohort: pd.DataFrame,
    split_assignments: pd.DataFrame,
    fp_matrix: np.ndarray,
    cell_feature_map: dict[str, np.ndarray],
    *,
    split_name: str,
    target: str,
    elastic_alpha: float = 0.02,
    elastic_l1_ratio: float = 0.9,
) -> list[dict[str, Any]]:
    """Run Ridge + Elastic Net for one split configuration."""

    data = cohort.merge(
        split_assignments, on="pair_id", how="inner", validate="one_to_one"
    )

    # Build index mapping from cohort pair_id to position in fp_matrix.
    pair_id_to_idx = {pid: idx for idx, pid in enumerate(cohort["pair_id"])}

    results: list[dict[str, Any]] = []

    # Prepare train features.
    train_mask = data["split"].eq("train")
    train_rows = data.loc[train_mask].copy()
    train_indices = np.array([pair_id_to_idx[pid] for pid in train_rows["pair_id"]])
    X_train = assemble_features(
        train_rows, fp_matrix, cell_feature_map, row_indices=train_indices
    )
    y_train = train_rows[target].to_numpy()

    logger.info(
        "Split %s — train shape: %s, target mean: %.3f, std: %.3f",
        split_name,
        X_train.shape,
        y_train.mean(),
        y_train.std(),
    )

    # Fit Ridge.
    t0 = time.time()
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge.fit(X_train, y_train)
    logger.info(
        "RidgeCV fitted in %.1fs — best alpha: %.2f",
        time.time() - t0,
        ridge.alpha_,
    )

    # Fit Elastic Net. We intentionally avoid ElasticNetCV on the full matrix:
    # cross-validation creates large fold-specific copies and can exhaust RAM.
    t0 = time.time()
    elastic = ElasticNet(
        alpha=elastic_alpha,
        l1_ratio=elastic_l1_ratio,
        max_iter=2000,
        random_state=42,
        selection="random",
        copy_X=False,
    )
    elastic.fit(X_train, y_train)
    logger.info(
        "ElasticNet fitted in %.1fs — alpha: %.4f, l1_ratio: %.2f",
        time.time() - t0,
        elastic_alpha,
        elastic_l1_ratio,
    )

    models = [("ridge", ridge), ("elastic_net", elastic)]

    for subset in ("validation", "test"):
        subset_mask = data["split"].eq(subset)
        subset_rows = data.loc[subset_mask].copy()
        subset_indices = np.array(
            [pair_id_to_idx[pid] for pid in subset_rows["pair_id"]]
        )
        X_subset = assemble_features(
            subset_rows, fp_matrix, cell_feature_map, row_indices=subset_indices
        )

        for model_name, model in models:
            results.append(
                evaluate_model(
                    model,
                    model_name,
                    X_subset,
                    subset_rows,
                    target=target,
                    split_name=split_name,
                    subset=subset,
                )
            )

    return results


def run_b1(
    cohort_path: str | Path = "data/processed/cohort_pairs.csv",
    split_dir: str | Path = "data/processed/splits",
    expression_path: str | Path = EXPRESSION_PATH,
    output: str | Path = "results/baselines/b1_metrics.csv",
    summary: str | Path = "data/reports/b1_summary.json",
    *,
    target: str = "ln_ic50",
    n_components: int = 256,
    split_names: tuple[str, ...] = DEFAULT_SPLITS,
    elastic_alpha: float = 0.02,
    elastic_l1_ratio: float = 0.9,
) -> pd.DataFrame:
    """Run B1 baselines across all splits."""

    cohort = pd.read_csv(cohort_path)
    split_dir = Path(split_dir)

    # Build fingerprints once (shared across splits).
    logger.info("Building Morgan fingerprint matrix for %d rows ...", len(cohort))
    fp_matrix = build_fingerprint_matrix(cohort["canonical_smiles"])
    logger.info("Fingerprint matrix shape: %s", fp_matrix.shape)

    all_results: list[dict[str, Any]] = []

    for split_name in split_names:
        logger.info("=== Processing split: %s ===", split_name)
        split_path = split_dir / f"{split_name}.csv"
        assignments = pd.read_csv(split_path)

        # Determine training cell lines for leakage-safe PCA.
        merged = cohort.merge(assignments, on="pair_id", how="inner")
        train_ids = set(merged.loc[merged["split"].eq("train"), "depmap_id"].unique())

        # Build cell features with PCA fit on train only.
        _pipeline, cell_feature_map = build_cell_features(
            str(expression_path),
            cohort,
            train_ids,
            n_components=n_components,
        )
        logger.info(
            "PCA explained variance: %.1f%%",
            _pipeline.explained_variance_ratio_sum * 100,
        )

        split_results = run_b1_for_split(
            cohort,
            assignments,
            fp_matrix,
            cell_feature_map,
            split_name=split_name,
            target=target,
            elastic_alpha=elastic_alpha,
            elastic_l1_ratio=elastic_l1_ratio,
        )
        all_results.extend(split_results)

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
        "fp_bits": 2048,
        "fp_radius": 2,
        "elastic_alpha": elastic_alpha,
        "elastic_l1_ratio": elastic_l1_ratio,
        "splits": list(split_names),
        "best_by_split_subset": _summarize_best(metrics),
    }
    summary_path = Path(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_data, handle, indent=2)

    return metrics


def _summarize_best(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    """Pick the best B1 variant by RMSE for each split/subset."""

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

    parser = argparse.ArgumentParser(
        description="Run B1 Ridge/Elastic Net baselines."
    )
    parser.add_argument(
        "--cohort",
        default="data/processed/cohort_pairs.csv",
        help="Input cohort CSV.",
    )
    parser.add_argument(
        "--split-dir",
        default="data/processed/splits",
        help="Directory containing split assignment CSVs.",
    )
    parser.add_argument(
        "--expression",
        default=EXPRESSION_PATH,
        help="Path to DepMap expression matrix.",
    )
    parser.add_argument(
        "--output",
        default="results/baselines/b1_metrics.csv",
        help="Output metrics CSV.",
    )
    parser.add_argument(
        "--summary",
        default="data/reports/b1_summary.json",
        help="Output JSON summary.",
    )
    parser.add_argument(
        "--target",
        default="ln_ic50",
        help="Regression target column.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=256,
        help="Number of PCA components for expression features.",
    )
    parser.add_argument(
        "--elastic-alpha",
        type=float,
        default=0.02,
        help="Fixed ElasticNet alpha. Avoids memory-heavy ElasticNetCV.",
    )
    parser.add_argument(
        "--elastic-l1-ratio",
        type=float,
        default=0.9,
        help="Fixed ElasticNet l1_ratio.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=list(DEFAULT_SPLITS),
        help="Splits to evaluate.",
    )
    return parser


def main() -> None:
    """Command-line entry point."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    args = build_parser().parse_args()
    metrics = run_b1(
        cohort_path=args.cohort,
        split_dir=args.split_dir,
        expression_path=args.expression,
        output=args.output,
        summary=args.summary,
        target=args.target,
        n_components=args.n_components,
        split_names=tuple(args.splits),
        elastic_alpha=args.elastic_alpha,
        elastic_l1_ratio=args.elastic_l1_ratio,
    )
    print(f"\nWrote B1 metrics to {args.output}")
    print("=" * 72)
    for _, row in metrics.sort_values(["split_name", "subset", "rmse"]).iterrows():
        print(
            f"  {row['split_name']:>12s} / {row['subset']:>10s} / {row['model']:>12s}: "
            f"RMSE={row['rmse']:.3f}  MAE={row['mae']:.3f}  "
            f"Pearson={row['pearson']:.3f}  R²={row['r2']:.3f}"
        )


if __name__ == "__main__":
    main()
