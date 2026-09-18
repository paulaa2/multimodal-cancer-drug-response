"""B2 XGBoost baseline with Morgan FP + PCA expression features.

Uses the same feature engineering as B1 (fingerprints + PCA) but fits an
XGBoost gradient-boosted tree regressor with early stopping on the
validation set.
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
from mcdrp.metrics import regression_metrics
from mcdrp.models.baseline_b1 import assemble_features
from mcdrp.results.predictions import prediction_frame, write_predictions
from mcdrp.splits.make_splits import DEFAULT_SPLITS

logger = logging.getLogger(__name__)

EXPRESSION_PATH = "data/raw/OmicsExpressionTPMLogp1HumanProteinCodingGenesStranded.csv"


def run_b2_for_split(
    cohort: pd.DataFrame,
    split_assignments: pd.DataFrame,
    fp_matrix: np.ndarray,
    cell_feature_map: dict[str, np.ndarray],
    *,
    split_name: str,
    target: str,
    xgb_params: dict[str, Any],
    device: str,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    """Run XGBoost for one split configuration."""

    data = cohort.merge(
        split_assignments, on="pair_id", how="inner", validate="one_to_one"
    )
    pair_id_to_idx = {pid: idx for idx, pid in enumerate(cohort["pair_id"])}

    results: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []

    # Prepare subsets.
    subsets: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for label in ("train", "validation", "test"):
        mask = data["split"].eq(label)
        rows = data.loc[mask].copy()
        indices = np.array([pair_id_to_idx[pid] for pid in rows["pair_id"]])
        X = assemble_features(
            rows, fp_matrix, cell_feature_map, row_indices=indices
        )
        subsets[label] = (rows, X)

    train_rows, X_train = subsets["train"]
    val_rows, X_val = subsets["validation"]
    y_train = train_rows[target].to_numpy()
    y_val = val_rows[target].to_numpy()

    logger.info(
        "Split %s — train: %s, val: %s",
        split_name,
        X_train.shape,
        X_val.shape,
    )

    # Fit XGBoost with early stopping on validation RMSE.
    t0 = time.time()
    model, actual_device = fit_xgboost(
        xgb_params,
        X_train,
        y_train,
        X_val,
        y_val,
        device=device,
    )
    best_iteration = model.best_iteration
    elapsed = time.time() - t0
    logger.info(
        "XGBoost fitted in %.1fs — best iteration: %d / %d",
        elapsed,
        best_iteration,
        xgb_params.get("n_estimators", 500),
    )

    for subset_name in ("validation", "test"):
        subset_rows, X_subset = subsets[subset_name]
        predictions = predict_xgboost(model, X_subset)
        metrics = regression_metrics(
            subset_rows[target].to_numpy(),
            predictions,
            cell_keys=subset_rows["depmap_id"].tolist(),
            drug_keys=subset_rows["drug_id"].tolist(),
        )
        results.append(
            {
                "model": "xgboost",
                "split_name": split_name,
                "subset": subset_name,
                "n_rows": int(len(subset_rows)),
                "n_cell_lines": int(subset_rows["depmap_id"].nunique()),
                "n_drugs": int(subset_rows["drug_id"].nunique()),
                "best_iteration": int(best_iteration),
                "device": actual_device,
                **metrics,
            }
        )
        frames.append(
            prediction_frame(
                subset_rows,
                subset_rows[target].to_numpy(),
                predictions,
                stage="B2",
                model="xgboost",
                split_name=split_name,
                subset=subset_name,
            )
        )

    return results, frames


def run_b2(
    cohort_path: str | Path = "data/processed/cohort_pairs.csv",
    split_dir: str | Path = "data/processed/splits",
    expression_path: str | Path = EXPRESSION_PATH,
    output: str | Path = "results/baselines/b2_metrics.csv",
    summary: str | Path = "data/reports/b2_summary.json",
    predictions_output: str | Path = "results/predictions/b2.csv",
    *,
    target: str = "ln_ic50",
    n_components: int = 256,
    split_names: tuple[str, ...] = DEFAULT_SPLITS,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    early_stopping_rounds: int = 30,
    device: str = "auto",
) -> pd.DataFrame:
    """Run B2 XGBoost baseline across all splits."""

    cohort = pd.read_csv(cohort_path)
    split_dir = Path(split_dir)

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

    # Build fingerprints once.
    logger.info("Building Morgan fingerprint matrix for %d rows ...", len(cohort))
    fp_matrix = build_fingerprint_matrix(cohort["canonical_smiles"])
    logger.info("Fingerprint matrix shape: %s", fp_matrix.shape)

    all_results: list[dict[str, Any]] = []
    all_frames: list[pd.DataFrame] = []

    for split_name in split_names:
        logger.info("=== Processing split: %s ===", split_name)
        split_path = split_dir / f"{split_name}.csv"
        assignments = pd.read_csv(split_path)

        # Determine training cell lines for leakage-safe PCA.
        merged = cohort.merge(assignments, on="pair_id", how="inner")
        train_ids = set(merged.loc[merged["split"].eq("train"), "depmap_id"].unique())

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

        split_results, split_frames = run_b2_for_split(
            cohort,
            assignments,
            fp_matrix,
            cell_feature_map,
            split_name=split_name,
            target=target,
            xgb_params=xgb_params,
            device=device,
        )
        all_results.extend(split_results)
        all_frames.extend(split_frames)

    metrics = pd.DataFrame(all_results)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)

    write_predictions(all_frames, predictions_output)

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
        "xgb_params": {k: v for k, v in xgb_params.items() if k != "n_jobs"},
        "xgb_device_requested": device,
        "splits": list(split_names),
        "best_by_split_subset": _summarize_best(metrics),
    }
    summary_path = Path(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_data, handle, indent=2)

    return metrics


def _summarize_best(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    """Pick the best result for each split/subset (only one model here)."""

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
                "best_iteration": int(best["best_iteration"]),
                "device": best["device"],
            }
        )
    return rows


def load_selected_xgb_params(
    path: str | Path | None,
    *,
    fallback: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Load per-split XGBoost params chosen on validation by ``tune_b2``.

    Missing files or missing splits fall back to ``fallback``, so ablations can
    still run before B2 has been tuned. Selected test rows are preferred; if a
    table has no ``selected_by_validation`` column, the lowest validation RMSE
    per split is used instead.
    """

    if path is None:
        return {}

    metric_path = Path(path)
    if not metric_path.exists():
        logger.warning(
            "B2 tuning file %s is missing; using fallback XGBoost params.",
            metric_path,
        )
        return {}

    from mcdrp.results.compare_baselines import is_truthy

    table = pd.read_csv(metric_path)
    if "selected_by_validation" in table.columns:
        selected_mask = table["selected_by_validation"].map(is_truthy).fillna(False)
        selected = table.loc[table["subset"].eq("test") & selected_mask]
    else:
        selected = pd.DataFrame()

    if selected.empty:
        selected = (
            table.loc[table["subset"].eq("validation")]
            .sort_values("rmse")
            .groupby("split_name", as_index=False)
            .first()
        )

    by_split: dict[str, dict[str, Any]] = {}
    for row in selected.to_dict("records"):
        raw_params = row.get("params", "{}")
        parsed = json.loads(raw_params) if isinstance(raw_params, str) else {}
        by_split[str(row["split_name"])] = {**fallback, **parsed}
    return by_split


def xgb_device_params(device: str) -> dict[str, str]:
    """Return XGBoost device parameters for the requested device mode."""

    if device == "cpu":
        return {"device": "cpu"}
    if device in {"auto", "cuda"}:
        return {"device": "cuda"}
    raise ValueError(f"Unsupported XGBoost device: {device}")


def fit_xgboost(
    xgb_params: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    device: str,
) -> tuple[xgb.XGBRegressor, str]:
    """Fit XGBoost, optionally falling back from CUDA to CPU in auto mode."""

    try:
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        return model, xgb_params.get("device", "cpu")
    except xgb.core.XGBoostError:
        if device != "auto":
            raise
        logger.warning("XGBoost CUDA training failed in auto mode; retrying on CPU.")
        cpu_params = {**xgb_params, "device": "cpu"}
        model = xgb.XGBRegressor(**cpu_params)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        return model, "cpu"


def predict_xgboost(model: xgb.XGBRegressor, features: np.ndarray) -> np.ndarray:
    """Predict through the Booster API to avoid CPU/GPU inplace warnings."""

    booster = model.get_booster()
    dmatrix = xgb.DMatrix(features)
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        return booster.predict(dmatrix)
    return booster.predict(dmatrix, iteration_range=(0, int(best_iteration) + 1))


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description="Run B2 XGBoost baseline.")
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
        default="results/baselines/b2_metrics.csv",
        help="Output metrics CSV.",
    )
    parser.add_argument(
        "--summary",
        default="data/reports/b2_summary.json",
        help="Output JSON summary.",
    )
    parser.add_argument(
        "--predictions-output",
        default="results/predictions/b2.csv",
        help="Output per-row predictions CSV consumed by mcdrp.results.metrics_report.",
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
        "--n-estimators",
        type=int,
        default=500,
        help="Max boosting rounds.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="XGBoost learning rate.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Max tree depth.",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=30,
        dest="early_stopping_rounds",
        help="Early stopping rounds.",
    )
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
    metrics = run_b2(
        cohort_path=args.cohort,
        split_dir=args.split_dir,
        expression_path=args.expression,
        output=args.output,
        summary=args.summary,
        predictions_output=args.predictions_output,
        target=args.target,
        n_components=args.n_components,
        split_names=tuple(args.splits),
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        early_stopping_rounds=args.early_stopping_rounds,
        device=args.device,
    )
    print(f"\nWrote B2 metrics to {args.output}")
    print("=" * 72)
    for _, row in metrics.sort_values(["split_name", "subset", "rmse"]).iterrows():
        print(
            f"  {row['split_name']:>12s} / {row['subset']:>10s} / {row['model']:>12s}: "
            f"RMSE={row['rmse']:.3f}  MAE={row['mae']:.3f}  "
            f"Pearson={row['pearson']:.3f}  R²={row['r2']:.3f}  "
            f"(iter={row['best_iteration']:.0f})"
        )


if __name__ == "__main__":
    main()
