"""B0 mean baselines for drug-response prediction.

These baselines intentionally use no transcriptomics and no chemical structure.
They give the reference that later models must beat.

``mean_effects`` is the strongest of them and the one that matters: it predicts
``mu_cell + mu_drug - mu`` and is the standard reference in drug-response
benchmarks, where most published models barely improve on it. ``global_mean``,
``drug_mean``, ``cell_mean``, and ``tissue_mean`` isolate how much of the
signal comes from each single marginal effect.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mcdrp.metrics import MeanEffectsReference, regression_metrics
from mcdrp.results.predictions import prediction_frame, write_predictions
from mcdrp.splits.make_splits import DEFAULT_SPLITS, TISSUE_COLUMN


def predict_global_mean(train: pd.DataFrame, target: str, rows: pd.DataFrame) -> pd.Series:
    return pd.Series(train[target].mean(), index=rows.index)


def predict_mean_effects(
    train: pd.DataFrame,
    target: str,
    rows: pd.DataFrame,
) -> pd.Series:
    """Predict the additive drug plus cell-line mean effect."""

    reference = MeanEffectsReference.fit(
        train["depmap_id"].tolist(),
        train["drug_id"].tolist(),
        train[target].to_numpy(),
    )
    predictions = reference.predict(
        rows["depmap_id"].tolist(),
        rows["drug_id"].tolist(),
    )
    return pd.Series(predictions, index=rows.index)


def predict_group_mean(
    train: pd.DataFrame,
    target: str,
    rows: pd.DataFrame,
    group_column: str,
) -> pd.Series:

    global_mean = train[target].mean()
    means = train.groupby(group_column)[target].mean()
    return rows[group_column].map(means).fillna(global_mean)


def evaluate_predictions(
    rows: pd.DataFrame,
    predictions: pd.Series,
    *,
    target: str,
    model_name: str,
    split_name: str,
    subset: str,
    reference_pred: np.ndarray | None = None,
) -> dict[str, Any]:
    metrics = regression_metrics(
        rows[target].to_numpy(),
        predictions.to_numpy(),
        reference_pred=reference_pred,
        cell_keys=rows["depmap_id"].tolist(),
        drug_keys=rows["drug_id"].tolist(),
    )
    return {
        "model": model_name,
        "split_name": split_name,
        "subset": subset,
        "n_rows": int(len(rows)),
        "n_cell_lines": int(rows["depmap_id"].nunique()),
        "n_drugs": int(rows["drug_id"].nunique()),
        **metrics,
    }


def b0_model_specs(
    train: pd.DataFrame,
    data: pd.DataFrame,
    target: str,
) -> list[tuple[str, Callable[[pd.DataFrame], pd.Series]]]:
    """List the B0 variants, adding tissue_mean when lineage is available."""

    specs: list[tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [
        ("global_mean", lambda rows: predict_global_mean(train, target, rows)),
        ("drug_mean", lambda rows: predict_group_mean(train, target, rows, "drug_id")),
        ("cell_mean", lambda rows: predict_group_mean(train, target, rows, "depmap_id")),
        ("mean_effects", lambda rows: predict_mean_effects(train, target, rows)),
    ]
    if TISSUE_COLUMN in data.columns:
        specs.append(
            (
                "tissue_mean",
                lambda rows: predict_group_mean(train, target, rows, TISSUE_COLUMN),
            )
        )
    return specs


def run_b0_for_split(
    cohort: pd.DataFrame,
    split_assignments: pd.DataFrame,
    *,
    split_name: str,
    target: str,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    """Evaluate every B0 variant on one split.

    Returns both the metric rows and tidy prediction frames. The predictions
    are the durable artifact: ``mcdrp.results.metrics_report`` recomputes all
    metric families from them without refitting anything.
    """

    data = cohort.merge(split_assignments, on="pair_id", how="inner", validate="one_to_one")
    train = data.loc[data["split"].eq("train")].copy()
    results: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []

    model_specs = b0_model_specs(train, data, target)

    for subset in ("validation", "test"):
        rows = data.loc[data["split"].eq(subset)].copy()
        # The same train-only reference normalizes every model on these rows.
        reference_pred = predict_mean_effects(train, target, rows).to_numpy()
        for model_name, predict in model_specs:
            predictions = predict(rows)
            results.append(
                evaluate_predictions(
                    rows,
                    predictions,
                    target=target,
                    model_name=model_name,
                    split_name=split_name,
                    subset=subset,
                    reference_pred=reference_pred,
                )
            )
            frames.append(
                prediction_frame(
                    rows,
                    rows[target].to_numpy(),
                    predictions.to_numpy(),
                    stage="B0",
                    model=model_name,
                    split_name=split_name,
                    subset=subset,
                )
            )

    return results, frames


def run_b0(
    cohort_path: str | Path = "data/processed/cohort_pairs.csv",
    split_dir: str | Path = "data/processed/splits",
    output: str | Path = "results/baselines/b0_metrics.csv",
    summary: str | Path = "data/reports/b0_summary.json",
    predictions_output: str | Path = "results/predictions/b0.csv",
    *,
    target: str = "ln_ic50",
    split_names: tuple[str, ...] = DEFAULT_SPLITS,
) -> pd.DataFrame:

    cohort = pd.read_csv(cohort_path)
    split_dir = Path(split_dir)

    all_results: list[dict[str, Any]] = []
    all_frames: list[pd.DataFrame] = []
    for split_name in split_names:
        split_path = split_dir / f"{split_name}.csv"
        assignments = pd.read_csv(split_path)
        results, frames = run_b0_for_split(
            cohort,
            assignments,
            split_name=split_name,
            target=target,
        )
        all_results.extend(results)
        all_frames.extend(frames)

    metrics = pd.DataFrame(all_results)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)

    write_predictions(all_frames, predictions_output)

    summary_data = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_path": str(cohort_path),
        "split_dir": str(split_dir),
        "output": str(output),
        "predictions_output": str(predictions_output),
        "target": target,
        "splits": list(split_names),
        "best_by_split_subset": summarize_best(metrics),
    }
    summary_path = Path(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_data, handle, indent=2)

    return metrics


def summarize_best(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize the best B0 variant by RMSE for each split/subset."""

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

    parser = argparse.ArgumentParser(description="Run B0 mean baselines.")
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
        "--output",
        default="results/baselines/b0_metrics.csv",
        help="Output metrics CSV.",
    )
    parser.add_argument(
        "--summary",
        default="data/reports/b0_summary.json",
        help="Output JSON summary.",
    )
    parser.add_argument(
        "--predictions-output",
        default="results/predictions/b0.csv",
        help="Output per-row predictions CSV consumed by mcdrp.results.metrics_report.",
    )
    parser.add_argument(
        "--target",
        default="ln_ic50",
        help="Regression target column.",
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
    args = build_parser().parse_args()
    metrics = run_b0(
        cohort_path=args.cohort,
        split_dir=args.split_dir,
        output=args.output,
        summary=args.summary,
        predictions_output=args.predictions_output,
        target=args.target,
        split_names=tuple(args.splits),
    )
    print(f"Wrote B0 metrics to {args.output}")
    print(f"Wrote B0 predictions to {args.predictions_output}")
    for _, row in metrics.sort_values(["split_name", "subset", "rmse"]).iterrows():
        print(
            f"- {row['split_name']} / {row['subset']} / {row['model']}: "
            f"RMSE={row['rmse']:.3f}, MAE={row['mae']:.3f}, "
            f"Pearson={row['pearson']:.3f}"
        )


if __name__ == "__main__":
    main()
