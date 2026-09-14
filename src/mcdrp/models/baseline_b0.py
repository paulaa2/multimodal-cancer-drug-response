"""B0 mean baselines for drug-response prediction.

This baseline intentionally uses no transcriptomics and no chemical structure.
It gives the minimum reference that later models must beat.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from mcdrp.metrics import regression_metrics


DEFAULT_SPLITS = ("random_pair", "cold_cell", "cold_drug")


def predict_global_mean(train: pd.DataFrame, target: str, rows: pd.DataFrame) -> pd.Series:
    return pd.Series(train[target].mean(), index=rows.index)


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
) -> dict[str, Any]:
    metrics = regression_metrics(rows[target].to_numpy(), predictions.to_numpy())
    return {
        "model": model_name,
        "split_name": split_name,
        "subset": subset,
        "n_rows": int(len(rows)),
        "n_cell_lines": int(rows["depmap_id"].nunique()),
        "n_drugs": int(rows["drug_id"].nunique()),
        **metrics,
    }


def run_b0_for_split(
    cohort: pd.DataFrame,
    split_assignments: pd.DataFrame,
    *,
    split_name: str,
    target: str,
) -> list[dict[str, Any]]:
    data = cohort.merge(split_assignments, on="pair_id", how="inner", validate="one_to_one")
    train = data.loc[data["split"].eq("train")].copy()
    results: list[dict[str, Any]] = []

    model_specs = [
        ("global_mean", lambda rows: predict_global_mean(train, target, rows)),
        ("drug_mean", lambda rows: predict_group_mean(train, target, rows, "drug_id")),
        ("cell_mean", lambda rows: predict_group_mean(train, target, rows, "depmap_id")),
    ]

    for subset in ("validation", "test"):
        rows = data.loc[data["split"].eq(subset)].copy()
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
                )
            )

    return results


def run_b0(
    cohort_path: str | Path = "data/processed/cohort_pairs.csv",
    split_dir: str | Path = "data/processed/splits",
    output: str | Path = "results/baselines/b0_metrics.csv",
    summary: str | Path = "data/reports/b0_summary.json",
    *,
    target: str = "ln_ic50",
    split_names: tuple[str, ...] = DEFAULT_SPLITS,
) -> pd.DataFrame:

    cohort = pd.read_csv(cohort_path)
    split_dir = Path(split_dir)

    all_results: list[dict[str, Any]] = []
    for split_name in split_names:
        split_path = split_dir / f"{split_name}.csv"
        assignments = pd.read_csv(split_path)
        all_results.extend(
            run_b0_for_split(
                cohort,
                assignments,
                split_name=split_name,
                target=target,
            )
        )

    metrics = pd.DataFrame(all_results)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)

    summary_data = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "cohort_path": str(cohort_path),
        "split_dir": str(split_dir),
        "output": str(output),
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
        target=args.target,
        split_names=tuple(args.splits),
    )
    print(f"Wrote B0 metrics to {args.output}")
    for _, row in metrics.sort_values(["split_name", "subset", "rmse"]).iterrows():
        print(
            f"- {row['split_name']} / {row['subset']} / {row['model']}: "
            f"RMSE={row['rmse']:.3f}, MAE={row['mae']:.3f}, "
            f"Pearson={row['pearson']:.3f}"
        )


if __name__ == "__main__":
    main()
