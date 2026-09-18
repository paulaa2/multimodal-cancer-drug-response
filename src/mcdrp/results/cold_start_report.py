"""Build cold-start evaluation tables from consolidated model metrics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SPLIT_ORDER = ("random_pair", "cold_cell", "cold_drug", "cold_scaffold", "cold_both")


def load_split_summary(path: str | Path) -> pd.DataFrame:
    """Flatten the split summary JSON into one row per split/subset."""

    summary_path = Path(path)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    rows: list[dict[str, Any]] = []
    for split_name, split_info in summary["splits"].items():
        split_summary = split_info["summary"]
        for subset, values in split_summary["by_split"].items():
            rows.append(
                {
                    "split_name": split_name,
                    "subset": subset,
                    "split_rows": values["rows"],
                    "split_unique_cell_lines": values["unique_cell_lines"],
                    "split_unique_drugs": values["unique_drugs"],
                    "split_target_mean": values["ln_ic50_mean"],
                    "split_target_std": values["ln_ic50_std"],
                    "unused_rows": split_summary.get("unused_rows", 0),
                }
            )
    return pd.DataFrame(rows)


def make_model_split_table(metrics: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Create one table with model metrics by split for a subset."""

    selected = metrics.loc[metrics["subset"].eq(subset)].copy()
    return selected.sort_values(["model_id", "split_name", "rmse"])[
        [
            "model_id",
            "stage",
            "source",
            "split_name",
            "subset",
            "n_rows",
            "n_cell_lines",
            "n_drugs",
            "rmse",
            "mae",
            "pearson",
            "spearman",
            "r2",
            "rmse_improvement_pct_vs_b0",
        ]
    ]


def make_performance_drop_table(metrics: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Measure RMSE degradation from random_pair to each cold-start split."""

    selected = metrics.loc[metrics["subset"].eq(subset)].copy()
    random_rows = selected.loc[selected["split_name"].eq("random_pair")][
        ["model_id", "rmse"]
    ].rename(columns={"rmse": "random_pair_rmse"})
    merged = selected.merge(random_rows, on="model_id", how="inner")
    merged = merged.loc[~merged["split_name"].eq("random_pair")].copy()
    merged["rmse_drop_vs_random_pair"] = merged["rmse"] - merged["random_pair_rmse"]
    merged["rmse_drop_pct_vs_random_pair"] = (
        merged["rmse_drop_vs_random_pair"] / merged["random_pair_rmse"] * 100.0
    )
    return merged.sort_values(["split_name", "rmse_drop_vs_random_pair"])[
        [
            "model_id",
            "split_name",
            "subset",
            "random_pair_rmse",
            "rmse",
            "rmse_drop_vs_random_pair",
            "rmse_drop_pct_vs_random_pair",
            "pearson",
            "r2",
        ]
    ]


def make_best_table(metrics: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Pick the best model by split for a subset."""

    selected = metrics.loc[metrics["subset"].eq(subset)].copy()
    return (
        selected.sort_values(["split_name", "rmse"])
        .groupby("split_name", as_index=False)
        .first()
    )


def cold_start_report(
    comparison_path: str | Path = "results/baselines/baseline_comparison.csv",
    splits_summary_path: str | Path = "data/reports/splits_summary.json",
    output_dir: str | Path = "results/reports",
    summary: str | Path = "data/reports/cold_start_evaluation_summary.json",
    *,
    subset: str = "test",
) -> dict[str, Any]:
    """Create cold-start evaluation CSVs and a compact JSON summary."""

    metrics = pd.read_csv(comparison_path)
    split_info = load_split_summary(splits_summary_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_split = make_model_split_table(metrics, subset)
    performance_drop = make_performance_drop_table(metrics, subset)
    best = make_best_table(metrics, subset)

    model_split_path = output_path / f"cold_start_model_metrics_{subset}.csv"
    drop_path = output_path / f"cold_start_rmse_drop_{subset}.csv"
    best_path = output_path / f"cold_start_best_models_{subset}.csv"
    split_info_path = output_path / "split_composition.csv"

    model_split.to_csv(model_split_path, index=False)
    performance_drop.to_csv(drop_path, index=False)
    best.to_csv(best_path, index=False)
    split_info.to_csv(split_info_path, index=False)

    best_rows: list[dict[str, Any]] = []
    for row in best.sort_values("split_name").to_dict("records"):
        best_rows.append(
            {
                "split_name": row["split_name"],
                "model_id": row["model_id"],
                "rmse": float(row["rmse"]),
                "mae": float(row["mae"]),
                "pearson": none_if_nan(row["pearson"]),
                "r2": none_if_nan(row["r2"]),
            }
        )

    summary_data = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_path": str(comparison_path),
        "splits_summary_path": str(splits_summary_path),
        "subset": subset,
        "outputs": {
            "model_split_metrics": str(model_split_path),
            "rmse_drop": str(drop_path),
            "best_models": str(best_path),
            "split_composition": str(split_info_path),
        },
        "best_by_split": best_rows,
    }
    summary_path = Path(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_data, handle, indent=2)
    return summary_data


def none_if_nan(value: Any) -> float | None:
    """Convert NaN to JSON null."""

    if pd.isna(value):
        return None
    return float(value)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description="Build cold-start evaluation report.")
    parser.add_argument(
        "--comparison",
        default="results/baselines/baseline_comparison.csv",
    )
    parser.add_argument(
        "--splits-summary",
        default="data/reports/splits_summary.json",
    )
    parser.add_argument("--output-dir", default="results/reports")
    parser.add_argument(
        "--summary",
        default="data/reports/cold_start_evaluation_summary.json",
    )
    parser.add_argument("--subset", default="test", choices=["validation", "test"])
    return parser


def main() -> None:
    """Command-line entry point."""

    args = build_parser().parse_args()
    summary = cold_start_report(
        comparison_path=args.comparison,
        splits_summary_path=args.splits_summary,
        output_dir=args.output_dir,
        summary=args.summary,
        subset=args.subset,
    )
    print(f"Wrote cold-start summary to {args.summary}")
    for row in summary["best_by_split"]:
        print(f"- {row['split_name']}: {row['model_id']} RMSE={row['rmse']:.3f}")


if __name__ == "__main__":
    main()
