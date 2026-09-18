"""Summarize B6 modality ablation results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_REFERENCE = "morgan_expression_pca"


def make_best_ablation_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Pick the best ablation for each split/subset."""

    return (
        metrics.sort_values(["split_name", "subset", "rmse"])
        .groupby(["split_name", "subset"], as_index=False)
        .first()
    )


def make_delta_table(
    metrics: pd.DataFrame,
    *,
    reference: str = DEFAULT_REFERENCE,
) -> pd.DataFrame:
    """Compare each ablation to a reference ablation within split/subset."""

    reference_rows = metrics.loc[metrics["model"].eq(reference)][
        ["split_name", "subset", "rmse", "mae", "pearson", "r2"]
    ].rename(
        columns={
            "rmse": "reference_rmse",
            "mae": "reference_mae",
            "pearson": "reference_pearson",
            "r2": "reference_r2",
        }
    )
    merged = metrics.merge(reference_rows, on=["split_name", "subset"], how="left")
    merged["rmse_delta_vs_reference"] = merged["rmse"] - merged["reference_rmse"]
    merged["rmse_delta_pct_vs_reference"] = (
        merged["rmse_delta_vs_reference"] / merged["reference_rmse"] * 100.0
    )
    merged["mae_delta_vs_reference"] = merged["mae"] - merged["reference_mae"]
    return merged.sort_values(["split_name", "subset", "rmse"])[
        [
            "model",
            "split_name",
            "subset",
            "n_features",
            "rmse",
            "reference_rmse",
            "rmse_delta_vs_reference",
            "rmse_delta_pct_vs_reference",
            "mae",
            "mae_delta_vs_reference",
            "pearson",
            "r2",
        ]
    ]


def make_modality_matrix(metrics: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Create a compact RMSE matrix: rows=splits, columns=ablations."""

    selected = metrics.loc[metrics["subset"].eq(subset)].copy()
    return selected.pivot_table(
        index="split_name",
        columns="model",
        values="rmse",
        aggfunc="min",
    ).reset_index()


def ablation_report(
    metrics_path: str | Path = "results/ablations/b6_modality_ablation_metrics.csv",
    output_dir: str | Path = "results/ablations",
    summary: str | Path = "data/reports/b6_ablation_report.json",
    *,
    reference: str = DEFAULT_REFERENCE,
    subset: str = "test",
) -> dict[str, Any]:
    """Write B6 ablation summary tables."""

    metrics = pd.read_csv(metrics_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best = make_best_ablation_table(metrics)
    deltas = make_delta_table(metrics, reference=reference)
    matrix = make_modality_matrix(metrics, subset)

    best_path = output_path / "b6_best_ablation_by_split.csv"
    delta_path = output_path / f"b6_ablation_deltas_vs_{reference}.csv"
    matrix_path = output_path / f"b6_ablation_rmse_matrix_{subset}.csv"

    best.to_csv(best_path, index=False)
    deltas.to_csv(delta_path, index=False)
    matrix.to_csv(matrix_path, index=False)

    best_rows: list[dict[str, Any]] = []
    for row in best.sort_values(["split_name", "subset"]).to_dict("records"):
        best_rows.append(
            {
                "split_name": row["split_name"],
                "subset": row["subset"],
                "best_ablation": row["model"],
                "rmse": float(row["rmse"]),
                "mae": float(row["mae"]),
                "pearson": none_if_nan(row["pearson"]),
                "r2": none_if_nan(row["r2"]),
            }
        )

    summary_data = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_path": str(metrics_path),
        "reference": reference,
        "subset": subset,
        "outputs": {
            "best_ablation_by_split": str(best_path),
            "deltas_vs_reference": str(delta_path),
            "rmse_matrix": str(matrix_path),
        },
        "best_by_split_subset": best_rows,
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

    parser = argparse.ArgumentParser(description="Summarize B6 ablation metrics.")
    parser.add_argument(
        "--metrics",
        default="results/ablations/b6_modality_ablation_metrics.csv",
    )
    parser.add_argument("--output-dir", default="results/ablations")
    parser.add_argument(
        "--summary",
        default="data/reports/b6_ablation_report.json",
    )
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--subset", default="test", choices=["validation", "test"])
    return parser


def main() -> None:
    """Command-line entry point."""

    args = build_parser().parse_args()
    summary = ablation_report(
        metrics_path=args.metrics,
        output_dir=args.output_dir,
        summary=args.summary,
        reference=args.reference,
        subset=args.subset,
    )
    print(f"Wrote B6 ablation report to {args.summary}")
    for row in summary["best_by_split_subset"]:
        print(
            f"- {row['split_name']} / {row['subset']}: "
            f"{row['best_ablation']} RMSE={row['rmse']:.3f}"
        )


if __name__ == "__main__":
    main()
