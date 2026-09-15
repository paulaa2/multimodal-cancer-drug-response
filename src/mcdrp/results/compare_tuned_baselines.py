"""Compatibility wrapper for the old tuned-only comparison command.

The main reporting command is now ``mcdrp.results.compare_baselines``. This
module remains so older commands still work, but it delegates to the unified
comparison implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mcdrp.results.compare_baselines import compare_baselines


def compare_tuned_baselines(
    b0: str | Path = "results/baselines/b0_metrics.csv",
    b1_tuned: str | Path = "results/baselines/b1_tuning_metrics.csv",
    b2_tuned: str | Path = "results/baselines/b2_tuning_metrics.csv",
    b3_tuned: str | Path = "results/baselines/b3_tuning_metrics.csv",
    output: str | Path = "results/baselines/tuned_baseline_comparison.csv",
    best_output: str | Path = "results/baselines/tuned_baseline_best_by_split.csv",
    summary: str | Path = "data/reports/tuned_baseline_comparison_summary.json",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build tuned baseline comparison outputs through the unified comparator."""

    return compare_baselines(
        inputs={"B0": str(b0)},
        optional_tuned_inputs={
            "B1_tuned": str(b1_tuned),
            "B2_tuned": str(b2_tuned),
            "B3_tuned": str(b3_tuned),
        },
        output=output,
        best_output=best_output,
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper. Prefer: python -m mcdrp.results.compare_baselines"
        )
    )
    parser.add_argument("--b0", default="results/baselines/b0_metrics.csv")
    parser.add_argument("--b1-tuned", default="results/baselines/b1_tuning_metrics.csv")
    parser.add_argument("--b2-tuned", default="results/baselines/b2_tuning_metrics.csv")
    parser.add_argument("--b3-tuned", default="results/baselines/b3_tuning_metrics.csv")
    parser.add_argument(
        "--output",
        default="results/baselines/tuned_baseline_comparison.csv",
    )
    parser.add_argument(
        "--best-output",
        default="results/baselines/tuned_baseline_best_by_split.csv",
    )
    parser.add_argument(
        "--summary",
        default="data/reports/tuned_baseline_comparison_summary.json",
    )
    return parser


def main() -> None:
    """Command-line entry point."""

    args = build_parser().parse_args()
    _comparison, best = compare_tuned_baselines(
        b0=args.b0,
        b1_tuned=args.b1_tuned,
        b2_tuned=args.b2_tuned,
        b3_tuned=args.b3_tuned,
        output=args.output,
        best_output=args.best_output,
        summary=args.summary,
    )

    print(f"Wrote tuned baseline comparison to {args.output}")
    print(f"Wrote tuned best-model table to {args.best_output}")
    print(f"Wrote summary to {args.summary}")
    print("Prefer the unified command: python -m mcdrp.results.compare_baselines")
    for row in best.sort_values(["split_name", "subset"]).to_dict("records"):
        print(
            f"- {row['split_name']} / {row['subset']}: "
            f"{row['model_id']} RMSE={row['rmse']:.3f}, "
            f"improvement vs B0={row['rmse_improvement_pct_vs_b0']:.1f}%"
        )


if __name__ == "__main__":
    main()
