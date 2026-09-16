"""Compare baseline metrics across standard and tuned baselines.

This script is the single reporting entry point for baseline results. It loads:

- B0 mean baselines
- standard B1/B2/B3 metrics when available
- validation-selected tuned B1/B2/B3 metrics when available
- optional B4/B5/B6 model and ablation metrics when available

For tuned outputs, validation rows are reduced to the best candidate per
split/model and test rows are reduced to the candidate marked
``selected_by_validation``. The final comparison adds improvements relative to
the best B0 model for each split/subset.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUTS = {
    "B0": "results/baselines/b0_metrics.csv",
}

OPTIONAL_INPUTS = {
    "B1": "results/baselines/b1_metrics.csv",
    "B2": "results/baselines/b2_metrics.csv",
    "B3": "results/baselines/b3_metrics.csv",
    "B4_GNN": "results/baselines/b4_gnn_metrics.csv",
    "B5_hybrid_GNN": "results/models/b5_hybrid_gnn_metrics.csv",
    "B6_ablation": "results/ablations/b6_modality_ablation_metrics.csv",
}

OPTIONAL_TUNED_INPUTS = {
    "B1_tuned": "results/baselines/b1_tuning_metrics.csv",
    "B2_tuned": "results/baselines/b2_tuning_metrics.csv",
    "B3_tuned": "results/baselines/b3_tuning_metrics.csv",
}

REQUIRED_COLUMNS = {
    "model",
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
}


def load_metric_table(stage: str, path: str | Path) -> pd.DataFrame:
    """Load one baseline metrics table and add its stage label."""

    metric_path = Path(path)
    if not metric_path.exists():
        raise FileNotFoundError(
            f"Missing metrics file for {stage}: {metric_path}. "
            "Run the corresponding baseline first."
        )

    table = pd.read_csv(metric_path)
    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"{metric_path} is missing columns: {sorted(missing)}")

    table = table.copy()
    table.insert(0, "stage", stage)
    table["params"] = "{}"
    table["source"] = "standard"
    table["model_id"] = table["stage"] + "_" + table["model"].astype(str)
    return table


def load_tuned_metric_table(stage: str, path: str | Path) -> pd.DataFrame:
    """Load one tuning output and keep only reportable rows.

    Validation rows represent all tried candidates. For reporting, we keep the
    best validation candidate per split/model. Test rows are kept only when they
    correspond to the validation-selected candidate.
    """

    metric_path = Path(path)
    if not metric_path.exists():
        raise FileNotFoundError(
            f"Missing tuned metrics file for {stage}: {metric_path}. "
            "Run the corresponding tuning script first."
        )

    table = pd.read_csv(metric_path)
    missing = (REQUIRED_COLUMNS | {"params"}) - set(table.columns)
    if missing:
        raise ValueError(f"{metric_path} is missing columns: {sorted(missing)}")

    validation = (
        table.loc[table["subset"].eq("validation")]
        .sort_values("rmse")
        .groupby(["split_name", "model"], as_index=False)
        .first()
    )

    if "selected_by_validation" in table.columns:
        selected = table["selected_by_validation"].map(is_truthy).fillna(False)
        test = table.loc[table["subset"].eq("test") & selected].copy()
    else:
        test = table.loc[table["subset"].eq("test")].copy()

    reportable = pd.concat([validation, test], ignore_index=True)
    reportable.insert(0, "stage", stage)
    reportable["source"] = "tuned"
    reportable["model_id"] = (
        reportable["stage"] + "_" + reportable["model"].astype(str)
    )
    return reportable


def load_all_metrics(
    inputs: dict[str, str],
    optional_inputs: dict[str, str] | None = None,
    optional_tuned_inputs: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load and concatenate all configured baseline metric tables."""

    tables = [load_metric_table(stage, path) for stage, path in inputs.items()]
    for stage, path in (optional_inputs or {}).items():
        metric_path = Path(path)
        if metric_path.exists():
            tables.append(load_metric_table(stage, metric_path))
    for stage, path in (optional_tuned_inputs or {}).items():
        metric_path = Path(path)
        if metric_path.exists():
            tables.append(load_tuned_metric_table(stage, metric_path))
    return pd.concat(tables, ignore_index=True)


def add_b0_improvements(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add absolute and percentage RMSE/MAE improvements over best B0."""

    output = metrics.copy()

    b0 = output.loc[output["stage"].eq("B0")].copy()
    best_b0 = (
        b0.sort_values("rmse")
        .groupby(["split_name", "subset"], as_index=False)
        .first()[
            [
                "split_name",
                "subset",
                "model_id",
                "rmse",
                "mae",
                "pearson",
                "r2",
            ]
        ]
        .rename(
            columns={
                "model_id": "b0_reference_model",
                "rmse": "b0_reference_rmse",
                "mae": "b0_reference_mae",
                "pearson": "b0_reference_pearson",
                "r2": "b0_reference_r2",
            }
        )
    )

    output = output.merge(best_b0, on=["split_name", "subset"], how="left")
    output["rmse_improvement_vs_b0"] = output["b0_reference_rmse"] - output["rmse"]
    output["mae_improvement_vs_b0"] = output["b0_reference_mae"] - output["mae"]
    output["rmse_improvement_pct_vs_b0"] = (
        output["rmse_improvement_vs_b0"] / output["b0_reference_rmse"] * 100.0
    )
    output["mae_improvement_pct_vs_b0"] = (
        output["mae_improvement_vs_b0"] / output["b0_reference_mae"] * 100.0
    )
    return output


def best_models(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return the best model by RMSE for each split/subset."""

    return (
        metrics.sort_values(["split_name", "subset", "rmse"])
        .groupby(["split_name", "subset"], as_index=False)
        .first()
    )


def make_summary(metrics: pd.DataFrame, best: pd.DataFrame) -> dict[str, Any]:
    """Build a compact JSON summary for the comparison report."""

    best_rows: list[dict[str, Any]] = []
    for row in best.to_dict("records"):
        best_rows.append(
            {
                "split_name": row["split_name"],
                "subset": row["subset"],
                "best_model": row["model_id"],
                "source": row["source"],
                "params": row["params"],
                "rmse": float(row["rmse"]),
                "mae": float(row["mae"]),
                "pearson": none_if_nan(row["pearson"]),
                "spearman": none_if_nan(row["spearman"]),
                "r2": none_if_nan(row["r2"]),
                "b0_reference_model": row["b0_reference_model"],
                "b0_reference_rmse": float(row["b0_reference_rmse"]),
                "rmse_improvement_vs_b0": float(row["rmse_improvement_vs_b0"]),
                "rmse_improvement_pct_vs_b0": float(
                    row["rmse_improvement_pct_vs_b0"]
                ),
            }
        )

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(metrics)),
        "stages": sorted(metrics["stage"].unique().tolist()),
        "best_by_split_subset": best_rows,
    }


def none_if_nan(value: Any) -> float | None:
    """Convert pandas/NumPy NaN values to JSON null."""

    if pd.isna(value):
        return None
    return float(value)


def is_truthy(value: Any) -> bool:
    """Parse permissive CSV truth values."""

    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def compare_baselines(
    inputs: dict[str, str] | None = None,
    optional_inputs: dict[str, str] | None = None,
    optional_tuned_inputs: dict[str, str] | None = None,
    output: str | Path = "results/baselines/baseline_comparison.csv",
    best_output: str | Path = "results/baselines/baseline_best_by_split.csv",
    summary: str | Path = "data/reports/baseline_comparison_summary.json",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create consolidated baseline comparison outputs."""

    if inputs is None:
        inputs = DEFAULT_INPUTS
        if optional_inputs is None:
            optional_inputs = OPTIONAL_INPUTS
        if optional_tuned_inputs is None:
            optional_tuned_inputs = OPTIONAL_TUNED_INPUTS
    else:
        optional_inputs = optional_inputs or {}
        optional_tuned_inputs = optional_tuned_inputs or {}

    metrics = load_all_metrics(inputs, optional_inputs, optional_tuned_inputs)
    comparison = add_b0_improvements(metrics)
    best = best_models(comparison)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)

    best_path = Path(best_output)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(best_path, index=False)

    summary_path = Path(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(make_summary(comparison, best), handle, indent=2)

    return comparison, best


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Compare standard and tuned baseline metrics."
    )
    parser.add_argument("--b0", default=DEFAULT_INPUTS["B0"], help="B0 metrics CSV.")
    parser.add_argument("--b1", default=OPTIONAL_INPUTS["B1"], help="B1 metrics CSV.")
    parser.add_argument("--b2", default=OPTIONAL_INPUTS["B2"], help="B2 metrics CSV.")
    parser.add_argument(
        "--b3",
        default=OPTIONAL_INPUTS["B3"],
        help="Optional B3 metrics CSV. Included only when the file exists.",
    )
    parser.add_argument(
        "--b4-gnn",
        default=OPTIONAL_INPUTS["B4_GNN"],
        help="Optional B4 GNN metrics CSV. Included only when the file exists.",
    )
    parser.add_argument(
        "--b5-hybrid-gnn",
        default=OPTIONAL_INPUTS["B5_hybrid_GNN"],
        help="Optional B5 hybrid GNN metrics CSV. Included only when the file exists.",
    )
    parser.add_argument(
        "--b6-ablation",
        default=OPTIONAL_INPUTS["B6_ablation"],
        help="Optional B6 modality ablation CSV. Included only when the file exists.",
    )
    parser.add_argument(
        "--b1-tuned",
        default=OPTIONAL_TUNED_INPUTS["B1_tuned"],
        help="Optional B1 tuning CSV. Included only when the file exists.",
    )
    parser.add_argument(
        "--b2-tuned",
        default=OPTIONAL_TUNED_INPUTS["B2_tuned"],
        help="Optional B2 tuning CSV. Included only when the file exists.",
    )
    parser.add_argument(
        "--b3-tuned",
        default=OPTIONAL_TUNED_INPUTS["B3_tuned"],
        help="Optional B3 tuning CSV. Included only when the file exists.",
    )
    parser.add_argument(
        "--output",
        default="results/baselines/baseline_comparison.csv",
        help="Output CSV with all baseline rows and B0 improvements.",
    )
    parser.add_argument(
        "--best-output",
        default="results/baselines/baseline_best_by_split.csv",
        help="Output CSV with best model per split/subset.",
    )
    parser.add_argument(
        "--summary",
        default="data/reports/baseline_comparison_summary.json",
        help="Output JSON summary.",
    )
    return parser


def main() -> None:
    """Command-line entry point."""

    args = build_parser().parse_args()
    _comparison, best = compare_baselines(
        inputs={"B0": args.b0},
        optional_inputs={
            "B1": args.b1,
            "B2": args.b2,
            "B3": args.b3,
            "B4_GNN": args.b4_gnn,
            "B5_hybrid_GNN": args.b5_hybrid_gnn,
            "B6_ablation": args.b6_ablation,
        },
        optional_tuned_inputs={
            "B1_tuned": args.b1_tuned,
            "B2_tuned": args.b2_tuned,
            "B3_tuned": args.b3_tuned,
        },
        output=args.output,
        best_output=args.best_output,
        summary=args.summary,
    )

    print(f"Wrote baseline comparison to {args.output}")
    print(f"Wrote best-model table to {args.best_output}")
    print(f"Wrote summary to {args.summary}")
    for row in best.sort_values(["split_name", "subset"]).to_dict("records"):
        print(
            f"- {row['split_name']} / {row['subset']}: "
            f"{row['model_id']} RMSE={row['rmse']:.3f}, "
            f"improvement vs B0={row['rmse_improvement_pct_vs_b0']:.1f}%"
        )


if __name__ == "__main__":
    main()
