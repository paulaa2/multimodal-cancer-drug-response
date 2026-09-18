"""Compute every metric family from stored predictions, in one place.

This is the single point where drug-response metrics are computed. Because it
sees the split assignments, it can fit the mean-effects reference on each
split's training rows and report normalized metrics for *every* model, which
the individual model scripts cannot do: they only ever hold one subset at a
time and would each need to re-derive the reference.

Having one implementation also means the metric definitions cannot drift
between stages, which they previously could, since each model script built its
own metric rows.

Usage:

    python -m mcdrp.results.metrics_report
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mcdrp.metrics import MeanEffectsReference, regression_metrics
from mcdrp.results.predictions import load_predictions
from mcdrp.splits.make_splits import DEFAULT_SPLITS, TISSUE_COLUMN

DEFAULT_PREDICTION_DIR = "results/predictions"
GROUP_COLUMNS = ("stage", "model", "split_name", "subset")


def build_metrics_report(
    predictions: pd.DataFrame,
    cohort: pd.DataFrame,
    split_dir: str | Path,
    *,
    target: str = "ln_ic50",
) -> pd.DataFrame:
    """Compute global, normalized, and stratified metrics for every model.

    One row per stage/model/split/subset, so the result is directly comparable
    across the whole model ladder.
    """

    split_dir = Path(split_dir)
    identity = cohort[["pair_id", "depmap_id", "drug_id", target]].copy()
    if TISSUE_COLUMN in cohort.columns:
        identity[TISSUE_COLUMN] = cohort[TISSUE_COLUMN]

    references = {
        split_name: fit_split_reference(
            cohort, split_dir / f"{split_name}.csv", target=target
        )
        for split_name in sorted(predictions["split_name"].unique())
    }

    rows: list[dict[str, Any]] = []
    annotated = predictions.merge(identity, on="pair_id", how="left", validate="many_to_one")
    if annotated["depmap_id"].isna().any():
        raise ValueError(
            "Some predictions reference pair_ids that are absent from the cohort."
        )

    for keys, group in annotated.groupby(list(GROUP_COLUMNS), dropna=False):
        stage, model, split_name, subset = keys
        reference = references[split_name]
        reference_pred = reference.predict(
            group["depmap_id"].tolist(), group["drug_id"].tolist()
        )
        metrics = regression_metrics(
            group["y_true"].to_numpy(),
            group["y_pred"].to_numpy(),
            reference_pred=reference_pred,
            cell_keys=group["depmap_id"].tolist(),
            drug_keys=group["drug_id"].tolist(),
        )
        rows.append(
            {
                "stage": stage,
                "model": model,
                "model_id": f"{stage}_{model}",
                "split_name": split_name,
                "subset": subset,
                "n_rows": int(len(group)),
                "n_cell_lines": int(group["depmap_id"].nunique()),
                "n_drugs": int(group["drug_id"].nunique()),
                **metrics,
            }
        )

    report = pd.DataFrame(rows)
    return report.sort_values(["split_name", "subset", "rmse"]).reset_index(drop=True)


def fit_split_reference(
    cohort: pd.DataFrame,
    split_path: str | Path,
    *,
    target: str = "ln_ic50",
) -> MeanEffectsReference:
    """Fit the mean-effects reference on the training rows of one split."""

    assignments = pd.read_csv(split_path)
    merged = cohort.merge(assignments, on="pair_id", how="inner", validate="one_to_one")
    train = merged.loc[merged["split"].eq("train")]
    if train.empty:
        raise ValueError(f"{split_path} has no training rows.")
    return MeanEffectsReference.fit(
        train["depmap_id"].tolist(),
        train["drug_id"].tolist(),
        train[target].to_numpy(),
    )


def add_baseline_comparison(report: pd.DataFrame) -> pd.DataFrame:
    """Add RMSE improvement relative to the best B0 baseline per split/subset.

    The reference is the strongest B0 variant, normally `mean_effects`, so the
    comparison is against a demanding baseline rather than a strawman.
    """

    output = report.copy()
    baselines = output.loc[output["stage"].eq("B0")]
    if baselines.empty:
        return output

    best = (
        baselines.sort_values("rmse")
        .groupby(["split_name", "subset"], as_index=False)
        .first()[["split_name", "subset", "model_id", "rmse"]]
        .rename(
            columns={
                "model_id": "b0_reference_model",
                "rmse": "b0_reference_rmse",
            }
        )
    )
    output = output.merge(best, on=["split_name", "subset"], how="left")
    output["rmse_improvement_vs_b0"] = output["b0_reference_rmse"] - output["rmse"]
    output["rmse_improvement_pct_vs_b0"] = (
        output["rmse_improvement_vs_b0"] / output["b0_reference_rmse"] * 100.0
    )
    return output


def best_by_split(report: pd.DataFrame) -> pd.DataFrame:
    """Return the best model per split/subset, ranked by RMSE."""

    return (
        report.sort_values(["split_name", "subset", "rmse"])
        .groupby(["split_name", "subset"], as_index=False)
        .first()
    )


def run_metrics_report(
    prediction_dir: str | Path = DEFAULT_PREDICTION_DIR,
    cohort_path: str | Path = "data/processed/cohort_pairs.csv",
    split_dir: str | Path = "data/processed/splits",
    output: str | Path = "results/reports/model_metrics.csv",
    best_output: str | Path = "results/reports/model_best_by_split.csv",
    summary: str | Path = "data/reports/model_metrics_summary.json",
    *,
    target: str = "ln_ic50",
) -> pd.DataFrame:
    """Load predictions, compute all metrics, and write the report."""

    prediction_paths = sorted(Path(prediction_dir).glob("*.csv"))
    if not prediction_paths:
        raise FileNotFoundError(
            f"No prediction files in {prediction_dir}. Run a model stage first."
        )

    predictions = load_predictions(prediction_paths)
    cohort = pd.read_csv(cohort_path)
    report = add_baseline_comparison(
        build_metrics_report(predictions, cohort, split_dir, target=target)
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

    best = best_by_split(report)
    best_path = Path(best_output)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(best_path, index=False)

    summary_data = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_files": [str(path) for path in prediction_paths],
        "target": target,
        "n_model_split_rows": int(len(report)),
        "splits": sorted(report["split_name"].unique().tolist()),
        "best_by_split_subset": [
            {
                "split_name": row["split_name"],
                "subset": row["subset"],
                "model_id": row["model_id"],
                "rmse": float(row["rmse"]),
                "r2_normalized": none_if_nan(row.get("r2_normalized")),
                "r2_per_drug": none_if_nan(row.get("r2_per_drug")),
            }
            for row in best.to_dict("records")
        ],
    }
    summary_path = Path(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_data, handle, indent=2)

    return report


def none_if_nan(value: Any) -> float | None:
    """Convert NaN to None so the JSON summary stays valid."""

    if value is None or pd.isna(value):
        return None
    return float(value)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Compute global, mean-effect normalized, and stratified metrics for "
            "every model from stored predictions."
        )
    )
    parser.add_argument("--prediction-dir", default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--cohort", default="data/processed/cohort_pairs.csv")
    parser.add_argument("--split-dir", default="data/processed/splits")
    parser.add_argument("--output", default="results/reports/model_metrics.csv")
    parser.add_argument(
        "--best-output", default="results/reports/model_best_by_split.csv"
    )
    parser.add_argument("--summary", default="data/reports/model_metrics_summary.json")
    parser.add_argument("--target", default="ln_ic50")
    return parser


def main() -> None:
    """Command-line entry point."""

    args = build_parser().parse_args()
    report = run_metrics_report(
        prediction_dir=args.prediction_dir,
        cohort_path=args.cohort,
        split_dir=args.split_dir,
        output=args.output,
        best_output=args.best_output,
        summary=args.summary,
        target=args.target,
    )

    print(f"Wrote metrics report to {args.output}")
    print(f"Wrote best-model table to {args.best_output}")
    print(f"Wrote summary to {args.summary}")
    print(f"{'split / subset':<30} {'model':<28} {'RMSE':>7} {'R2norm':>8} {'R2drug':>8}")
    for row in best_by_split(report).to_dict("records"):
        print(
            f"{row['split_name'] + ' / ' + row['subset']:<30} "
            f"{row['model_id']:<28} "
            f"{row['rmse']:>7.3f} "
            f"{format_metric(row.get('r2_normalized')):>8} "
            f"{format_metric(row.get('r2_per_drug')):>8}"
        )


def format_metric(value: Any) -> str:
    """Format a possibly missing metric for console output."""

    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_SPLITS",
    "add_baseline_comparison",
    "best_by_split",
    "build_metrics_report",
    "fit_split_reference",
    "run_metrics_report",
]
