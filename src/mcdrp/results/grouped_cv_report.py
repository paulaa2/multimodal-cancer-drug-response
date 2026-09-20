"""Aggregate metrics across grouped cross-validation folds.

Metrics are computed per fold and then averaged, rather than computed once on
the pooled predictions. The reason is the mean-effects reference: each fold has
its own training rows, so `r2_normalized` is only defined relative to that
fold's reference. Pooling first would compare every fold against one arbitrary
reference and quietly change what the normalized metrics mean.

Averaging per-fold metrics also produces the thing a single partition could
never give: the spread across folds. That spread is the honest error bar on a
cold-split claim, and it is reported alongside the mean.

As in `mcdrp.experiments.multiseed`, no p-values are produced. Folds share
training data and the units of replication are drugs and cell lines, so a
standard deviation here describes dispersion, not significance.

Usage:

    python -m mcdrp.results.grouped_cv_report --split-name cold_drug
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from mcdrp.results.metrics_report import build_metrics_report
from mcdrp.results.predictions import load_predictions

AGGREGATE_KEYS = ("stage", "model", "variant", "subset")
AGGREGATED_METRICS = (
    "rmse",
    "mae",
    "pearson",
    "spearman",
    "r2",
    "r2_normalized",
    "pearson_per_drug",
    "spearman_per_drug",
    "r2_per_drug",
    "pearson_per_cell",
    "spearman_per_cell",
    "r2_per_cell",
)


def discover_folds(predictions_dir: str | Path) -> list[int]:
    """Return the fold indices that have prediction files, in order."""

    directory = Path(predictions_dir)
    folds = []
    for path in directory.glob("fold_*"):
        stem = path.name.removesuffix(".csv")
        suffix = stem.removeprefix("fold_")
        if suffix.isdigit():
            folds.append(int(suffix))
    return sorted(set(folds))


def fold_metrics(
    split_name: str,
    *,
    cv_dir: str | Path,
    predictions_dir: str | Path,
    cohort: pd.DataFrame,
    folds: list[int],
    target: str = "ln_ic50",
) -> pd.DataFrame:
    """Compute one metrics table per fold, against that fold's own reference."""

    cv_dir = Path(cv_dir)
    predictions_dir = Path(predictions_dir)

    tables = []
    for fold in folds:
        # A fold directory holds one prediction file per model, so several
        # models can be compared on identical held-out groups. A bare
        # fold_<k>.csv is still accepted for single-model runs.
        fold_dir = predictions_dir / f"fold_{fold}"
        if fold_dir.is_dir():
            prediction_paths = sorted(fold_dir.glob("*.csv"))
        else:
            single = predictions_dir / f"fold_{fold}.csv"
            prediction_paths = [single] if single.exists() else []
        if not prediction_paths:
            raise FileNotFoundError(
                f"No predictions for fold {fold} of {split_name} under "
                f"{predictions_dir}."
            )

        fold_split_dir = cv_dir / f"fold_{fold}"
        if not (fold_split_dir / f"{split_name}.csv").exists():
            raise FileNotFoundError(
                f"Missing split assignments at {fold_split_dir / f'{split_name}.csv'}. "
                "Build the folds with mcdrp.splits.grouped_cv first."
            )

        predictions = load_predictions(prediction_paths)
        table = build_metrics_report(
            predictions, cohort, fold_split_dir, target=target
        )
        table.insert(0, "fold", fold)
        tables.append(table)

    return pd.concat(tables, ignore_index=True)


def aggregate_folds(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average each metric over folds and report its spread.

    The standard deviation describes how much the estimate moves when the
    held-out groups change. It is not a significance test.
    """

    present = [column for column in AGGREGATED_METRICS if column in metrics.columns]
    grouped = metrics.groupby(list(AGGREGATE_KEYS), as_index=False, dropna=False)

    summary = grouped.agg(
        n_folds=("fold", "nunique"),
        n_rows_total=("n_rows", "sum"),
    )
    for column in present:
        stats = grouped[column].agg(["mean", "std", "min", "max"])
        for stat in ("mean", "std", "min", "max"):
            summary[f"{column}_{stat}"] = stats[stat]

    summary["folds"] = grouped["fold"].agg(
        lambda values: ",".join(str(value) for value in sorted(values))
    )["fold"]
    return summary.sort_values(["subset", "rmse_mean"]).reset_index(drop=True)


def run_grouped_cv_report(
    split_name: str,
    cv_dir: str | Path | None = None,
    predictions_dir: str | Path | None = None,
    cohort_path: str | Path = "data/processed/cohort_pairs.csv",
    output: str | Path | None = None,
    summary: str | Path | None = None,
    *,
    target: str = "ln_ic50",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-fold and aggregated metrics for one grouped CV split."""

    cv_dir = Path(cv_dir or f"data/processed/splits/cv/{split_name}")
    predictions_dir = Path(predictions_dir or f"results/predictions/cv/{split_name}")
    output = Path(output or f"results/reports/cv/{split_name}_fold_metrics.csv")
    summary = Path(summary or f"results/reports/cv/{split_name}_aggregated.csv")

    folds = discover_folds(predictions_dir)
    if not folds:
        raise FileNotFoundError(
            f"No fold prediction files under {predictions_dir}. Run a model over "
            f"each fold directory in {cv_dir} first."
        )

    cohort = pd.read_csv(cohort_path)
    per_fold = fold_metrics(
        split_name,
        cv_dir=cv_dir,
        predictions_dir=predictions_dir,
        cohort=cohort,
        folds=folds,
        target=target,
    )
    aggregated = aggregate_folds(per_fold)

    output.parent.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(output, index=False)
    aggregated.to_csv(summary, index=False)

    report_path = summary.with_suffix(".json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "split_name": split_name,
                "folds": folds,
                "cv_dir": str(cv_dir).replace("\\", "/"),
                "predictions_dir": str(predictions_dir).replace("\\", "/"),
                "fold_metrics_path": str(output).replace("\\", "/"),
                "aggregated_path": str(summary).replace("\\", "/"),
                "hypothesis_tests": "none",
                "note": (
                    "Metrics are computed per fold against that fold's own "
                    "mean-effects reference, then averaged. The standard "
                    "deviation is dispersion across held-out group sets, not a "
                    "significance test."
                ),
            },
            handle,
            indent=2,
        )

    return per_fold, aggregated


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Aggregate model metrics across grouped CV folds."
    )
    parser.add_argument(
        "--split-name",
        required=True,
        help="Group split whose folds should be aggregated, e.g. cold_drug.",
    )
    parser.add_argument(
        "--cv-dir",
        default=None,
        help="Fold split directories. Defaults to data/processed/splits/cv/<split>.",
    )
    parser.add_argument(
        "--predictions-dir",
        default=None,
        help="Per-fold predictions. Defaults to results/predictions/cv/<split>.",
    )
    parser.add_argument(
        "--cohort",
        default="data/processed/cohort_pairs.csv",
        help="Input cohort CSV.",
    )
    parser.add_argument("--output", default=None, help="Per-fold metrics CSV.")
    parser.add_argument("--summary", default=None, help="Aggregated metrics CSV.")
    parser.add_argument("--target", default="ln_ic50", help="Regression target column.")
    return parser


def main() -> None:
    """Command-line entry point."""

    args = build_parser().parse_args()
    _per_fold, aggregated = run_grouped_cv_report(
        args.split_name,
        cv_dir=args.cv_dir,
        predictions_dir=args.predictions_dir,
        cohort_path=args.cohort,
        output=args.output,
        summary=args.summary,
        target=args.target,
    )

    test_rows = aggregated.loc[aggregated["subset"].eq("test")]
    print(f"Aggregated {args.split_name} over folds: {len(test_rows)} model rows")
    for row in test_rows.to_dict("records"):
        print(
            f"- {row['stage']}_{row['model']} ({row['variant']}): "
            f"RMSE {row['rmse_mean']:.3f} +/- {row['rmse_std']:.3f} "
            f"over {row['n_folds']} folds"
        )


if __name__ == "__main__":
    main()
