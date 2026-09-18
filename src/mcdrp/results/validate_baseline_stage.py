"""Validate that the baseline stage is ready before moving to GNN models.

This module performs static checks over generated CSV artifacts. It does not
train models. The goal is to catch leakage, malformed metric files, incomplete
tuning outputs, and stale comparison reports before starting the next modelling
phase.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mcdrp.results.compare_baselines import REQUIRED_COLUMNS, is_truthy
from mcdrp.splits.make_splits import DEFAULT_SPLITS, SPLIT_LABELS

STANDARD_METRICS = {
    "B0": "results/baselines/b0_metrics.csv",
    "B1": "results/baselines/b1_metrics.csv",
    "B2": "results/baselines/b2_metrics.csv",
    "B3": "results/baselines/b3_metrics.csv",
    "B4_GNN": "results/baselines/b4_gnn_metrics.csv",
    "B5_hybrid_GNN": "results/models/b5_hybrid_gnn_metrics.csv",
}
TUNING_METRICS = {
    "B1_tuned": "results/baselines/b1_tuning_metrics.csv",
    "B2_tuned": "results/baselines/b2_tuning_metrics.csv",
    "B3_tuned": "results/baselines/b3_tuning_metrics.csv",
}


@dataclass
class CheckResult:
    """Result for one validation check."""

    name: str
    status: str
    message: str
    details: dict[str, Any]


def pass_check(name: str, message: str, **details: Any) -> CheckResult:
    """Create a passing check."""

    return CheckResult(name=name, status="pass", message=message, details=details)


def fail_check(name: str, message: str, **details: Any) -> CheckResult:
    """Create a failing check."""

    return CheckResult(name=name, status="fail", message=message, details=details)


def warn_check(name: str, message: str, **details: Any) -> CheckResult:
    """Create a warning check."""

    return CheckResult(name=name, status="warning", message=message, details=details)


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV with a helpful missing-file error."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")
    return pd.read_csv(csv_path)


def check_required_files(
    paths: Mapping[str, str | Path],
    *,
    required: tuple[str, ...],
    check_name: str,
) -> CheckResult:
    """Check that required files exist and report optional file availability."""

    missing = [name for name in required if not Path(paths[name]).exists()]
    present = [name for name, path in paths.items() if Path(path).exists()]
    if missing:
        return fail_check(
            check_name,
            "Required files are missing.",
            missing=missing,
            present=present,
        )
    return pass_check(
        check_name,
        "Required files are present.",
        present=present,
        optional_missing=[
            name for name, path in paths.items() if name not in present and name not in required
        ],
    )


def validate_cohort(cohort_path: str | Path) -> CheckResult:
    """Validate the model-ready cohort table."""

    required = {
        "pair_id",
        "depmap_id",
        "drug_id",
        "canonical_smiles",
        "ln_ic50",
    }
    cohort = read_csv(cohort_path)
    missing = required - set(cohort.columns)
    duplicate_pairs = int(cohort["pair_id"].duplicated().sum()) if "pair_id" in cohort else -1
    null_counts = {
        column: int(cohort[column].isna().sum())
        for column in required & set(cohort.columns)
    }

    if missing:
        return fail_check(
            "cohort_schema",
            "Cohort is missing required columns.",
            missing=sorted(missing),
        )
    if duplicate_pairs:
        return fail_check(
            "cohort_schema",
            "Cohort contains duplicate pair_id values.",
            duplicate_pair_ids=duplicate_pairs,
        )
    null_required = {column: count for column, count in null_counts.items() if count > 0}
    if null_required:
        return fail_check(
            "cohort_schema",
            "Cohort has null values in required columns.",
            null_counts=null_required,
        )

    return pass_check(
        "cohort_schema",
        "Cohort schema and required values look valid.",
        rows=int(len(cohort)),
        cell_lines=int(cohort["depmap_id"].nunique()),
        drugs=int(cohort["drug_id"].nunique()),
    )


def validate_split(
    cohort: pd.DataFrame,
    split_dir: str | Path,
    split_name: str,
) -> CheckResult:
    """Validate one split assignment file and leakage policy."""

    split_path = Path(split_dir) / f"{split_name}.csv"
    if not split_path.exists():
        return fail_check(
            f"split_{split_name}",
            "Split assignment file is missing.",
            path=str(split_path),
        )

    assignments = pd.read_csv(split_path)
    required = {"pair_id", "split"}
    missing = required - set(assignments.columns)
    if missing:
        return fail_check(
            f"split_{split_name}",
            "Split file is missing required columns.",
            missing=sorted(missing),
        )

    merged = cohort.merge(assignments, on="pair_id", how="inner", validate="one_to_one")
    labels = set(merged["split"].unique())
    missing_labels = sorted(set(SPLIT_LABELS) - labels)
    unknown_labels = sorted(labels - set(SPLIT_LABELS))
    row_count_ok = len(merged) == len(cohort) == len(assignments)

    details: dict[str, Any] = {
        "path": str(split_path),
        "rows": int(len(merged)),
        "by_split": {
            label: int(merged["split"].eq(label).sum()) for label in SPLIT_LABELS
        },
    }
    if not row_count_ok:
        return fail_check(
            f"split_{split_name}",
            "Split assignments do not match cohort rows one-to-one.",
            cohort_rows=int(len(cohort)),
            assignment_rows=int(len(assignments)),
            merged_rows=int(len(merged)),
        )
    if missing_labels or unknown_labels:
        return fail_check(
            f"split_{split_name}",
            "Split labels are incomplete or invalid.",
            missing_labels=missing_labels,
            unknown_labels=unknown_labels,
            **details,
        )

    train = merged.loc[merged["split"].eq("train")]
    validation = merged.loc[merged["split"].eq("validation")]
    test = merged.loc[merged["split"].eq("test")]
    cell_val_overlap = set(train["depmap_id"]) & set(validation["depmap_id"])
    cell_test_overlap = set(train["depmap_id"]) & set(test["depmap_id"])
    drug_val_overlap = set(train["drug_id"]) & set(validation["drug_id"])
    drug_test_overlap = set(train["drug_id"]) & set(test["drug_id"])

    details.update(
        {
            "cell_overlap_train_validation": len(cell_val_overlap),
            "cell_overlap_train_test": len(cell_test_overlap),
            "drug_overlap_train_validation": len(drug_val_overlap),
            "drug_overlap_train_test": len(drug_test_overlap),
        }
    )

    if split_name == "cold_cell" and (cell_val_overlap or cell_test_overlap):
        return fail_check(
            f"split_{split_name}",
            "Cold-cell split leaks cell lines into validation/test.",
            **details,
        )
    if split_name == "cold_drug" and (drug_val_overlap or drug_test_overlap):
        return fail_check(
            f"split_{split_name}",
            "Cold-drug split leaks drugs into validation/test.",
            **details,
        )

    return pass_check(
        f"split_{split_name}",
        "Split file is complete and respects leakage policy.",
        **details,
    )


def validate_metric_table(stage: str, path: str | Path) -> CheckResult:
    """Validate a standard baseline metric CSV."""

    metric_path = Path(path)
    if not metric_path.exists():
        return warn_check(
            f"metrics_{stage}",
            "Metric file is optional and not present.",
            path=str(metric_path),
        )

    table = pd.read_csv(metric_path)
    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        return fail_check(
            f"metrics_{stage}",
            "Metric file is missing required columns.",
            path=str(metric_path),
            missing=sorted(missing),
        )
    return validate_metric_values(
        table,
        f"metrics_{stage}",
        f"{stage} metric file has valid schema and values.",
        path=str(metric_path),
    )


def validate_tuning_table(stage: str, path: str | Path) -> CheckResult:
    """Validate a tuning metric CSV."""

    metric_path = Path(path)
    if not metric_path.exists():
        return warn_check(
            f"tuning_{stage}",
            "Tuning file is optional and not present.",
            path=str(metric_path),
        )

    table = pd.read_csv(metric_path)
    missing = (REQUIRED_COLUMNS | {"params"}) - set(table.columns)
    if missing:
        return fail_check(
            f"tuning_{stage}",
            "Tuning file is missing required columns.",
            path=str(metric_path),
            missing=sorted(missing),
        )

    value_check = validate_metric_values(
        table,
        f"tuning_{stage}",
        f"{stage} tuning file has valid metric values.",
        path=str(metric_path),
    )
    if value_check.status == "fail":
        return value_check

    # Keep the column schema so the pair comparison below works even when no
    # candidate was marked as selected.
    selected_test = table.iloc[:0]
    if "selected_by_validation" in table.columns:
        selected = table["selected_by_validation"].map(is_truthy).fillna(False)
        selected_test = table.loc[table["subset"].eq("test") & selected]

    validation_models = table.loc[table["subset"].eq("validation"), ["split_name", "model"]]
    expected_pairs = {
        tuple(row)
        for row in validation_models.drop_duplicates().itertuples(index=False, name=None)
    }
    selected_pairs = {
        tuple(row)
        for row in selected_test[["split_name", "model"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    missing_selected = sorted(expected_pairs - selected_pairs)

    if missing_selected:
        return fail_check(
            f"tuning_{stage}",
            "Some tuned split/model pairs have no selected test row.",
            path=str(metric_path),
            missing_selected_test_rows=missing_selected,
        )

    return pass_check(
        f"tuning_{stage}",
        "Tuning file has validation candidates and selected test rows.",
        path=str(metric_path),
        rows=int(len(table)),
        selected_test_rows=int(len(selected_test)),
    )


def validate_metric_values(
    table: pd.DataFrame,
    name: str,
    message: str,
    **details: Any,
) -> CheckResult:
    """Validate metric values and required split/subset coverage."""

    allowed_subsets = {"validation", "test"}
    invalid_subsets = sorted(set(table["subset"]) - allowed_subsets)
    required_metric_columns = ["rmse", "mae"]
    optional_metric_columns = ["pearson", "spearman", "r2"]
    null_required_metrics = {
        column: int(table[column].isna().sum())
        for column in required_metric_columns
        if int(table[column].isna().sum()) > 0
    }
    null_optional_metrics = {
        column: int(table[column].isna().sum())
        for column in optional_metric_columns
        if int(table[column].isna().sum()) > 0
    }
    negative_errors = int((table["rmse"].lt(0) | table["mae"].lt(0)).sum())

    if invalid_subsets:
        return fail_check(
            name,
            "Metric file contains invalid subset labels.",
            invalid_subsets=invalid_subsets,
            **details,
        )
    if null_required_metrics:
        return fail_check(
            name,
            "Metric file contains null RMSE or MAE values.",
            null_metrics=null_required_metrics,
            **details,
        )
    if negative_errors:
        return fail_check(
            name,
            "Metric file contains negative RMSE or MAE values.",
            negative_error_rows=negative_errors,
            **details,
        )

    details.update(
        {
            "rows": int(len(table)),
            "splits": sorted(table["split_name"].unique().tolist()),
            "subsets": sorted(table["subset"].unique().tolist()),
            "models": sorted(table["model"].unique().tolist()),
            "optional_null_metrics": null_optional_metrics,
        }
    )
    return pass_check(name, message, **details)


def validate_comparison_outputs(
    comparison_path: str | Path,
    best_path: str | Path,
    summary_path: str | Path,
) -> list[CheckResult]:
    """Validate consolidated comparison outputs."""

    checks: list[CheckResult] = []
    paths = {
        "comparison": Path(comparison_path),
        "best": Path(best_path),
        "summary": Path(summary_path),
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        return [
            fail_check(
                "comparison_outputs",
                "One or more comparison outputs are missing.",
                missing=missing,
            )
        ]

    comparison = pd.read_csv(paths["comparison"])
    best = pd.read_csv(paths["best"])
    required_comparison = REQUIRED_COLUMNS | {
        "stage",
        "model_id",
        "source",
        "b0_reference_model",
        "b0_reference_rmse",
        "rmse_improvement_vs_b0",
    }
    missing_columns = required_comparison - set(comparison.columns)
    if missing_columns:
        checks.append(
            fail_check(
                "comparison_schema",
                "Comparison table is missing required columns.",
                missing=sorted(missing_columns),
            )
        )
    else:
        checks.append(
            pass_check(
                "comparison_schema",
                "Comparison table schema is valid.",
                rows=int(len(comparison)),
                stages=sorted(comparison["stage"].unique().tolist()),
            )
        )

    expected_best = (
        comparison.sort_values(["split_name", "subset", "rmse"])
        .groupby(["split_name", "subset"], as_index=False)
        .first()[["split_name", "subset", "model_id", "rmse"]]
    )
    observed_best = best[["split_name", "subset", "model_id", "rmse"]]
    merged = expected_best.merge(
        observed_best,
        on=["split_name", "subset"],
        suffixes=("_expected", "_observed"),
        how="outer",
    )
    mismatches = merged.loc[
        merged["model_id_expected"].ne(merged["model_id_observed"])
        | merged["rmse_expected"].round(10).ne(merged["rmse_observed"].round(10))
    ]
    if len(mismatches):
        checks.append(
            fail_check(
                "comparison_best_rows",
                "Best-by-split table is stale or inconsistent with comparison table.",
                mismatches=mismatches.to_dict("records"),
            )
        )
    else:
        checks.append(
            pass_check(
                "comparison_best_rows",
                "Best-by-split table matches comparison table.",
                rows=int(len(best)),
            )
        )

    with paths["summary"].open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    summary_rows = summary.get("best_by_split_subset", [])
    if len(summary_rows) != len(best):
        checks.append(
            fail_check(
                "comparison_summary",
                "Summary best-row count does not match best table.",
                summary_rows=len(summary_rows),
                best_rows=int(len(best)),
            )
        )
    else:
        checks.append(
            pass_check(
                "comparison_summary",
                "Comparison summary matches best table size.",
                best_rows=int(len(best)),
            )
        )

    return checks


def validate_baseline_stage(
    cohort_path: str | Path = "data/processed/cohort_pairs.csv",
    split_dir: str | Path = "data/processed/splits",
    comparison_path: str | Path = "results/baselines/baseline_comparison.csv",
    best_path: str | Path = "results/baselines/baseline_best_by_split.csv",
    comparison_summary_path: str | Path = "data/reports/baseline_comparison_summary.json",
    output: str | Path = "data/reports/baseline_stage_validation.json",
    *,
    split_names: tuple[str, ...] = DEFAULT_SPLITS,
) -> dict[str, Any]:
    """Run all baseline-stage validation checks and write a JSON report."""

    checks: list[CheckResult] = []
    checks.append(validate_cohort(cohort_path))

    cohort = read_csv(cohort_path)
    for split_name in split_names:
        checks.append(validate_split(cohort, split_dir, split_name))

    checks.append(
        check_required_files(
            STANDARD_METRICS,
            required=("B0",),
            check_name="standard_metric_files",
        )
    )
    checks.append(
        check_required_files(
            TUNING_METRICS,
            required=(),
            check_name="tuning_metric_files",
        )
    )

    for stage, path in STANDARD_METRICS.items():
        checks.append(validate_metric_table(stage, path))
    for stage, path in TUNING_METRICS.items():
        checks.append(validate_tuning_table(stage, path))

    checks.extend(
        validate_comparison_outputs(
            comparison_path,
            best_path,
            comparison_summary_path,
        )
    )

    status = "pass"
    if any(check.status == "fail" for check in checks):
        status = "fail"
    elif any(check.status == "warning" for check in checks):
        status = "warning"

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "checks": [asdict(check) for check in checks],
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Validate baseline-stage artifacts before starting GNN work."
    )
    parser.add_argument("--cohort", default="data/processed/cohort_pairs.csv")
    parser.add_argument("--split-dir", default="data/processed/splits")
    parser.add_argument(
        "--comparison",
        default="results/baselines/baseline_comparison.csv",
    )
    parser.add_argument(
        "--best",
        default="results/baselines/baseline_best_by_split.csv",
    )
    parser.add_argument(
        "--comparison-summary",
        default="data/reports/baseline_comparison_summary.json",
    )
    parser.add_argument(
        "--output",
        default="data/reports/baseline_stage_validation.json",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=list(DEFAULT_SPLITS),
    )
    return parser


def main() -> None:
    """Command-line entry point."""

    args = build_parser().parse_args()
    report = validate_baseline_stage(
        cohort_path=args.cohort,
        split_dir=args.split_dir,
        comparison_path=args.comparison,
        best_path=args.best,
        comparison_summary_path=args.comparison_summary,
        output=args.output,
        split_names=tuple(args.splits),
    )

    print(f"Wrote baseline-stage validation report to {args.output}")
    print(f"Status: {report['status']}")
    for check in report["checks"]:
        print(f"- {check['status']:>7s} {check['name']}: {check['message']}")


if __name__ == "__main__":
    main()
