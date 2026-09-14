"""Create leakage-safe train/validation/test splits."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SPLIT_LABELS = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitSpec:
    """Split fractions and random seed."""

    validation_size: float
    test_size: float
    seed: int

    @property
    def train_size(self) -> float:
        return 1.0 - self.validation_size - self.test_size


def validate_spec(spec: SplitSpec) -> None:
    """Validate split fractions."""

    if spec.validation_size <= 0 or spec.test_size <= 0:
        raise ValueError("validation_size and test_size must be positive.")
    if spec.train_size <= 0:
        raise ValueError("Split sizes must leave a positive train fraction.")


def assign_labels(n_items: int, spec: SplitSpec, rng: np.random.Generator) -> np.ndarray:
    """Assign train/validation/test labels to shuffled items."""

    if n_items < 3:
        raise ValueError("At least three items are required to create splits.")

    labels = np.full(n_items, "train", dtype=object)
    shuffled = rng.permutation(n_items)

    n_test = max(1, int(round(n_items * spec.test_size)))
    n_validation = max(1, int(round(n_items * spec.validation_size)))
    if n_test + n_validation >= n_items:
        n_test = 1
        n_validation = 1

    test_idx = shuffled[:n_test]
    validation_idx = shuffled[n_test : n_test + n_validation]

    labels[test_idx] = "test"
    labels[validation_idx] = "validation"
    return labels


def make_random_pair_split(cohort: pd.DataFrame, spec: SplitSpec) -> pd.DataFrame:
    """Split individual response pairs at random."""

    rng = np.random.default_rng(spec.seed)
    assignments = cohort[["pair_id"]].copy()
    assignments["split"] = assign_labels(len(assignments), spec, rng)
    return assignments


def make_group_split(
    cohort: pd.DataFrame,
    group_column: str,
    spec: SplitSpec,
) -> pd.DataFrame:
    """Split pairs by holding out whole groups."""

    rng = np.random.default_rng(spec.seed)
    groups = pd.Series(cohort[group_column].dropna().unique()).sort_values().to_numpy()
    group_labels = assign_labels(len(groups), spec, rng)
    mapping = dict(zip(groups, group_labels, strict=True))

    assignments = cohort[["pair_id", group_column]].copy()
    assignments["split"] = assignments[group_column].map(mapping)
    if assignments["split"].isna().any():
        raise ValueError(f"Some rows could not be assigned for group {group_column}.")

    return assignments[["pair_id", "split"]]


def make_split(
    cohort: pd.DataFrame,
    split_name: str,
    spec: SplitSpec,
) -> pd.DataFrame:
    """Create one named split assignment table."""

    if split_name == "random_pair":
        return make_random_pair_split(cohort, spec)
    if split_name == "cold_cell":
        return make_group_split(cohort, "depmap_id", spec)
    if split_name == "cold_drug":
        return make_group_split(cohort, "drug_id", spec)

    raise ValueError(f"Unknown split name: {split_name}")


def summarize_split(cohort: pd.DataFrame, assignments: pd.DataFrame) -> dict[str, Any]:
    """Summarize split sizes and leakage checks."""

    merged = cohort.merge(assignments, on="pair_id", how="inner", validate="one_to_one")
    if len(merged) != len(cohort):
        raise ValueError("Split assignment row count does not match cohort row count.")

    by_split: dict[str, Any] = {}
    for label in SPLIT_LABELS:
        part = merged.loc[merged["split"].eq(label)]
        by_split[label] = {
            "rows": int(len(part)),
            "unique_cell_lines": int(part["depmap_id"].nunique()),
            "unique_drugs": int(part["drug_id"].nunique()),
            "ln_ic50_mean": float(part["ln_ic50"].mean()),
            "ln_ic50_std": float(part["ln_ic50"].std()),
        }

    train = merged.loc[merged["split"].eq("train")]
    validation = merged.loc[merged["split"].eq("validation")]
    test = merged.loc[merged["split"].eq("test")]

    return {
        "by_split": by_split,
        "cell_overlap_train_validation": sorted(
            set(train["depmap_id"]) & set(validation["depmap_id"])
        ),
        "cell_overlap_train_test": sorted(set(train["depmap_id"]) & set(test["depmap_id"])),
        "drug_overlap_train_validation": sorted(
            set(train["drug_id"]) & set(validation["drug_id"])
        ),
        "drug_overlap_train_test": sorted(set(train["drug_id"]) & set(test["drug_id"])),
    }


def expected_overlap_policy(split_name: str) -> dict[str, bool]:
    """Describe which entity overlaps are expected for a split."""

    if split_name == "cold_cell":
        return {"cell_overlap_allowed": False, "drug_overlap_allowed": True}
    if split_name == "cold_drug":
        return {"cell_overlap_allowed": True, "drug_overlap_allowed": False}
    return {"cell_overlap_allowed": True, "drug_overlap_allowed": True}


def validate_no_leakage(split_name: str, summary: dict[str, Any]) -> None:
    """Fail if a cold split leaks held-out groups into train."""

    policy = expected_overlap_policy(split_name)
    if not policy["cell_overlap_allowed"]:
        if summary["cell_overlap_train_validation"] or summary["cell_overlap_train_test"]:
            raise ValueError(f"Cell-line leakage detected in {split_name}.")
    if not policy["drug_overlap_allowed"]:
        if summary["drug_overlap_train_validation"] or summary["drug_overlap_train_test"]:
            raise ValueError(f"Drug leakage detected in {split_name}.")


def build_splits(
    cohort_path: str | Path = "data/processed/cohort_pairs.csv",
    output_dir: str | Path = "data/processed/splits",
    summary_path: str | Path = "data/reports/splits_summary.json",
    *,
    seed: int = 42,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    split_names: tuple[str, ...] = ("random_pair", "cold_cell", "cold_drug"),
) -> dict[str, Any]:
    """Build split assignment files."""

    spec = SplitSpec(
        validation_size=float(validation_size),
        test_size=float(test_size),
        seed=int(seed),
    )
    validate_spec(spec)

    cohort_path = Path(cohort_path)
    output_dir = Path(output_dir)
    summary_path = Path(summary_path)

    cohort = pd.read_csv(cohort_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_report: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "input": str(cohort_path).replace("\\", "/"),
        "output_dir": str(output_dir).replace("\\", "/"),
        "seed": spec.seed,
        "validation_size": spec.validation_size,
        "test_size": spec.test_size,
        "splits": {},
    }

    for split_name in split_names:
        assignments = make_split(cohort, split_name, spec)
        summary = summarize_split(cohort, assignments)
        validate_no_leakage(split_name, summary)

        output_path = output_dir / f"{split_name}.csv"
        assignments.to_csv(output_path, index=False)
        summary_report["splits"][split_name] = {
            "path": str(output_path).replace("\\", "/"),
            "policy": expected_overlap_policy(split_name),
            "summary": compact_summary(split_name, summary),
        }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_report, handle, indent=2)

    return summary_report


def compact_summary(split_name: str, summary: dict[str, Any]) -> dict[str, Any]:
    """Keep the JSON summary readable while preserving leakage evidence."""

    policy = expected_overlap_policy(split_name)
    compact = {"by_split": summary["by_split"]}
    if not policy["cell_overlap_allowed"]:
        compact["cell_overlap_train_validation_count"] = len(
            summary["cell_overlap_train_validation"]
        )
        compact["cell_overlap_train_test_count"] = len(summary["cell_overlap_train_test"])
    if not policy["drug_overlap_allowed"]:
        compact["drug_overlap_train_validation_count"] = len(
            summary["drug_overlap_train_validation"]
        )
        compact["drug_overlap_train_test_count"] = len(summary["drug_overlap_train_test"])
    return compact


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description="Create train/validation/test splits.")
    parser.add_argument(
        "--cohort",
        default="data/processed/cohort_pairs.csv",
        help="Input cohort CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/splits",
        help="Directory where split CSVs will be written.",
    )
    parser.add_argument(
        "--summary",
        default="data/reports/splits_summary.json",
        help="Output JSON summary.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.15,
        help="Validation fraction.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Test fraction.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["random_pair", "cold_cell", "cold_drug"],
        choices=["random_pair", "cold_cell", "cold_drug"],
        help="Split types to create.",
    )
    return parser


def main() -> None:
    """Command-line entry point."""

    args = build_parser().parse_args()
    summary_report = build_splits(
        cohort_path=args.cohort,
        output_dir=args.output_dir,
        summary_path=args.summary,
        seed=args.seed,
        validation_size=args.validation_size,
        test_size=args.test_size,
        split_names=tuple(args.splits),
    )
    print(f"Wrote split summary to {args.summary}")
    for name, info in summary_report["splits"].items():
        rows = {
            split: values["rows"]
            for split, values in info["summary"]["by_split"].items()
        }
        print(f"- {name}: {rows}")


if __name__ == "__main__":
    main()
