"""Grouped K-fold cross-validation for the cold splits.

A single group-level partition cannot support a conclusion about the cold
settings. With 238 drugs and a 15% test fraction, `cold_drug` tests on 36 drugs
and validates on a *different* 36, and because drug potency dominates ln(IC50),
the two subsets end up measuring different things: in the seed-42 split the
validation mean is 1.924 and the test mean is 3.463, a gap of 1.54 on a target
whose standard deviation is about 2.7. `cold_scaffold` shows the same gap with
the sign reversed. Three consequences were visible in the results: tuning B4
made test RMSE worse on exactly those two splits, XGBoost early-stopped at
iteration 0 on `cold_scaffold`, and the B0 baseline alone swung between 2.148
and 3.390 RMSE across seeds.

K-fold over the grouping entity fixes the measurement rather than the model.
Every group is tested exactly once, so the pooled test set covers all 238 drugs
instead of a lucky 36, and the per-fold spread becomes an observable rather
than something a seed silently hides.

Each fold is written as its own directory mirroring the flat split layout::

    data/processed/splits/cv/cold_drug/fold_0/cold_drug.csv
    data/processed/splits/cv/cold_drug/fold_1/cold_drug.csv

so every model consumes a fold unchanged, by pointing ``--split-dir`` at it.
No model or tuner needs to know that cross-validation exists.

Usage:

    python -m mcdrp.splits.grouped_cv --splits cold_drug cold_scaffold
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mcdrp.splits.make_splits import (
    SPLIT_LABELS,
    TISSUE_COLUMN,
    expected_overlap_policy,
    smiles_to_scaffold,
    summarize_split,
    validate_no_leakage,
)

DEFAULT_N_FOLDS = 5

# Splits defined by holding out whole values of one entity column. `random_pair`
# has no grouping entity, and `cold_both` holds out two entities at once, which
# needs a different fold construction than this module implements.
GROUP_COLUMN_BY_SPLIT = {
    "cold_cell": "depmap_id",
    "cold_tissue": TISSUE_COLUMN,
    "cold_drug": "drug_id",
    "cold_scaffold": "chemical_scaffold",
}


def annotate_group_column(cohort: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Return the cohort carrying the column a split groups on.

    Only `cold_scaffold` needs work: its grouping key is derived from SMILES
    rather than present in the cohort.
    """

    if split_name not in GROUP_COLUMN_BY_SPLIT:
        raise ValueError(
            f"{split_name} is not a single-entity group split. "
            f"Supported: {sorted(GROUP_COLUMN_BY_SPLIT)}."
        )

    if split_name != "cold_scaffold":
        column = GROUP_COLUMN_BY_SPLIT[split_name]
        if column not in cohort.columns:
            raise ValueError(f"{split_name} requires a {column} column.")
        if cohort[column].isna().any():
            raise ValueError(f"{split_name} requires a non-null {column} for every pair.")
        return cohort

    if "canonical_smiles" not in cohort.columns:
        raise ValueError("cold_scaffold requires a canonical_smiles column.")

    scaffold_frame = cohort[["drug_id", "canonical_smiles"]].drop_duplicates("drug_id")
    scaffold_map = {
        str(row.drug_id): smiles_to_scaffold(row.canonical_smiles, str(row.drug_id))
        for row in scaffold_frame.itertuples(index=False)
    }
    annotated = cohort.copy()
    annotated["chemical_scaffold"] = (
        annotated["drug_id"].astype(str).map(scaffold_map)
    )
    if annotated["chemical_scaffold"].isna().any():
        raise ValueError("Some drugs could not be assigned to a chemical scaffold.")
    return annotated


def assign_group_folds(
    n_groups: int,
    n_folds: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Assign each group to a fold, keeping fold sizes as even as possible.

    Dealing shuffled groups round-robin keeps fold sizes within one of each
    other, which matters because the smallest fold bounds how much the test
    estimate can be trusted.
    """

    if n_folds < 2:
        raise ValueError("Cross-validation needs at least two folds.")
    if n_groups < n_folds:
        raise ValueError(
            f"Cannot build {n_folds} folds from {n_groups} groups: "
            "every fold must hold at least one group."
        )

    folds = np.empty(n_groups, dtype=int)
    folds[rng.permutation(n_groups)] = np.arange(n_groups) % n_folds
    return folds


def fold_assignments(
    cohort: pd.DataFrame,
    split_name: str,
    *,
    fold: int,
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    """Build one fold's train/validation/test assignment table.

    Fold ``k`` tests on its own groups and validates on those of fold
    ``k + 1``, so validation stays a genuine held-out group set usable for
    early stopping and hyperparameter selection, while every group is tested
    exactly once across the whole run.
    """

    annotated = annotate_group_column(cohort, split_name)
    column = GROUP_COLUMN_BY_SPLIT[split_name]

    groups = pd.Series(annotated[column].dropna().unique()).sort_values().to_numpy()
    # Fold membership must not depend on which fold is being built, so the
    # deal is seeded once per split rather than once per fold.
    membership = assign_group_folds(len(groups), n_folds, np.random.default_rng(seed))

    validation_fold = (fold + 1) % n_folds
    labels = np.where(
        membership == fold,
        "test",
        np.where(membership == validation_fold, "validation", "train"),
    )
    mapping = dict(zip(groups, labels, strict=True))

    assignments = annotated[["pair_id", column]].copy()
    assignments["split"] = assignments[column].map(mapping)
    if assignments["split"].isna().any():
        raise ValueError(f"Some rows could not be assigned for group {column}.")

    if split_name == "cold_scaffold":
        return assignments[["pair_id", "split", "chemical_scaffold"]]
    return assignments[["pair_id", "split"]]


def build_grouped_cv(
    cohort_path: str | Path = "data/processed/cohort_pairs.csv",
    output_dir: str | Path = "data/processed/splits/cv",
    summary_path: str | Path = "data/reports/grouped_cv_summary.json",
    *,
    seed: int = 42,
    n_folds: int = DEFAULT_N_FOLDS,
    split_names: tuple[str, ...] = ("cold_drug", "cold_scaffold"),
) -> dict[str, Any]:
    """Write one split-dir per fold and a summary describing fold balance."""

    cohort = pd.read_csv(cohort_path)
    output_dir = Path(output_dir)

    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(cohort_path).replace("\\", "/"),
        "output_dir": str(output_dir).replace("\\", "/"),
        "seed": int(seed),
        "n_folds": int(n_folds),
        "splits": {},
        "note": (
            "Each fold directory mirrors the flat split layout, so models run "
            "unchanged via --split-dir. Pool the per-fold test predictions to "
            "get one prediction per pair covering every group."
        ),
    }

    for split_name in split_names:
        fold_entries = []
        for fold in range(n_folds):
            assignments = fold_assignments(
                cohort, split_name, fold=fold, n_folds=n_folds, seed=seed
            )
            summary = summarize_split(cohort, assignments)
            validate_no_leakage(split_name, summary)

            fold_dir = output_dir / split_name / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            fold_path = fold_dir / f"{split_name}.csv"
            assignments[["pair_id", "split"]].to_csv(fold_path, index=False)

            fold_entries.append(
                {
                    "fold": fold,
                    "split_dir": str(fold_dir).replace("\\", "/"),
                    "by_split": {
                        label: {
                            key: summary["by_split"][label][key]
                            for key in ("rows", "unique_drugs", "unique_cell_lines")
                        }
                        | {"ln_ic50_mean": summary["by_split"][label]["ln_ic50_mean"]}
                        for label in SPLIT_LABELS
                    },
                }
            )

        report["splits"][split_name] = {
            "policy": expected_overlap_policy(split_name),
            "group_column": GROUP_COLUMN_BY_SPLIT[split_name],
            "folds": fold_entries,
            "test_coverage": test_coverage(cohort, split_name, fold_entries, n_folds),
            "fold_balance": fold_balance(fold_entries),
        }

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


def test_coverage(
    cohort: pd.DataFrame,
    split_name: str,
    fold_entries: list[dict[str, Any]],
    n_folds: int,
) -> dict[str, Any]:
    """Confirm the folds jointly test every group exactly once.

    This is the property that makes pooling folds legitimate, so it is checked
    rather than assumed.
    """

    annotated = annotate_group_column(cohort, split_name)
    column = GROUP_COLUMN_BY_SPLIT[split_name]
    all_groups = set(annotated[column].dropna().unique())

    tested: list[Any] = []
    for entry in fold_entries:
        assignments = pd.read_csv(Path(entry["split_dir"]) / f"{split_name}.csv")
        merged = annotated.merge(assignments, on="pair_id", validate="one_to_one")
        tested.extend(merged.loc[merged["split"].eq("test"), column].unique())

    counts = pd.Series(tested).value_counts() if tested else pd.Series(dtype=int)
    return {
        "n_groups": len(all_groups),
        "n_groups_tested": int(counts.size),
        "groups_never_tested": sorted(str(g) for g in all_groups - set(counts.index)),
        "groups_tested_more_than_once": sorted(
            str(g) for g in counts.loc[counts > 1].index
        ),
        "n_folds": n_folds,
    }


def fold_balance(fold_entries: list[dict[str, Any]]) -> dict[str, float]:
    """Report how much the target mean moves between validation and test.

    The single-partition failure showed up here first: a large gap means
    validation RMSE cannot be used to select anything.
    """

    gaps = [
        abs(
            entry["by_split"]["validation"]["ln_ic50_mean"]
            - entry["by_split"]["test"]["ln_ic50_mean"]
        )
        for entry in fold_entries
    ]
    test_means = [entry["by_split"]["test"]["ln_ic50_mean"] for entry in fold_entries]
    return {
        "validation_test_mean_gap_max": float(np.max(gaps)),
        "validation_test_mean_gap_mean": float(np.mean(gaps)),
        "test_mean_min": float(np.min(test_means)),
        "test_mean_max": float(np.max(test_means)),
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Build grouped K-fold cross-validation split directories."
    )
    parser.add_argument(
        "--cohort",
        default="data/processed/cohort_pairs.csv",
        help="Input cohort CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/splits/cv",
        help="Root directory for per-split, per-fold split directories.",
    )
    parser.add_argument(
        "--summary",
        default="data/reports/grouped_cv_summary.json",
        help="Output JSON summary.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--n-folds",
        type=int,
        default=DEFAULT_N_FOLDS,
        help="Number of folds over the grouping entity.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["cold_drug", "cold_scaffold"],
        choices=sorted(GROUP_COLUMN_BY_SPLIT),
        help="Group splits to build folds for.",
    )
    return parser


def main() -> None:
    """Command-line entry point."""

    args = build_parser().parse_args()
    report = build_grouped_cv(
        cohort_path=args.cohort,
        output_dir=args.output_dir,
        summary_path=args.summary,
        seed=args.seed,
        n_folds=args.n_folds,
        split_names=tuple(args.splits),
    )

    print(f"Wrote grouped CV summary to {args.summary}")
    for name, info in report["splits"].items():
        coverage = info["test_coverage"]
        balance = info["fold_balance"]
        print(
            f"- {name}: {coverage['n_groups_tested']}/{coverage['n_groups']} "
            f"{info['group_column']} values tested across "
            f"{coverage['n_folds']} folds"
        )
        print(
            f"    validation/test target mean gap: "
            f"max {balance['validation_test_mean_gap_max']:.3f}, "
            f"mean {balance['validation_test_mean_gap_mean']:.3f}"
        )
        for entry in info["folds"]:
            rows = {
                label: entry["by_split"][label]["rows"] for label in SPLIT_LABELS
            }
            print(f"    fold {entry['fold']}: {rows}")


if __name__ == "__main__":
    main()
