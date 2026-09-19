"""Tests for grouped K-fold cross-validation over the cold splits."""

import numpy as np
import pandas as pd
import pytest

from mcdrp.results.grouped_cv_report import aggregate_folds, discover_folds
from mcdrp.splits.grouped_cv import (
    annotate_group_column,
    assign_group_folds,
    build_grouped_cv,
    fold_assignments,
)


def cohort(n_cells: int = 12, n_drugs: int = 20) -> pd.DataFrame:
    """A dense cell-by-drug cohort with a drug-dominated target."""

    rng = np.random.default_rng(0)
    rows = []
    for cell in range(n_cells):
        for drug in range(n_drugs):
            rows.append(
                {
                    "pair_id": f"c{cell}_d{drug}",
                    "depmap_id": f"ACH-{cell:06d}",
                    "drug_id": f"D{drug:03d}",
                    "oncotree_lineage": f"tissue_{cell % 4}",
                    "ln_ic50": 3.0 * drug + float(cell) + rng.normal(0, 0.2),
                }
            )
    return pd.DataFrame(rows)


class TestFoldAssignment:
    def test_folds_are_balanced_within_one_group(self) -> None:
        folds = assign_group_folds(23, 5, np.random.default_rng(0))
        sizes = np.bincount(folds, minlength=5)

        assert sizes.sum() == 23
        assert sizes.max() - sizes.min() <= 1

    def test_too_few_groups_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="every fold must hold at least one"):
            assign_group_folds(3, 5, np.random.default_rng(0))

    def test_single_fold_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least two folds"):
            assign_group_folds(10, 1, np.random.default_rng(0))


class TestFoldMembership:
    def test_membership_does_not_depend_on_which_fold_is_built(self) -> None:
        """Fold k validates on the groups that fold k+1 tests on.

        That identity only holds if membership is dealt once per split. If it
        were re-dealt per fold, a group could be tested twice or never, and
        pooling the folds would no longer be legitimate.
        """

        frame = cohort()
        first = fold_assignments(frame, "cold_drug", fold=0, n_folds=5, seed=42)
        second = fold_assignments(frame, "cold_drug", fold=1, n_folds=5, seed=42)

        validated_first = set(first.loc[first["split"].eq("validation"), "pair_id"])
        tested_second = set(second.loc[second["split"].eq("test"), "pair_id"])
        assert validated_first == tested_second

    def test_no_drug_appears_in_both_train_and_test(self) -> None:
        frame = cohort()
        for fold in range(5):
            assignments = fold_assignments(
                frame, "cold_drug", fold=fold, n_folds=5, seed=42
            )
            merged = frame.merge(assignments, on="pair_id", validate="one_to_one")
            train = set(merged.loc[merged["split"].eq("train"), "drug_id"])
            test = set(merged.loc[merged["split"].eq("test"), "drug_id"])
            validation = set(merged.loc[merged["split"].eq("validation"), "drug_id"])

            assert not train & test
            assert not train & validation
            assert not validation & test

    def test_every_group_is_tested_exactly_once_across_folds(self) -> None:
        frame = cohort()
        tested: list[str] = []
        for fold in range(5):
            assignments = fold_assignments(
                frame, "cold_drug", fold=fold, n_folds=5, seed=42
            )
            merged = frame.merge(assignments, on="pair_id", validate="one_to_one")
            tested.extend(merged.loc[merged["split"].eq("test"), "drug_id"].unique())

        counts = pd.Series(tested).value_counts()
        assert set(counts.index) == set(frame["drug_id"].unique())
        assert (counts == 1).all()

    def test_unsupported_split_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a single-entity group split"):
            annotate_group_column(cohort(), "cold_both")


class TestBuildGroupedCV:
    def test_fold_directories_mirror_the_flat_split_layout(self, tmp_path) -> None:
        """Models locate a split at ``<split_dir>/<split_name>.csv``.

        Preserving that shape is what lets every model and tuner consume a
        fold without modification.
        """

        frame = cohort()
        cohort_path = tmp_path / "cohort.csv"
        frame.to_csv(cohort_path, index=False)

        report = build_grouped_cv(
            cohort_path=cohort_path,
            output_dir=tmp_path / "cv",
            summary_path=tmp_path / "summary.json",
            n_folds=4,
            split_names=("cold_drug",),
        )

        for fold in range(4):
            path = tmp_path / "cv" / "cold_drug" / f"fold_{fold}" / "cold_drug.csv"
            assert path.exists()
            assignments = pd.read_csv(path)
            assert list(assignments.columns) == ["pair_id", "split"]
            assert set(assignments["split"]) == {"train", "validation", "test"}

        coverage = report["splits"]["cold_drug"]["test_coverage"]
        assert coverage["groups_never_tested"] == []
        assert coverage["groups_tested_more_than_once"] == []
        assert coverage["n_groups_tested"] == coverage["n_groups"]

    def test_tissue_folds_respect_tissue_groups(self, tmp_path) -> None:
        frame = cohort()
        cohort_path = tmp_path / "cohort.csv"
        frame.to_csv(cohort_path, index=False)

        report = build_grouped_cv(
            cohort_path=cohort_path,
            output_dir=tmp_path / "cv",
            summary_path=tmp_path / "summary.json",
            n_folds=2,
            split_names=("cold_tissue",),
        )

        assert report["splits"]["cold_tissue"]["group_column"] == "oncotree_lineage"
        assert report["splits"]["cold_tissue"]["test_coverage"]["n_groups"] == 4


class TestAggregation:
    def fold_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "fold": [0, 1, 2, 0, 1, 2],
                "stage": ["B0"] * 6,
                "model": ["mean_effects"] * 6,
                "variant": ["standard"] * 6,
                "subset": ["test"] * 3 + ["validation"] * 3,
                "n_rows": [100, 200, 300, 10, 20, 30],
                "rmse": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "r2_normalized": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            }
        )

    def test_mean_and_spread_are_reported_per_subset(self) -> None:
        summary = aggregate_folds(self.fold_table())
        test = summary.loc[summary["subset"].eq("test")].iloc[0]

        assert test["n_folds"] == 3
        assert test["n_rows_total"] == 600
        assert test["rmse_mean"] == pytest.approx(2.0)
        assert test["rmse_std"] == pytest.approx(1.0)
        assert test["rmse_min"] == pytest.approx(1.0)
        assert test["rmse_max"] == pytest.approx(3.0)
        assert test["folds"] == "0,1,2"

    def test_no_significance_columns_are_produced(self) -> None:
        """Folds share training data, so a p-value here would be pseudoreplication."""

        summary = aggregate_folds(self.fold_table())
        assert not [c for c in summary.columns if "p_value" in c or "pvalue" in c]

    def test_discover_folds_ignores_unrelated_files(self, tmp_path) -> None:
        for name in ("fold_0.csv", "fold_10.csv", "fold_2.csv", "fold_x.csv", "n.csv"):
            (tmp_path / name).write_text("", encoding="utf-8")

        assert discover_folds(tmp_path) == [0, 2, 10]
