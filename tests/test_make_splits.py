import pandas as pd
import pytest

from mcdrp.splits.make_splits import (
    SplitSpec,
    make_cold_both_split,
    make_group_split,
    make_tissue_split,
    summarize_split,
    validate_no_leakage,
)


def test_group_split_has_no_train_overlap_for_group_column() -> None:
    cohort = pd.DataFrame(
        {
            "pair_id": [f"p{i}" for i in range(12)],
            "depmap_id": [f"c{i // 2 + 1}" for i in range(12)],
            "drug_id": ["d1", "d2"] * 6,
            "ln_ic50": list(range(12)),
        }
    )
    spec = SplitSpec(validation_size=0.2, test_size=0.2, seed=1)

    assignments = make_group_split(cohort, "depmap_id", spec)
    summary = summarize_split(cohort, assignments)

    assert summary["cell_overlap_train_validation"] == []
    assert summary["cell_overlap_train_test"] == []


def test_cold_both_split_holds_out_cells_and_drugs() -> None:
    rows = []
    for cell_idx in range(6):
        for drug_idx in range(6):
            rows.append(
                {
                    "pair_id": f"c{cell_idx}_d{drug_idx}",
                    "depmap_id": f"c{cell_idx}",
                    "drug_id": f"d{drug_idx}",
                    "ln_ic50": float(cell_idx + drug_idx),
                }
            )
    cohort = pd.DataFrame(
        rows
    )
    spec = SplitSpec(validation_size=0.2, test_size=0.2, seed=3)

    assignments = make_cold_both_split(cohort, spec)
    summary = summarize_split(cohort, assignments)

    validate_no_leakage("cold_both", summary)
    assert set(assignments["split"]) == {"train", "validation", "test", "unused"}
    assert summary["cell_overlap_train_validation"] == []
    assert summary["cell_overlap_train_test"] == []
    assert summary["drug_overlap_train_validation"] == []
    assert summary["drug_overlap_train_test"] == []


def tissue_cohort() -> pd.DataFrame:
    """Build a cohort with four cell lines per tissue across six tissues."""

    rows = []
    for tissue_idx in range(6):
        for cell_idx in range(4):
            rows.append(
                {
                    "pair_id": f"t{tissue_idx}_c{cell_idx}",
                    "depmap_id": f"t{tissue_idx}_c{cell_idx}",
                    "drug_id": f"d{cell_idx}",
                    "oncotree_lineage": f"tissue{tissue_idx}",
                    "ln_ic50": float(tissue_idx + cell_idx),
                }
            )
    return pd.DataFrame(rows)


def test_tissue_split_holds_out_whole_lineages() -> None:
    cohort = tissue_cohort()
    spec = SplitSpec(validation_size=0.2, test_size=0.2, seed=5)

    assignments = make_tissue_split(cohort, spec)
    summary = summarize_split(cohort, assignments)

    validate_no_leakage("cold_tissue", summary)
    assert summary["tissue_overlap_train_validation"] == []
    assert summary["tissue_overlap_train_test"] == []
    # Holding out a tissue necessarily holds out its cell lines too.
    assert summary["cell_overlap_train_test"] == []
    # Drugs are shared across tissues, so they should still overlap.
    assert summary["drug_overlap_train_test"] != []


def test_tissue_split_requires_lineage_column() -> None:
    cohort = tissue_cohort().drop(columns=["oncotree_lineage"])
    spec = SplitSpec(validation_size=0.2, test_size=0.2, seed=5)

    with pytest.raises(ValueError, match="oncotree_lineage"):
        make_tissue_split(cohort, spec)
