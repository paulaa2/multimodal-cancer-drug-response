import pandas as pd

from mcdrp.splits.make_splits import SplitSpec, make_group_split, summarize_split


def test_group_split_has_no_train_overlap_for_group_column() -> None:
    cohort = pd.DataFrame(
        {
            "pair_id": [f"p{i}" for i in range(12)],
            "depmap_id": ["c1", "c1", "c2", "c2", "c3", "c3", "c4", "c4", "c5", "c5", "c6", "c6"],
            "drug_id": ["d1", "d2"] * 6,
            "ln_ic50": list(range(12)),
        }
    )
    spec = SplitSpec(validation_size=0.2, test_size=0.2, seed=1)

    assignments = make_group_split(cohort, "depmap_id", spec)
    summary = summarize_split(cohort, assignments)

    assert summary["cell_overlap_train_validation"] == []
    assert summary["cell_overlap_train_test"] == []
