"""A tuning table from another partition must never be silently reused."""

import pandas as pd
import pytest

from mcdrp.models.baseline_b2 import load_selected_xgb_params
from mcdrp.splits.validation import selection_fingerprint


def selection_data():
    cohort = pd.DataFrame({"pair_id": ["a", "b", "c"], "auc": [0.1, 0.2, 0.3]})
    assignments = pd.DataFrame(
        {"pair_id": ["a", "b", "c"], "split": ["train", "validation", "test"]}
    )
    return cohort, assignments


def test_selection_identity_ignores_row_order_and_test_outcomes():
    cohort, assignments = selection_data()
    expected = selection_fingerprint(cohort, assignments, target="auc")
    cohort.loc[2, "auc"] = 0.9
    actual = selection_fingerprint(cohort.iloc[::-1], assignments.iloc[::-1], target="auc")
    assert actual == expected


@pytest.mark.parametrize("change", ["training_target", "partition", "feature_context"])
def test_selection_identity_tracks_selection_inputs(change):
    cohort, assignments = selection_data()
    expected = selection_fingerprint(cohort, assignments, target="auc")
    context = None
    if change == "training_target":
        cohort.loc[0, "auc"] = 0.9
    elif change == "partition":
        assignments.loc[0, "split"] = "validation"
    else:
        context = {"n_components": 128}
    assert selection_fingerprint(cohort, assignments, target="auc", context=context) != expected


@pytest.mark.parametrize("stored", [None, "different_partition"])
def test_unverifiable_tuning_is_rejected(stored, tmp_path):
    path = tmp_path / "tuning.csv"
    pd.DataFrame(
        [
            {
                "split_name": "cold_drug",
                "subset": "test",
                "selected_by_validation": True,
                "params": '{"max_depth": 4}',
                "selection_fingerprint": stored,
            }
        ]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="provenance"):
        load_selected_xgb_params(
            path, fallback={}, expected_fingerprints={"cold_drug": "current_partition"}
        )


def test_matching_tuning_provenance_is_accepted(tmp_path):
    path = tmp_path / "tuning.csv"
    pd.DataFrame(
        [
            {
                "split_name": "cold_drug",
                "subset": "validation",
                "rmse": 0.5,
                "params": '{"max_depth": 4}',
                "selection_fingerprint": "current",
            }
        ]
    ).to_csv(path, index=False)
    selected = load_selected_xgb_params(
        path, fallback={"n_estimators": 700}, expected_fingerprints={"cold_drug": "current"}
    )
    assert selected["cold_drug"] == {"n_estimators": 700, "max_depth": 4}
