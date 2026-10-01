"""Paired replication comparisons must preserve controls and common cohorts."""

import pandas as pd
import pytest

from mcdrp.results.grouped_cv_report import aggregate_folds
from mcdrp.results.replication_report import replication_report


def write_replications(root, *, reference=True, mismatch=False):
    for seed in (7, 42):
        rows = []
        if reference:
            rows.append(
                {"stage": "B0", "model": "mean_effects", "variant": "standard", "rmse": 1.0}
            )
        for variant, rmse in (("standard", 0.8), ("shuffled_targets", 1.2)):
            rows.append(
                {
                    "stage": "B6",
                    "model": "targets_expression_pca",
                    "variant": variant,
                    "rmse": rmse,
                }
            )
        for row in rows:
            row.update(
                split_name="transfer_prism",
                subset="test",
                n_rows=20,
                r2_normalized=0.0,
                spearman_per_drug=0.2,
                r2_per_drug=-0.2,
            )
        if mismatch:
            rows[-1]["n_rows"] = 19
        directory = root / f"seed_{seed}"
        directory.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(directory / "model_metrics.csv", index=False)


def test_target_permutations_remain_separate_in_paired_report(tmp_path):
    root = tmp_path / "runs"
    write_replications(root)
    report = replication_report(str(root), str(tmp_path / "summary.json"))
    paired = {row["variant"]: row for row in report["paired_deltas"]}
    assert paired["standard"]["paired_delta_mean"] == pytest.approx(-0.2)
    assert paired["standard"]["seeds_won"] == 2
    assert paired["shuffled_targets"]["paired_delta_mean"] == pytest.approx(0.2)
    assert paired["shuffled_targets"]["seeds_won"] == 0


@pytest.mark.parametrize(
    "reference,mismatch,message",
    [
        (False, False, "reference"),
        (True, True, "cohort size"),
    ],
)
def test_invalid_paired_replications_are_rejected(tmp_path, reference, mismatch, message):
    root = tmp_path / "runs"
    write_replications(root, reference=reference, mismatch=mismatch)
    with pytest.raises(ValueError, match=message):
        replication_report(str(root), str(tmp_path / "summary.json"))
    assert not (tmp_path / "summary.json").exists()


def test_grouped_aggregation_rejects_missing_comparison_fold():
    metrics = pd.DataFrame(
        {
            "stage": ["B0", "B0", "B6"],
            "model": ["mean_effects", "mean_effects", "targets"],
            "variant": ["standard"] * 3,
            "subset": ["test"] * 3,
            "fold": [0, 1, 0],
            "n_rows": [20] * 3,
            "rmse": [1.0, 1.1, 0.8],
        }
    )
    with pytest.raises(ValueError, match="same CV folds"):
        aggregate_folds(metrics)
