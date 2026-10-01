"""Regression tests for scientific integrity issues found in the repository audit."""

import numpy as np
import pandas as pd
import pytest

from mcdrp.features.expression import load_expression_matrix
from mcdrp.metrics import regression_metrics
from mcdrp.results.metrics_report import add_baseline_comparison, build_metrics_report
from mcdrp.results.predictions import prediction_frame
from mcdrp.results.transfer_diagnostics import assay_concordance
from mcdrp.splits.grouped_cv import assign_group_folds, fold_assignments
from mcdrp.splits.validation import validate_assignments


def test_expression_selects_default_profile_independently_of_file_order(tmp_path):
    rows = pd.DataFrame(
        {
            "ModelID": ["c1", "c1", "c2"],
            "IsDefaultEntryForModel": ["No", "Yes", "Yes"],
            "TP53 (7157)": [99.0, 1.0, 2.0],
        }
    )
    for i, frame in enumerate((rows, rows.iloc[::-1])):
        path = tmp_path / f"expr_{i}.csv"
        frame.to_csv(path, index=False)
        loaded = load_expression_matrix(str(path))
        assert loaded.index.is_unique
        assert loaded.loc["c1", "TP53 (7157)"] == 1.0


def test_expression_rejects_ambiguous_profiles(tmp_path):
    path = tmp_path / "expr.csv"
    pd.DataFrame({"ModelID": ["c1", "c1"], "TP53 (7157)": [1.0, 2.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="one default profile"):
        load_expression_matrix(str(path))


@pytest.mark.parametrize("pred", [np.ones((2, 1)), np.ones(3), np.array([1.0, np.inf])])
def test_metrics_reject_broadcasting_and_nonfinite_values(pred):
    with pytest.raises(ValueError):
        regression_metrics(np.array([1.0, 2.0]), pred)


def test_split_validation_rejects_silent_row_loss():
    frame = pd.DataFrame({"pair_id": ["a", "b"]})
    with pytest.raises(ValueError, match="complete cohort"):
        validate_assignments(frame, pd.DataFrame({"pair_id": ["a"], "split": ["train"]}))


def test_two_folds_cannot_leave_training_data():
    with pytest.raises(ValueError, match="three folds"):
        assign_group_folds(5, 2, np.random.default_rng(0))
    with pytest.raises(ValueError, match="fold must"):
        fold_assignments(
            pd.DataFrame({"drug_id": ["d"], "pair_id": ["a"]}),
            "cold_drug",
            fold=3,
            n_folds=3,
            seed=0,
        )


def test_reference_is_fixed_even_when_another_baseline_wins_test():
    report = pd.DataFrame(
        {
            "stage": ["B0", "B0"],
            "model": ["mean_effects", "global_mean"],
            "model_id": ["B0_mean_effects", "B0_global_mean"],
            "split_name": ["cold_drug"] * 2,
            "subset": ["test"] * 2,
            "rmse": [2.0, 1.0],
        }
    )
    result = add_baseline_comparison(report)
    assert result["b0_reference_model"].eq("B0_mean_effects").all()


@pytest.mark.parametrize("corruption", ["truth", "subset", "duplicate", "missing"])
def test_reporting_rejects_stale_or_incomplete_predictions(tmp_path, corruption):
    cohort = pd.DataFrame(
        {
            "pair_id": ["a", "b", "c", "d"],
            "depmap_id": ["c1"] * 4,
            "drug_id": ["d1"] * 4,
            "ln_ic50": [1.0, 2.0, 3.0, 4.0],
        }
    )
    pd.DataFrame(
        {"pair_id": cohort.pair_id, "split": ["train", "validation", "test", "test"]}
    ).to_csv(tmp_path / "random_pair.csv", index=False)
    test = cohort.iloc[2:]
    preds = prediction_frame(
        test,
        test.ln_ic50.to_numpy(),
        test.ln_ic50.to_numpy(),
        stage="B1",
        model="ridge",
        split_name="random_pair",
        subset="test",
    )
    if corruption == "truth":
        preds["y_true"] += 1
    elif corruption == "subset":
        preds["subset"] = "validation"
    elif corruption == "duplicate":
        preds = pd.concat([preds, preds])
    else:
        preds = preds.iloc[:1]
    with pytest.raises(ValueError):
        build_metrics_report(preds, cohort, tmp_path)


def test_concordance_uses_matched_cells_per_drug_and_keeps_small_groups():
    rows = []
    for study, values in (("gdsc2", [0.1, 0.2, 0.3]), ("prism", [0.6, 0.4, 0.2])):
        rows.extend(
            {"study": study, "depmap_id": str(i), "drug_id": "d", "auc": value}
            for i, value in enumerate(values)
        )
        rows.append({"study": study, "depmap_id": "single", "drug_id": "small", "auc": 0.5})
    result = assay_concordance(pd.DataFrame(rows), min_cells=2).set_index("drug_id")
    assert result.loc["d", "spearman"] == pytest.approx(-1.0)
    assert result.loc["d", "n_shared_cells"] == 3
    assert not result.loc["small", "eligible"]
    assert np.isnan(result.loc["small", "spearman"])


def test_sklearn_mlp_uses_external_validation_without_training_on_it(monkeypatch):
    from mcdrp.models import baseline_b3

    calls = []

    class FakeMLP:
        def __init__(self, **kwargs):
            assert not kwargs["early_stopping"]
            self.n_iter_ = 0
            self.loss_ = 0.0

        def partial_fit(self, x, y):
            calls.append(x.copy())
            self.n_iter_ += 1

        def predict(self, x):
            # External validation prefers the very first epoch.
            return np.full(len(x), float(self.n_iter_))

    monkeypatch.setattr(baseline_b3, "MLPRegressor", FakeMLP)
    x_train = np.ones((5, 2))
    fit = baseline_b3.fit_sklearn_mlp(
        x_train,
        np.zeros(5),
        hidden_layer_sizes=(4,),
        alpha=0.001,
        learning_rate_init=0.001,
        batch_size=2,
        max_iter=20,
        early_stopping=True,
        validation_fraction=0.9,
        random_state=0,
        validation_data=(np.full((2, 2), 999.0), np.ones(2)),
    )
    assert fit.n_iter == 1
    assert len(calls) == 13
    assert all(np.array_equal(call, x_train) for call in calls)
