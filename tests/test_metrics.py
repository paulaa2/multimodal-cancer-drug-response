import math

import numpy as np
import pytest

from mcdrp.metrics import MeanEffectsReference, grouped_metrics, regression_metrics


def test_regression_metrics_for_perfect_prediction() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    metrics = regression_metrics(y_true, y_pred)

    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["pearson"] == 1.0
    assert metrics["spearman"] == 1.0
    assert metrics["r2"] == 1.0


def test_mean_effects_reference_recovers_additive_structure() -> None:
    cells = ["c1", "c1", "c2", "c2"]
    drugs = ["d1", "d2", "d1", "d2"]
    # Perfectly additive: y = cell effect + drug effect.
    y = np.array([0.0, 2.0, 1.0, 3.0])

    reference = MeanEffectsReference.fit(cells, drugs, y)

    assert reference.predict(cells, drugs) == pytest.approx(y)


def test_mean_effects_reference_falls_back_to_global_mean_for_unseen_keys() -> None:
    reference = MeanEffectsReference.fit(
        ["c1", "c2"],
        ["d1", "d2"],
        np.array([1.0, 3.0]),
    )

    # An unseen cell and an unseen drug leave only the global mean.
    assert reference.predict(["unseen"], ["unseen"]) == pytest.approx([2.0])
    # A known drug still contributes its own effect.
    assert reference.predict(["unseen"], ["d1"]) == pytest.approx([1.0])


def test_normalized_metrics_reveal_mean_effect_memorization() -> None:
    cells = ["c1", "c1", "c2", "c2"]
    drugs = ["d1", "d2", "d1", "d2"]
    y_true = np.array([0.0, 10.0, 1.0, 11.0])
    reference = MeanEffectsReference.fit(cells, drugs, y_true)
    # A model that only reproduces the mean effects looks strong globally.
    y_pred = reference.predict(cells, drugs)

    metrics = regression_metrics(
        y_true,
        y_pred,
        reference_pred=y_pred,
        cell_keys=cells,
        drug_keys=drugs,
    )

    assert metrics["pearson"] > 0.99
    # But it explains none of the remaining differential signal.
    assert math.isnan(metrics["r2_normalized"])
    assert math.isnan(metrics["pearson_normalized"])


def test_grouped_metrics_skip_groups_with_a_single_row() -> None:
    y_true = np.array([1.0, 2.0, 5.0])
    y_pred = np.array([1.0, 2.0, 9.0])

    grouped = grouped_metrics(y_true, y_pred, ["d1", "d1", "d2"])

    assert grouped["n_groups"] == 1.0
    assert grouped["pearson"] == pytest.approx(1.0)
