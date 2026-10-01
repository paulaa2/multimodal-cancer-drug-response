"""Scientific boundary checks for the bounded, exploratory improvement phase."""

import numpy as np
import pandas as pd
import pytest

from mcdrp.models.improvement_core import (
    apply_adapter,
    cell_partition,
    fit_adapter,
    select_ensemble,
    simplex_grid,
)
from mcdrp.splits.validation import selection_fingerprint


def test_convex_selection_uses_validation_and_can_keep_single_reference():
    y = np.array([1.0, 2.0, 3.0])
    prediction = np.column_stack([y, y + 3, y - 1])
    selected = select_ensemble(y, prediction, 0.25)
    assert selected["validation_rmse"] == 0
    assert sum(selected["weights"]) == 1
    assert len(selected["candidates"]) == 15
    with pytest.raises(ValueError):
        simplex_grid(3, 0)
    with pytest.raises(ValueError):
        select_ensemble(y, np.full((3, 2), np.nan), 0.25)


def test_target_partition_is_cell_disjoint_and_independent_of_labels_and_row_order():
    data = pd.DataFrame(
        {
            "pair_id": [f"p{i}" for i in range(100)],
            "depmap_id": [f"c{i // 2}" for i in range(100)],
            "auc": np.arange(100),
        }
    )
    arguments = {
        "seed": 123,
        "fractions": [0.2, 0.1, 0.2, 0.5],
        "roles": ["train", "validation", "calibration", "test"],
    }
    a = cell_partition(data, **arguments).set_index("pair_id").sort_index()
    data["auc"] = -data["auc"] * 100
    b = (
        cell_partition(data.sample(frac=1, random_state=9), **arguments)
        .set_index("pair_id")
        .sort_index()
    )
    pd.testing.assert_frame_equal(a, b)
    assert a.groupby("depmap_id")["split"].nunique().max() == 1
    assert a.groupby("split")["depmap_id"].nunique().to_dict() == {
        "train": 10,
        "validation": 5,
        "calibration": 10,
        "test": 25,
    }


def test_target_adapter_shrinks_offsets_and_handles_unseen_chemicals():
    pred = np.array([0.4, 0.5, 0.4, 0.5])
    y = pred + np.array([0.2, 0.2, -0.2, -0.2])
    chemistry = np.array(["A", "A", "B", "B"])
    a = fit_adapter(y, pred, chemistry, 0)
    b = fit_adapter(y, pred, chemistry, 100)
    np.testing.assert_allclose(apply_adapter(a, pred, chemistry), y)
    assert abs(b["offsets"]["A"]) < abs(a["offsets"]["A"])
    assert apply_adapter(a, np.array([0.5]), np.array(["new"]))[0] == pytest.approx(0.5)
    assert np.all(apply_adapter(a, np.array([-100, 100]), np.array(["new", "new"])) >= 0)


def test_selection_identity_excludes_calibration_and_test_labels():
    data = pd.DataFrame({"pair_id": ["a", "b", "c", "d"], "auc": [0.1, 0.2, 0.3, 0.4]})
    labels = pd.DataFrame(
        {"pair_id": data.pair_id, "split": ["train", "validation", "calibration", "test"]}
    )
    fingerprint = selection_fingerprint(data, labels, target="auc")
    data.loc[2:, "auc"] = 100
    assert selection_fingerprint(data, labels, target="auc") == fingerprint
    data.loc[1, "auc"] = 100
    assert selection_fingerprint(data, labels, target="auc") != fingerprint


def test_neural_scaling_fits_only_training_entities():
    pytest.importorskip("torch")
    from mcdrp.models.residual_neural import feature_tables

    rows = pd.DataFrame({"drug_id": [1, 2], "depmap_id": ["A", "B"]})
    features = {
        "embeddings": {"1": np.array([2.0]), "2": np.array([100.0])},
        "targets": {"1": np.array([1]), "2": np.array([0])},
        "cells": {"A": np.array([3.0]), "B": np.array([200.0])},
    }
    table = feature_tables(rows, features, ["targets", "embeddings"], rows.iloc[:1])
    assert table["scalers"]["embeddings"].mean_[0] == 2
    assert table["scalers"]["cells"].mean_[0] == 3
    assert table["drug"][1, 1] == 98
    assert table["cell"][1, 0] == 197


def test_neural_fit_repeats_and_restores_without_evaluation_labels():
    pytest.importorskip("torch")
    from mcdrp.models.residual_neural import feature_tables, fit, predict, restore

    rows = pd.DataFrame(
        {"drug_id": [1, 2] * 12, "depmap_id": ["A", "B", "C"] * 8, "y": np.linspace(0, 1, 24)}
    )
    train, val = rows.iloc[:18], rows.iloc[18:]
    features = {
        "targets": {"1": np.array([1.0, 0]), "2": np.array([0, 1.0])},
        "cells": {c: np.array([i * 1.0, i + 1.0]) for i, c in enumerate("ABC")},
    }
    table = feature_tables(rows, features, ["targets"], train)
    candidate = {"blocks": ["targets"], "width": 4}
    config = {
        "epochs": 3,
        "patience": 2,
        "batch_size": 8,
        "learning_rate": 0.01,
        "weight_decay": 0.01,
        "dropout": 0.2,
        "device": "cpu",
    }
    a, _ = fit(train, val, table, target="y", candidate=candidate, config=config, seed=42)
    b, _ = fit(train, val, table, target="y", candidate=candidate, config=config, seed=42)
    for key in a["state"]:
        np.testing.assert_array_equal(a["state"][key], b["state"][key])
    predicted = predict(
        restore(a, "cpu"), val, table, device="cpu", mean=a["mean"], scale=a["scale"]
    )
    assert np.isfinite(predicted).all()
    assert len(predicted) == len(val)
