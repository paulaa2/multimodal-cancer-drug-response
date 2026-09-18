import numpy as np
import pandas as pd
import pytest

from mcdrp.results.predictions import (
    PREDICTION_COLUMNS,
    load_predictions,
    prediction_frame,
    write_predictions,
)


def eval_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair_id": ["p1", "p2", "p3"],
            "depmap_id": ["c1", "c1", "c2"],
            "drug_id": ["d1", "d2", "d1"],
            "ln_ic50": [1.0, 2.0, 3.0],
        }
    )


def test_prediction_frame_uses_the_shared_schema() -> None:
    rows = eval_rows()

    frame = prediction_frame(
        rows,
        rows["ln_ic50"].to_numpy(),
        np.array([1.1, 2.1, 2.9]),
        stage="B1",
        model="ridge",
        split_name="random_pair",
        subset="test",
    )

    assert list(frame.columns) == list(PREDICTION_COLUMNS)
    assert frame["pair_id"].tolist() == ["p1", "p2", "p3"]
    assert frame["stage"].unique().tolist() == ["B1"]


def test_prediction_frame_rejects_misaligned_predictions() -> None:
    rows = eval_rows()

    # A shuffled DataLoader would produce exactly this mismatch.
    with pytest.raises(ValueError, match="different order"):
        prediction_frame(
            rows,
            rows["ln_ic50"].to_numpy(),
            np.array([1.0, 2.0]),
            stage="B4",
            model="gnn",
            split_name="random_pair",
            subset="test",
        )


def test_prediction_frame_requires_pair_id() -> None:
    rows = eval_rows().drop(columns=["pair_id"])

    with pytest.raises(ValueError, match="pair_id"):
        prediction_frame(
            rows,
            np.zeros(3),
            np.zeros(3),
            stage="B0",
            model="global_mean",
            split_name="random_pair",
            subset="test",
        )


def test_write_and_load_predictions_round_trip(tmp_path) -> None:
    rows = eval_rows()
    frame = prediction_frame(
        rows,
        rows["ln_ic50"].to_numpy(),
        np.array([1.1, 2.1, 2.9]),
        stage="B1",
        model="ridge",
        split_name="random_pair",
        subset="test",
    )
    path = tmp_path / "b1.csv"

    write_predictions([frame], path)
    loaded = load_predictions(path)

    assert len(loaded) == 3
    assert loaded["model"].unique().tolist() == ["ridge"]


def test_load_predictions_skips_missing_files(tmp_path) -> None:
    rows = eval_rows()
    frame = prediction_frame(
        rows,
        rows["ln_ic50"].to_numpy(),
        np.zeros(3),
        stage="B0",
        model="global_mean",
        split_name="random_pair",
        subset="test",
    )
    present = tmp_path / "b0.csv"
    write_predictions([frame], present)

    loaded = load_predictions([present, tmp_path / "absent.csv"])

    assert len(loaded) == 3
