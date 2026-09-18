import numpy as np
import pandas as pd
import pytest

from mcdrp.metrics import MeanEffectsReference
from mcdrp.results.metrics_report import (
    add_baseline_comparison,
    best_by_split,
    build_metrics_report,
)
from mcdrp.results.predictions import prediction_frame


def cohort() -> pd.DataFrame:
    """Six cell lines by four drugs, with an additive response plus noise."""

    rows = []
    rng = np.random.default_rng(0)
    for cell in range(6):
        for drug in range(4):
            rows.append(
                {
                    "pair_id": f"c{cell}_d{drug}",
                    "depmap_id": f"c{cell}",
                    "drug_id": f"d{drug}",
                    "ln_ic50": float(cell) + 3.0 * drug + rng.normal(0, 0.3),
                }
            )
    return pd.DataFrame(rows)


def write_split(tmp_path, frame: pd.DataFrame) -> None:
    """Assign the first four drugs' rows to train and the rest to test."""

    labels = ["train" if int(pid.split("_d")[1]) < 2 else "test" for pid in frame["pair_id"]]
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(exist_ok=True)
    pd.DataFrame({"pair_id": frame["pair_id"], "split": labels}).to_csv(
        splits_dir / "random_pair.csv", index=False
    )


def test_reference_model_scores_exactly_zero_normalized_r2(tmp_path) -> None:
    """A model reproducing the reference removes none of its error."""

    frame = cohort()
    write_split(tmp_path, frame)
    train = frame[frame["pair_id"].str.contains("_d0|_d1")]
    test = frame[~frame["pair_id"].str.contains("_d0|_d1")]

    reference = MeanEffectsReference.fit(
        train["depmap_id"].tolist(), train["drug_id"].tolist(), train["ln_ic50"].to_numpy()
    )
    predictions = prediction_frame(
        test,
        test["ln_ic50"].to_numpy(),
        reference.predict(test["depmap_id"].tolist(), test["drug_id"].tolist()),
        stage="B0",
        model="mean_effects",
        split_name="random_pair",
        subset="test",
    )

    report = build_metrics_report(predictions, frame, tmp_path / "splits")

    assert len(report) == 1
    assert report.loc[0, "r2_normalized"] == pytest.approx(0.0, abs=1e-9)
    # Its residual prediction is constant zero, so there is no signal to correlate.
    assert np.isnan(report.loc[0, "pearson_normalized"])


def test_perfect_model_scores_one_normalized_r2(tmp_path) -> None:
    """A model with zero error removes all of the reference's error."""

    frame = cohort()
    write_split(tmp_path, frame)
    test = frame[~frame["pair_id"].str.contains("_d0|_d1")]

    predictions = prediction_frame(
        test,
        test["ln_ic50"].to_numpy(),
        test["ln_ic50"].to_numpy(),
        stage="B5",
        model="hybrid_gnn",
        split_name="random_pair",
        subset="test",
    )

    report = build_metrics_report(predictions, frame, tmp_path / "splits")

    assert report.loc[0, "r2_normalized"] == pytest.approx(1.0)
    assert report.loc[0, "pearson_normalized"] == pytest.approx(1.0)


def test_report_covers_every_model_split_subset_combination(tmp_path) -> None:
    frame = cohort()
    write_split(tmp_path, frame)
    test = frame[~frame["pair_id"].str.contains("_d0|_d1")]

    frames = [
        prediction_frame(
            test,
            test["ln_ic50"].to_numpy(),
            test["ln_ic50"].to_numpy() + offset,
            stage=stage,
            model=model,
            split_name="random_pair",
            subset="test",
        )
        for stage, model, offset in (("B0", "global_mean", 2.0), ("B1", "ridge", 0.1))
    ]
    predictions = pd.concat(frames, ignore_index=True)

    report = add_baseline_comparison(
        build_metrics_report(predictions, frame, tmp_path / "splits")
    )

    assert set(report["model_id"]) == {"B0_global_mean", "B1_ridge"}
    # B1 is much closer to the truth, so it must improve on the B0 reference.
    ridge = report.loc[report["model_id"].eq("B1_ridge")].iloc[0]
    assert ridge["rmse_improvement_pct_vs_b0"] > 0
    assert best_by_split(report).iloc[0]["model_id"] == "B1_ridge"


def test_report_rejects_predictions_with_unknown_pairs(tmp_path) -> None:
    frame = cohort()
    write_split(tmp_path, frame)
    test = frame[~frame["pair_id"].str.contains("_d0|_d1")].copy()
    test["pair_id"] = "unknown_" + test["pair_id"]

    predictions = prediction_frame(
        test,
        test["ln_ic50"].to_numpy(),
        test["ln_ic50"].to_numpy(),
        stage="B1",
        model="ridge",
        split_name="random_pair",
        subset="test",
    )

    with pytest.raises(ValueError, match="absent from the cohort"):
        build_metrics_report(predictions, frame, tmp_path / "splits")
