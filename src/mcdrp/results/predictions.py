"""Tidy per-row prediction artifacts.

Models used to write only aggregate metrics, which meant no new metric could
be computed without retraining, error analysis was impossible, and each model
script had to build its own metric rows. Every model now also emits its
predictions in one shared schema, and ``mcdrp.results.metrics_report`` turns
those into the full metric families in a single place.

The schema is deliberately narrow and long rather than wide: one row per
prediction, identified by ``pair_id`` so it can be joined back to the cohort
for any grouping we later want (tissue, scaffold, dose range).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

PREDICTION_COLUMNS = (
    "pair_id",
    "stage",
    "model",
    "split_name",
    "subset",
    "y_true",
    "y_pred",
)


def prediction_frame(
    rows: pd.DataFrame,
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    *,
    stage: str,
    model: str,
    split_name: str,
    subset: str,
    params: str | None = None,
) -> pd.DataFrame:
    """Build one tidy prediction frame.

    ``rows`` must be the evaluation rows in the same order as the predictions,
    and must carry ``pair_id``. Callers that predict through a DataLoader are
    responsible for keeping the loader unshuffled.
    """

    if "pair_id" not in rows.columns:
        raise ValueError("Prediction rows must carry a pair_id column.")

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(rows) != len(y_pred) or len(y_true) != len(y_pred):
        raise ValueError(
            "Row count and prediction count disagree: "
            f"{len(rows)} rows, {len(y_true)} targets, {len(y_pred)} predictions. "
            "This usually means predictions were produced in a different order "
            "than the evaluation rows."
        )

    frame = pd.DataFrame(
        {
            "pair_id": rows["pair_id"].to_numpy(),
            "stage": stage,
            "model": model,
            "split_name": split_name,
            "subset": subset,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    if params is not None:
        frame["params"] = params
    return frame


def write_predictions(
    frames: Iterable[pd.DataFrame],
    path: str | Path,
) -> pd.DataFrame:
    """Concatenate prediction frames, validate the schema, and write them."""

    collected = [frame for frame in frames if not frame.empty]
    if not collected:
        raise ValueError("No predictions to write.")

    predictions = pd.concat(collected, ignore_index=True)
    missing = set(PREDICTION_COLUMNS) - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    return predictions


def load_predictions(paths: str | Path | Iterable[str | Path]) -> pd.DataFrame:
    """Load and concatenate one or more prediction files.

    Missing files are skipped, so reporting can run over whichever stages have
    been executed so far.
    """

    if isinstance(paths, (str, Path)):
        paths = [paths]

    frames = []
    for path in paths:
        prediction_path = Path(path)
        if not prediction_path.exists():
            continue
        frame = pd.read_csv(prediction_path)
        missing = set(PREDICTION_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(
                f"{prediction_path} is missing columns: {sorted(missing)}"
            )
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No prediction files found in {list(paths)}.")
    return pd.concat(frames, ignore_index=True)


def write_prediction_summary(
    predictions: pd.DataFrame,
    path: str | Path,
    **extra: object,
) -> Path:
    """Write a small JSON summary describing a prediction file."""

    summary = {
        "n_rows": int(len(predictions)),
        "stages": sorted(predictions["stage"].unique().tolist()),
        "models": sorted(predictions["model"].unique().tolist()),
        "splits": sorted(predictions["split_name"].unique().tolist()),
        "subsets": sorted(predictions["subset"].unique().tolist()),
        **extra,
    }
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary_path
