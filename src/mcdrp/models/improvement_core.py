"""Validation-only convex ensembles and separately labelled target-domain adaptation."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from mcdrp.results.uncertainty import affine_parameters


def simplex_grid(n_members: int, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("Grid step must be positive.")
    units = int(round(1 / step))
    if n_members < 1 or step <= 0 or not np.isclose(units * step, 1):
        raise ValueError("Grid step must divide one, and members must be positive.")
    return (
        np.array(
            [x for x in product(range(units + 1), repeat=n_members) if sum(x) == units],
            dtype=float,
        )
        / units
    )


def select_ensemble(y: np.ndarray, predictions: np.ndarray, step: float) -> dict[str, Any]:
    """Accept validation arrays only. Keep the complete candidate table."""
    if predictions.ndim != 2 or len(y) != len(predictions) or not len(y):
        raise ValueError("Aligned nonempty validation arrays are required.")
    if not np.isfinite(predictions).all() or not np.isfinite(y).all():
        raise ValueError("Validation inputs must be finite.")
    weights = simplex_grid(predictions.shape[1], step)
    scores = np.sqrt(np.mean((predictions @ weights.T - y[:, None]) ** 2, axis=0))
    best = int(np.argmin(scores))
    return {
        "weights": weights[best].tolist(),
        "validation_rmse": float(scores[best]),
        "candidates": [
            {"weights": w.tolist(), "validation_rmse": float(s)}
            for w, s in zip(weights, scores, strict=True)
        ],
    }


def cell_partition(
    rows: pd.DataFrame, *, seed: int, fractions: list[float], roles: list[str]
) -> pd.DataFrame:
    """Allocate complete target-domain cells independently of response values."""
    if (
        len(fractions) != len(roles)
        or len(set(roles)) != len(roles)
        or any(f <= 0 for f in fractions)
        or not np.isclose(sum(fractions), 1)
    ):
        raise ValueError("Distinct roles and positive fractions summing to one are required.")
    if rows["depmap_id"].isna().any() or rows["pair_id"].duplicated().any():
        raise ValueError("Cell IDs must be non-null and pair IDs unique.")
    cells = np.array(sorted(rows["depmap_id"].unique()))
    np.random.default_rng(seed).shuffle(cells)
    cuts = np.floor(np.cumsum(fractions)[:-1] * len(cells)).astype(int)
    parts = np.split(cells, cuts)
    if any(len(x) == 0 for x in parts):
        raise ValueError("Too few cells for nonempty partitions.")
    mapping = {cell: role for role, part in zip(roles, parts, strict=True) for cell in part}
    result = rows[["pair_id", "depmap_id"]].copy()
    result["split"] = result["depmap_id"].map(mapping)
    return result


def fit_adapter(
    y: np.ndarray, prediction: np.ndarray, chemistry: np.ndarray, shrinkage: float | None
) -> dict[str, Any]:
    """Fit affine mapping and optional shrunk chemical offsets on development labels."""
    if (
        len(y) == 0
        or len(y) != len(prediction)
        or len(y) != len(chemistry)
        or not np.isfinite(y).all()
        or not np.isfinite(prediction).all()
        or pd.isna(chemistry).any()
        or (shrinkage is not None and shrinkage < 0)
    ):
        raise ValueError("Finite aligned development data and nonnegative shrinkage required.")
    intercept, slope = affine_parameters(y, prediction)
    offsets = {}
    if shrinkage is not None:
        data = pd.DataFrame(
            {"chemical_id": chemistry, "residual": y - (intercept + slope * prediction)}
        )
        stats = data.groupby("chemical_id")["residual"].agg(["sum", "count"])
        offsets = (stats["sum"] / (stats["count"] + shrinkage)).to_dict()
    return {"intercept": intercept, "slope": slope, "offsets": offsets, "shrinkage": shrinkage}


def apply_adapter(adapter: dict, prediction: np.ndarray, chemistry: np.ndarray) -> np.ndarray:
    offsets = np.array([adapter["offsets"].get(str(c), 0) for c in chemistry])
    return np.clip(adapter["intercept"] + adapter["slope"] * prediction + offsets, 0, 1)
