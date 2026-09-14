"""Regression metrics used across baselines and models."""

from __future__ import annotations

import math

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard regression metrics for drug-response prediction."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_true - y_pred

    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    return {
        "rmse": math.sqrt(mse),
        "mae": mae,
        "pearson": safe_corr(y_true, y_pred, method="pearson"),
        "spearman": safe_corr(y_true, y_pred, method="spearman"),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray, *, method: str) -> float:
    """Compute a correlation, returning NaN for constant inputs."""

    if len(y_true) < 2:
        return float("nan")

    if method == "spearman":
        y_true = rankdata(y_true)
        y_pred = rankdata(y_pred)
    elif method != "pearson":
        raise ValueError(f"Unknown correlation method: {method}")

    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan")

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    """Rank values with average ranks for ties."""

    values = np.asarray(values)
    sorter = np.argsort(values, kind="mergesort")
    sorted_values = values[sorter]
    ranks = np.empty(len(values), dtype=float)

    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        ranks[sorter[start:end]] = average_rank
        start = end

    return ranks
