"""Regression metrics used across baselines and models.

Global RMSE and Pearson correlation are not sufficient for drug-response
prediction. Most of the variance in ``ln(IC50)`` comes from differences in mean
potency between drugs, so a model that only memorizes each drug's mean response
already looks excellent on global metrics. This is a form of Simpson's paradox
and it hides whether a model learned anything about drug-cell-line interaction,
which is the biologically interesting signal.

To make that visible, this module provides three metric families:

- global metrics (``rmse``, ``mae``, ``pearson``, ``spearman``, ``r2``)
- mean-effect-normalized metrics, computed after subtracting a reference
  mean-effects prediction from both the true and predicted responses
- metrics stratified by drug and by cell line, averaged over groups

See ``docs/evaluation_protocol.md`` for the rationale and references.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeanEffectsReference:
    """Additive drug and cell-line mean-effects reference model.

    Predicts ``y_hat = mu_cell + mu_drug - mu`` following the naive mean-effects
    predictor used as the standard reference baseline in drug-response
    benchmarks. It uses no transcriptomic and no chemical features, so any model
    that does not beat it has not learned anything from its input modalities.

    Unknown keys fall back to the global training mean, which makes the same
    reference usable in cold-cell, cold-drug, and cold-both settings.
    """

    global_mean: float
    cell_effects: Mapping[str, float]
    drug_effects: Mapping[str, float]

    @classmethod
    def fit(
        cls,
        cell_keys: Sequence[object],
        drug_keys: Sequence[object],
        y: np.ndarray,
    ) -> MeanEffectsReference:
        """Fit the reference on training rows only."""

        y = np.asarray(y, dtype=float)
        if len(y) == 0:
            raise ValueError("Cannot fit a mean-effects reference on zero rows.")

        global_mean = float(np.mean(y))
        return cls(
            global_mean=global_mean,
            cell_effects=_group_means(cell_keys, y),
            drug_effects=_group_means(drug_keys, y),
        )

    def predict(
        self,
        cell_keys: Sequence[object],
        drug_keys: Sequence[object],
    ) -> np.ndarray:
        """Predict responses for evaluation rows."""

        cell = np.array(
            [self.cell_effects.get(str(key), self.global_mean) for key in cell_keys],
            dtype=float,
        )
        drug = np.array(
            [self.drug_effects.get(str(key), self.global_mean) for key in drug_keys],
            dtype=float,
        )
        return cell + drug - self.global_mean


def _group_means(keys: Sequence[object], y: np.ndarray) -> dict[str, float]:
    """Compute the mean of ``y`` for each distinct key."""

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for key, value in zip(keys, y, strict=True):
        name = str(key)
        totals[name] = totals.get(name, 0.0) + float(value)
        counts[name] = counts.get(name, 0) + 1
    return {name: totals[name] / counts[name] for name in totals}


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    reference_pred: np.ndarray | None = None,
    cell_keys: Sequence[object] | None = None,
    drug_keys: Sequence[object] | None = None,
) -> dict[str, float]:
    """Compute regression metrics for drug-response prediction.

    ``reference_pred`` should hold mean-effects predictions for the same rows.
    When given, normalized metrics are added that measure only the differential
    response signal beyond drug and cell-line mean effects. When ``drug_keys``
    or ``cell_keys`` are given, correlations stratified by that entity are added.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_true - y_pred

    mse = float(np.mean(error**2))
    metrics = {
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(error))),
        "pearson": safe_corr(y_true, y_pred, method="pearson"),
        "spearman": safe_corr(y_true, y_pred, method="spearman"),
        "r2": r2_score(y_true, y_pred),
    }

    if reference_pred is not None:
        reference_pred = np.asarray(reference_pred, dtype=float)
        residual_true = y_true - reference_pred
        residual_pred = y_pred - reference_pred
        metrics.update(
            {
                "r2_normalized": r2_score(residual_true, residual_pred),
                "pearson_normalized": safe_corr(
                    residual_true, residual_pred, method="pearson"
                ),
                "spearman_normalized": safe_corr(
                    residual_true, residual_pred, method="spearman"
                ),
            }
        )

    for label, keys in (("drug", drug_keys), ("cell", cell_keys)):
        if keys is None:
            continue
        grouped = grouped_metrics(y_true, y_pred, keys)
        metrics.update({f"{name}_per_{label}": value for name, value in grouped.items()})

    return metrics


def grouped_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    keys: Sequence[object],
) -> dict[str, float]:
    """Average Pearson, Spearman, and R2 within groups defined by ``keys``.

    Groups with fewer than two rows carry no correlation information and are
    skipped. ``n_groups`` reports how many groups actually contributed.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    indices: dict[str, list[int]] = {}
    for position, key in enumerate(keys):
        indices.setdefault(str(key), []).append(position)

    pearson: list[float] = []
    spearman: list[float] = []
    r2: list[float] = []
    for positions in indices.values():
        if len(positions) < 2:
            continue
        selection = np.asarray(positions, dtype=int)
        group_true = y_true[selection]
        group_pred = y_pred[selection]
        for values, target in (
            (pearson, safe_corr(group_true, group_pred, method="pearson")),
            (spearman, safe_corr(group_true, group_pred, method="spearman")),
            (r2, r2_score(group_true, group_pred)),
        ):
            if not math.isnan(target):
                values.append(target)

    return {
        "pearson": _mean_or_nan(pearson),
        "spearman": _mean_or_nan(spearman),
        "r2": _mean_or_nan(r2),
        "n_groups": float(len(pearson)),
    }


def _mean_or_nan(values: Sequence[float]) -> float:
    """Average a list of metric values, returning NaN when it is empty."""

    return float(np.mean(values)) if values else float("nan")


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination, returning NaN for constant targets."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


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
