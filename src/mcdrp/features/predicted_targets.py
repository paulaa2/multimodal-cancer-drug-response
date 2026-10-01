"""Predict GDSC target profiles from Morgan fingerprints, train-drugs only.

This is the leakage-safe stand-in for a BindingDB DTI: a linear map from
structure to the same multi-hot vocabulary used by the annotated profiles.
This tests one Ridge map on this small annotation vocabulary. Failure does not
establish that molecular structure contains no target information, or reproduce
the pretraining and datasets of published target-prediction architectures.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


def predict_targets_from_fingerprints(
    cohort: pd.DataFrame,
    train_drug_ids: set[str],
    fp_matrix: np.ndarray,
    target_matrix: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Fit a multi-output Ridge on training drugs and predict every row.

    Both matrices are aligned to ``cohort`` rows. The estimator sees one row
    per training drug, so a drug that appears in many pairs cannot dominate
    the fit. Predictions are clipped to [0, 1].
    """

    if len(fp_matrix) != len(cohort) or len(target_matrix) != len(cohort):
        raise ValueError("Fingerprint and target matrices must align with the cohort.")
    if fp_matrix.shape[1] == 0:
        raise ValueError("predict_targets_from_fingerprints needs Morgan bits.")
    if target_matrix.shape[1] == 0:
        raise ValueError("predict_targets_from_fingerprints needs a target vocabulary.")

    drug_ids = cohort["drug_id"].astype(str).to_numpy()
    train_mask = np.array([drug_id in train_drug_ids for drug_id in drug_ids])
    if not train_mask.any():
        raise ValueError("No training drugs available to fit the fingerprint DTI.")

    train_drugs, train_index = np.unique(drug_ids[train_mask], return_index=True)
    # np.unique sorts; train_index is into the *masked* array, so remap.
    train_positions = np.flatnonzero(train_mask)
    x_train = fp_matrix[train_positions[train_index]].astype(np.float32)
    y_train = target_matrix[train_positions[train_index]].astype(np.float32)

    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)

    all_drugs, all_index = np.unique(drug_ids, return_index=True)
    predicted_unique = np.clip(
        model.predict(fp_matrix[all_index].astype(np.float32)),
        0.0,
        1.0,
    ).astype(np.float32)
    lookup = {str(drug_id): predicted_unique[i] for i, drug_id in enumerate(all_drugs)}

    predicted = np.zeros_like(target_matrix, dtype=np.float32)
    for i, drug_id in enumerate(drug_ids):
        predicted[i] = lookup[drug_id]

    logger.info(
        "Fingerprint DTI: %d training drugs, %d targets, %d drugs predicted",
        len(train_drugs),
        target_matrix.shape[1],
        len(all_drugs),
    )
    return predicted
