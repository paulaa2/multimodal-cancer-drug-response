"""Shared checks for split identity before training or scoring."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd


def validate_assignments(cohort: pd.DataFrame, assignments: pd.DataFrame) -> None:
    """Require one known label per cohort pair, without silently dropping rows."""

    for name, frame in (("Cohort", cohort), ("Split", assignments)):
        if "pair_id" not in frame or frame["pair_id"].isna().any():
            raise ValueError(f"{name} must have non-null pair_ids.")
        if frame["pair_id"].duplicated().any():
            raise ValueError(f"{name} has duplicate pair_ids.")
    if "split" not in assignments:
        raise ValueError("Split assignments need a split column.")
    if (
        not assignments["split"]
        .isin(("train", "validation", "calibration", "test", "unused"))
        .all()
    ):
        raise ValueError("Split assignments contain invalid labels.")
    if set(cohort["pair_id"]) != set(assignments["pair_id"]):
        raise ValueError("Split pair_ids must match the complete cohort exactly.")


def selection_fingerprint(
    cohort: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    target: str,
    context: dict[str, Any] | None = None,
) -> str:
    """Identify selection data independently of CSV ordering and test labels.

    Include train/validation targets, feature identities and explicit feature
    configuration; test outcomes must not participate in model selection.
    """

    validate_assignments(cohort, assignments)
    columns = ["pair_id", target]
    columns.extend(
        key
        for key in ("depmap_id", "drug_id", "canonical_smiles")
        if key in cohort and key not in columns
    )
    selected = cohort[columns].merge(assignments, on="pair_id", validate="one_to_one")
    selected = selected.loc[selected["split"].isin(("train", "validation"))]
    selected = selected.sort_values("pair_id").reset_index(drop=True)
    payload = {
        "target": target,
        "context": context or {},
        "rows": json.loads(selected.to_json(orient="records", double_precision=15)),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
