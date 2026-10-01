"""Carve a response-independent calibration set out of outer training entities."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mcdrp.splits.validation import validate_assignments


def reserve_calibration(
    cohort: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    group: str,
    fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    validate_assignments(cohort, assignments)
    if not 0 < fraction < 1 or group not in cohort:
        raise ValueError("Choose an existing grouping column and a fraction in (0,1).")
    if assignments["split"].eq("calibration").any():
        raise ValueError("Calibration is already reserved.")
    aligned = assignments.merge(cohort[["pair_id", group]], on="pair_id", validate="one_to_one")
    if aligned[group].isna().any():
        raise ValueError("Calibration entities cannot be missing.")
    entities = sorted(aligned.loc[aligned["split"].eq("train"), group].astype(str).unique())
    if len(entities) < 3:
        raise ValueError("At least three training entities are required.")
    count = max(1, min(len(entities) - 1, int(np.ceil(fraction * len(entities)))))
    selected = set(np.random.default_rng(seed).choice(entities, count, replace=False))
    result = assignments.copy()
    mask = aligned["split"].eq("train") & aligned[group].astype(str).isin(selected)
    result.loc[mask.to_numpy(), "split"] = "calibration"
    # Exclude selected entities from validation too, so affine calibration/early
    # stopping cannot see labels from the calibration unit. Outer test is preserved.
    validation_selected = aligned["split"].eq("validation") & aligned[group].astype(str).isin(
        selected
    )
    result.loc[validation_selected.to_numpy(), "split"] = "unused"
    if not result["split"].eq("validation").any():
        raise ValueError("The calibration reservation leaves no validation rows.")
    validate_assignments(cohort, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group", default="drug_id")
    parser.add_argument("--fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = reserve_calibration(
        pd.read_csv(args.cohort),
        pd.read_csv(args.assignments),
        group=args.group,
        fraction=args.fraction,
        seed=args.seed,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
