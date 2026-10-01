"""Alias records must stay together through outer folds and calibration."""

import numpy as np
import pandas as pd
import pytest

from mcdrp.features.chemical_identity import annotate_chemical_identity
from mcdrp.splits.calibration import reserve_calibration
from mcdrp.splits.grouped_cv import fold_assignments


def test_chemical_aliases_do_not_cross_training_validation_test_or_calibration():
    pytest.importorskip("rdkit")
    # Different SMILES encodings/IDs of ethanol represent the same full identity.
    smiles = ["CCO", "OCC", "CO", "CCC", "CCCC", "CCN", "CCCl", "CCBr", "CCF", "CC=O"]
    cohort = pd.DataFrame(
        {
            "pair_id": [f"p{k}" for k in range(20)],
            "drug_id": np.repeat(range(10), 2),
            "canonical_smiles": np.repeat(smiles, 2),
            "depmap_id": [f"c{k % 2}" for k in range(20)],
        }
    )
    frame, report = annotate_chemical_identity(cohort)
    assert report["chemical_identities"] == 9
    assert report["aliases"][0]["drug_ids"] == ["0", "1"]
    tested = []
    for fold in range(3):
        assignments = fold_assignments(
            frame, "cold_drug", fold=fold, n_folds=3, seed=42,
            group_column="chemical_id",
        )
        rows = frame.merge(assignments, on="pair_id")
        assert rows.groupby("chemical_id")["split"].nunique().eq(1).all()
        tested.extend(rows.loc[rows["split"].eq("test"), "pair_id"])
        reserved = reserve_calibration(frame, assignments, group="chemical_id", seed=42)
        calibrated = frame.merge(reserved, on="pair_id")
        assert calibrated.groupby("chemical_id")["split"].nunique().eq(1).all()
        assert set(calibrated["split"]) == {"train", "validation", "calibration", "test"}
    assert len(tested) == len(set(tested)) == len(cohort)
