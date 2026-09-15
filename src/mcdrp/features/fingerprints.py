"""Morgan fingerprint features from SMILES strings.

Converts canonical SMILES into fixed-length Morgan fingerprint bit vectors
using RDKit.  Invalid SMILES produce a zero vector and a logged warning.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def smiles_to_morgan(
    smiles: str,
    *,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Convert a single SMILES string to a Morgan fingerprint bit vector.

    Parameters
    ----------
    smiles:
        Canonical SMILES string.
    radius:
        Morgan fingerprint radius (default 2 ≈ ECFP4).
    n_bits:
        Length of the folded bit vector.

    Returns
    -------
    np.ndarray of shape ``(n_bits,)`` with dtype ``np.uint8``.
    A zero vector is returned when the SMILES cannot be parsed.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning("Could not parse SMILES: %s — returning zero vector.", smiles)
        return np.zeros(n_bits, dtype=np.uint8)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    fp_on_bits = fp.GetOnBits()
    arr[list(fp_on_bits)] = 1
    return arr


def build_fingerprint_matrix(
    smiles_series: pd.Series,
    *,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Build a fingerprint matrix from a pandas Series of SMILES strings.

    Parameters
    ----------
    smiles_series:
        Series where each element is a canonical SMILES string.
    radius:
        Morgan fingerprint radius.
    n_bits:
        Length of each fingerprint vector.

    Returns
    -------
    np.ndarray of shape ``(len(smiles_series), n_bits)`` with dtype
    ``np.uint8``.
    """
    # Suppress RDKit parser warnings for invalid SMILES; we log our own.
    from rdkit import RDLogger

    rd_logger = RDLogger.logger()
    rd_logger.setLevel(RDLogger.ERROR)

    n_rows = len(smiles_series)
    matrix = np.zeros((n_rows, n_bits), dtype=np.uint8)
    n_failed = 0

    for idx, smiles in enumerate(smiles_series):
        if not isinstance(smiles, str) or not smiles.strip():
            n_failed += 1
            continue
        vec = smiles_to_morgan(smiles, radius=radius, n_bits=n_bits)
        matrix[idx] = vec
        if vec.sum() == 0:
            n_failed += 1

    if n_failed > 0:
        warnings.warn(
            f"{n_failed}/{n_rows} SMILES failed to produce a fingerprint.",
            stacklevel=2,
        )

    return matrix
