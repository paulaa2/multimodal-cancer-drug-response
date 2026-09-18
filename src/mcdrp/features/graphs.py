"""Molecular graph features built from SMILES strings.

This module intentionally keeps the graph representation lightweight and based
only on RDKit + NumPy so the first GNN baseline does not require PyTorch
Geometric.  The resulting arrays are consumed by the pure-PyTorch B4 model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


ATOM_FEATURE_DIM = 32
ATOM_TYPES = (6, 7, 8, 9, 15, 16, 17, 35, 53)


@dataclass(frozen=True)
class MolecularGraph:
    """Array representation of one molecule."""

    node_features: np.ndarray
    adjacency: np.ndarray
    n_nodes: int
    valid: bool


def atom_features(atom: Any) -> np.ndarray:
    """Create a compact numeric atom feature vector."""

    features = np.zeros(ATOM_FEATURE_DIM, dtype=np.float32)
    atomic_num = atom.GetAtomicNum()
    if atomic_num in ATOM_TYPES:
        features[ATOM_TYPES.index(atomic_num)] = 1.0
    else:
        features[len(ATOM_TYPES)] = 1.0

    offset = len(ATOM_TYPES) + 1
    features[offset + min(atom.GetDegree(), 5)] = 1.0
    offset += 6
    features[offset + min(atom.GetTotalNumHs(), 4)] = 1.0
    offset += 5
    features[offset + min(atom.GetFormalCharge() + 2, 4)] = 1.0
    offset += 5
    features[offset] = float(atom.GetIsAromatic())
    features[offset + 1] = float(atom.IsInRing())
    features[offset + 2] = float(atom.GetMass() / 200.0)
    return features


def smiles_to_graph(smiles: str) -> MolecularGraph:
    """Convert one SMILES string into node features and an adjacency matrix."""

    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None or mol.GetNumAtoms() == 0:
        return MolecularGraph(
            node_features=np.zeros((1, ATOM_FEATURE_DIM), dtype=np.float32),
            adjacency=np.ones((1, 1), dtype=np.float32),
            n_nodes=1,
            valid=False,
        )

    n_atoms = mol.GetNumAtoms()
    node_features = np.stack([atom_features(atom) for atom in mol.GetAtoms()])
    adjacency = np.eye(n_atoms, dtype=np.float32)
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        adjacency[begin, end] = 1.0
        adjacency[end, begin] = 1.0

    degree = adjacency.sum(axis=1, keepdims=True)
    adjacency = adjacency / np.maximum(degree, 1.0)
    return MolecularGraph(
        node_features=node_features.astype(np.float32),
        adjacency=adjacency.astype(np.float32),
        n_nodes=n_atoms,
        valid=True,
    )


def build_graph_map(
    cohort: pd.DataFrame,
    *,
    drug_column: str = "drug_id",
    smiles_column: str = "canonical_smiles",
) -> dict[str, MolecularGraph]:
    """Build one graph per unique drug in the cohort."""

    graph_map: dict[str, MolecularGraph] = {}
    failed = 0
    unique_drugs = cohort[[drug_column, smiles_column]].drop_duplicates(drug_column)
    for row in unique_drugs.itertuples(index=False):
        drug_id = getattr(row, drug_column)
        smiles = getattr(row, smiles_column)
        graph = smiles_to_graph(smiles)
        if not graph.valid:
            failed += 1
        graph_map[str(drug_id)] = graph

    if failed:
        logger.warning("%d/%d drug SMILES failed graph parsing.", failed, len(graph_map))
    return graph_map
