"""Utilities for external drug embedding tables.

The project already supports handcrafted Morgan fingerprints and molecular
graphs. This module adds a neutral interface for learned drug representations
generated outside the training script, for example ChemBERTa, MolFormer,
Uni-Mol, or any other SMILES/molecule encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_METADATA_COLUMNS = {
    "drug_id",
    "drug_name",
    "name",
    "canonical_smiles",
    "smiles",
    "pubchem_cid",
    "chembl_id",
    "n_response_rows",
    "encoder_name",
    "encoder_version",
    "encoder_revision",
    "pooling",
}


@dataclass(frozen=True)
class DrugEmbeddingTable:
    """Loaded external drug embeddings."""

    embedding_map: dict[str, np.ndarray]
    key_column: str
    embedding_columns: tuple[str, ...]
    source_path: str

    @property
    def embedding_dim(self) -> int:
        """Number of numeric dimensions in each embedding vector."""

        return len(self.embedding_columns)

    @property
    def n_drugs(self) -> int:
        """Number of unique keys with embeddings."""

        return len(self.embedding_map)


def infer_embedding_columns(
    table: pd.DataFrame,
    *,
    key_column: str,
    embedding_prefix: str | None = None,
) -> list[str]:
    """Infer numeric embedding columns from a loaded table."""

    if embedding_prefix:
        columns = [c for c in table.columns if c.startswith(embedding_prefix)]
    else:
        metadata = DEFAULT_METADATA_COLUMNS | {key_column}
        columns = [c for c in table.columns if c not in metadata]

    numeric_columns = []
    for column in columns:
        if pd.api.types.is_numeric_dtype(table[column]):
            numeric_columns.append(column)
            continue
        converted = pd.to_numeric(table[column], errors="coerce")
        if converted.notna().any():
            numeric_columns.append(column)
    if not numeric_columns:
        hint = (
            f" starting with prefix {embedding_prefix!r}"
            if embedding_prefix
            else " after excluding metadata columns"
        )
        raise ValueError(f"No numeric embedding columns found{hint}.")
    return numeric_columns


def load_drug_embeddings(
    path: str | Path,
    *,
    key_column: str = "drug_id",
    embedding_prefix: str | None = None,
) -> DrugEmbeddingTable:
    """Load a CSV containing one external embedding vector per drug.

    The CSV must contain a key column and numeric embedding columns. Duplicate
    keys are averaged, which makes the loader tolerant of chunked embedding
    exports while keeping a single deterministic vector per drug.
    """

    embedding_path = Path(path)
    if not embedding_path.exists():
        raise FileNotFoundError(
            f"Missing drug embedding table: {embedding_path}. "
            "Create it first, for example with ChemBERTa/MolFormer/Uni-Mol "
            "embeddings, or point --drug-embeddings to the correct CSV."
        )

    table = pd.read_csv(embedding_path)
    if key_column not in table.columns:
        raise ValueError(
            f"{embedding_path} is missing key column {key_column!r}. "
            f"Available columns: {list(table.columns)}"
        )

    table = table.dropna(subset=[key_column]).copy()
    table[key_column] = table[key_column].astype(str)
    embedding_columns = infer_embedding_columns(
        table,
        key_column=key_column,
        embedding_prefix=embedding_prefix,
    )

    values = table[[key_column, *embedding_columns]].copy()
    for column in embedding_columns:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    values = values.dropna(subset=embedding_columns)
    if values.empty:
        raise ValueError(f"No complete embedding rows found in {embedding_path}.")

    grouped = values.groupby(key_column, as_index=True)[embedding_columns].mean()
    embedding_map = {
        str(key): row.to_numpy(dtype=np.float32)
        for key, row in grouped.iterrows()
    }
    return DrugEmbeddingTable(
        embedding_map=embedding_map,
        key_column=key_column,
        embedding_columns=tuple(embedding_columns),
        source_path=str(embedding_path),
    )


def build_drug_embedding_matrix(
    rows: pd.DataFrame,
    embedding_map: dict[str, np.ndarray],
    *,
    cohort_key_column: str = "drug_id",
    missing: str = "error",
) -> np.ndarray:
    """Build an embedding matrix aligned to cohort or split rows."""

    if missing not in {"error", "zero"}:
        raise ValueError("missing must be either 'error' or 'zero'.")
    if cohort_key_column not in rows.columns:
        raise ValueError(f"Rows are missing cohort key column {cohort_key_column!r}.")
    if not embedding_map:
        raise ValueError("embedding_map is empty.")

    embedding_dim = next(iter(embedding_map.values())).shape[0]
    matrix = np.zeros((len(rows), embedding_dim), dtype=np.float32)
    missing_keys: list[str] = []

    for idx, key in enumerate(rows[cohort_key_column].astype(str)):
        embedding = embedding_map.get(key)
        if embedding is None:
            missing_keys.append(key)
            continue
        matrix[idx] = embedding

    if missing_keys and missing == "error":
        sample = sorted(set(missing_keys))[:10]
        raise ValueError(
            f"Missing embeddings for {len(set(missing_keys))} unique values from "
            f"{cohort_key_column!r}. Examples: {sample}"
        )
    return matrix
