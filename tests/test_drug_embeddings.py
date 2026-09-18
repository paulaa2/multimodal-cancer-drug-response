from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mcdrp.features.drug_embeddings import (
    build_drug_embedding_matrix,
    load_drug_embeddings,
)


def test_load_drug_embeddings_averages_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.csv"
    pd.DataFrame(
        {
            "drug_id": ["D1", "D1", "D2"],
            "drug_name": ["a", "a", "b"],
            "emb_0": [1.0, 3.0, 10.0],
            "emb_1": [2.0, 4.0, 20.0],
        }
    ).to_csv(path, index=False)

    table = load_drug_embeddings(path, embedding_prefix="emb_")

    assert table.embedding_dim == 2
    assert table.n_drugs == 2
    np.testing.assert_allclose(table.embedding_map["D1"], np.array([2.0, 3.0]))


def test_load_drug_embeddings_ignores_generator_metadata(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.csv"
    pd.DataFrame(
        {
            "drug_id": ["D1"],
            "canonical_smiles": ["CCC"],
            "n_response_rows": [12],
            "encoder_name": ["toy"],
            "encoder_revision": ["abc123"],
            "pooling": ["mean"],
            "emb_0": [1.0],
            "emb_1": [2.0],
        }
    ).to_csv(path, index=False)

    table = load_drug_embeddings(path)

    assert table.embedding_columns == ("emb_0", "emb_1")


def test_build_drug_embedding_matrix_errors_on_missing_key() -> None:
    rows = pd.DataFrame({"drug_id": ["D1", "D2"]})
    embedding_map = {"D1": np.array([1.0, 2.0], dtype=np.float32)}

    with pytest.raises(ValueError, match="Missing embeddings"):
        build_drug_embedding_matrix(rows, embedding_map)


def test_build_drug_embedding_matrix_can_zero_missing_key() -> None:
    rows = pd.DataFrame({"drug_id": ["D1", "D2"]})
    embedding_map = {"D1": np.array([1.0, 2.0], dtype=np.float32)}

    matrix = build_drug_embedding_matrix(rows, embedding_map, missing="zero")

    np.testing.assert_allclose(matrix[0], np.array([1.0, 2.0]))
    np.testing.assert_allclose(matrix[1], np.array([0.0, 0.0]))
