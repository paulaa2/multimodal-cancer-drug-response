from pathlib import Path

import numpy as np
import pandas as pd

from mcdrp.features.create_drug_embeddings import (
    extract_unique_drugs,
    write_embedding_table,
)


def test_extract_unique_drugs_keeps_most_frequent_smiles() -> None:
    cohort = pd.DataFrame(
        {
            "drug_id": ["D1", "D1", "D1", "D2"],
            "canonical_smiles": ["CCC", "CCC", "CCN", "OOO"],
        }
    )

    drugs = extract_unique_drugs(cohort)

    assert drugs.to_dict("records") == [
        {"drug_id": "D1", "canonical_smiles": "CCC", "n_response_rows": 3},
        {"drug_id": "D2", "canonical_smiles": "OOO", "n_response_rows": 1},
    ]


def test_write_embedding_table_uses_b7_column_contract(tmp_path: Path) -> None:
    drugs = pd.DataFrame(
        {
            "drug_id": ["D1", "D2"],
            "canonical_smiles": ["CCC", "OOO"],
            "n_response_rows": [3, 1],
        }
    )
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    output = write_embedding_table(
        drugs,
        embeddings,
        tmp_path / "drug_embeddings.csv",
        model_name="toy-model",
        revision="abc123",
        pooling="mean",
    )

    table = pd.read_csv(output)
    assert table.columns.tolist() == [
        "drug_id",
        "canonical_smiles",
        "n_response_rows",
        "encoder_name",
        "encoder_revision",
        "pooling",
        "emb_0",
        "emb_1",
    ]
    assert table.loc[0, "emb_0"] == 1.0
