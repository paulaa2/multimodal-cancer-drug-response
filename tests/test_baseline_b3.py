import numpy as np
import pandas as pd
import pytest

from mcdrp.features.fingerprints import build_fingerprint_matrix
from mcdrp.models.baseline_b3 import run_b3_for_split


@pytest.fixture()
def synthetic_cohort() -> pd.DataFrame:
    """Create a small synthetic cohort with valid SMILES."""

    rng = np.random.default_rng(42)
    cell_ids = [f"ACH-{i:06d}" for i in range(6)]
    drug_ids = [f"D{i}" for i in range(5)]
    smiles_map = {
        "D0": "CCO",
        "D1": "CC(=O)O",
        "D2": "c1ccccc1",
        "D3": "CC(=O)Oc1ccccc1C(=O)O",
        "D4": "CC",
    }

    rows = []
    for i in range(30):
        drug_id = drug_ids[i % len(drug_ids)]
        rows.append(
            {
                "pair_id": f"pair_{i:07d}",
                "depmap_id": cell_ids[i % len(cell_ids)],
                "drug_id": drug_id,
                "canonical_smiles": smiles_map[drug_id],
                "ln_ic50": rng.normal(2.5, 2.0),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_split(synthetic_cohort: pd.DataFrame) -> pd.DataFrame:
    """Assign first 18 rows to train, next 6 to validation, last 6 to test."""

    labels = ["train"] * 18 + ["validation"] * 6 + ["test"] * 6
    return pd.DataFrame(
        {"pair_id": synthetic_cohort["pair_id"], "split": labels}
    )


@pytest.fixture()
def synthetic_cell_features() -> dict[str, np.ndarray]:
    """Create a fake cell feature map simulating PCA output."""

    rng = np.random.default_rng(42)
    return {
        f"ACH-{i:06d}": rng.standard_normal(10).astype(np.float32)
        for i in range(6)
    }


def test_run_b3_for_split_returns_validation_and_test_metrics(
    synthetic_cohort: pd.DataFrame,
    synthetic_split: pd.DataFrame,
    synthetic_cell_features: dict[str, np.ndarray],
) -> None:
    fp_matrix = build_fingerprint_matrix(synthetic_cohort["canonical_smiles"])

    results, frames = run_b3_for_split(
        synthetic_cohort,
        synthetic_split,
        fp_matrix,
        synthetic_cell_features,
        split_name="random_pair",
        target="ln_ic50",
        hidden_layer_sizes=(8,),
        alpha=0.0001,
        learning_rate_init=0.001,
        batch_size=8,
        max_iter=5,
        early_stopping=False,
        validation_fraction=0.1,
        random_state=42,
        device="cpu",
    )

    assert len(results) == 2
    assert {row["subset"] for row in results} == {"validation", "test"}
    for row in results:
        assert row["model"] == "mlp"
        assert row["rmse"] >= 0
        assert row["n_iter"] >= 1
        assert "loss" in row
    assert len(frames) == len(results)
    assert frames[0]["stage"].unique().tolist() == ["B3"]
