import numpy as np
import pandas as pd
import pytest

from mcdrp.features.fingerprints import build_fingerprint_matrix
from mcdrp.models.baseline_b1 import assemble_features
from mcdrp.models.baseline_b2 import run_b2_for_split


@pytest.fixture()
def synthetic_cohort() -> pd.DataFrame:
    """Create a small synthetic cohort with valid SMILES."""
    rng = np.random.default_rng(42)
    n = 30
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
    for i in range(n):
        did = drug_ids[i % len(drug_ids)]
        rows.append(
            {
                "pair_id": f"pair_{i:07d}",
                "depmap_id": cell_ids[i % len(cell_ids)],
                "drug_id": did,
                "canonical_smiles": smiles_map[did],
                "ln_ic50": rng.normal(2.5, 2.0),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_split(synthetic_cohort: pd.DataFrame) -> pd.DataFrame:
    """Assign first 18 rows to train, next 6 to validation, last 6 to test."""
    n = len(synthetic_cohort)
    labels = ["train"] * 18 + ["validation"] * 6 + ["test"] * 6
    return pd.DataFrame(
        {"pair_id": synthetic_cohort["pair_id"], "split": labels[:n]}
    )


@pytest.fixture()
def synthetic_cell_features() -> dict[str, np.ndarray]:
    """Create a fake cell feature map (simulating PCA output)."""
    rng = np.random.default_rng(42)
    return {
        f"ACH-{i:06d}": rng.standard_normal(10).astype(np.float32)
        for i in range(6)
    }


class TestRunB2ForSplit:
    """Smoke test for the full B2 XGBoost pipeline on synthetic data."""

    def test_pipeline_runs_without_error(
        self,
        synthetic_cohort: pd.DataFrame,
        synthetic_split: pd.DataFrame,
        synthetic_cell_features: dict[str, np.ndarray],
    ) -> None:
        fp_matrix = build_fingerprint_matrix(synthetic_cohort["canonical_smiles"])

        xgb_params = {
            "n_estimators": 10,
            "learning_rate": 0.1,
            "max_depth": 3,
            "early_stopping_rounds": 3,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": 1,
        }

        results = run_b2_for_split(
            synthetic_cohort,
            synthetic_split,
            fp_matrix,
            synthetic_cell_features,
            split_name="random_pair",
            target="ln_ic50",
            xgb_params=xgb_params,
            device="cpu",
        )
        # 1 model × 2 subsets = 2 result rows.
        assert len(results) == 2
        for r in results:
            assert r["model"] == "xgboost"
            assert r["subset"] in ("validation", "test")
            assert "rmse" in r
            assert "pearson" in r
            assert r["rmse"] >= 0
            assert "best_iteration" in r
