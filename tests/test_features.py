"""Tests for the features subpackage (fingerprints and expression PCA)."""

import numpy as np
import pandas as pd
import pytest

from mcdrp.features.expression import (
    fit_expression_pipeline,
    transform_expression,
)
from mcdrp.features.fingerprints import build_fingerprint_matrix, smiles_to_morgan
from mcdrp.features.pathways import (
    PathwayDefinition,
    expression_column_to_symbol,
    fit_pathway_pipeline,
    transform_pathways,
)

# ---------------------------------------------------------------------------
# Morgan fingerprint tests
# ---------------------------------------------------------------------------


class TestSmilesToMorgan:
    """Tests for single-SMILES fingerprint conversion."""

    def test_valid_smiles_returns_correct_shape(self) -> None:
        # Aspirin SMILES
        fp = smiles_to_morgan("CC(=O)Oc1ccccc1C(=O)O")
        assert fp.shape == (2048,)
        assert fp.dtype == np.uint8

    def test_valid_smiles_has_nonzero_bits(self) -> None:
        fp = smiles_to_morgan("CC(=O)Oc1ccccc1C(=O)O")
        assert fp.sum() > 0, "Morgan FP for aspirin should have on-bits."

    def test_custom_n_bits(self) -> None:
        fp = smiles_to_morgan("CCO", n_bits=1024)
        assert fp.shape == (1024,)

    def test_custom_radius(self) -> None:
        fp_r2 = smiles_to_morgan("c1ccccc1", radius=2)
        fp_r3 = smiles_to_morgan("c1ccccc1", radius=3)
        # Different radii may produce different fingerprints.
        assert fp_r2.shape == fp_r3.shape == (2048,)

    def test_invalid_smiles_returns_zeros(self) -> None:
        fp = smiles_to_morgan("NOT_A_REAL_SMILES_STRING")
        assert fp.shape == (2048,)
        assert fp.sum() == 0

    def test_empty_string_returns_zeros(self) -> None:
        fp = smiles_to_morgan("")
        assert fp.shape == (2048,)
        assert fp.sum() == 0


class TestBuildFingerprintMatrix:
    """Tests for batch fingerprint matrix construction."""

    def test_matrix_shape(self) -> None:
        smiles = pd.Series(["CCO", "CC(=O)O", "c1ccccc1"])
        matrix = build_fingerprint_matrix(smiles)
        assert matrix.shape == (3, 2048)
        assert matrix.dtype == np.uint8

    def test_matrix_with_invalid_smiles_warns(self) -> None:
        smiles = pd.Series(["CCO", "INVALID", "c1ccccc1"])
        with pytest.warns(UserWarning, match="1/3 SMILES failed"):
            matrix = build_fingerprint_matrix(smiles)
        assert matrix.shape == (3, 2048)
        # The invalid row should be all zeros.
        assert matrix[1].sum() == 0
        # Valid rows should have on-bits.
        assert matrix[0].sum() > 0
        assert matrix[2].sum() > 0

    def test_custom_bits_in_matrix(self) -> None:
        smiles = pd.Series(["CCO", "CC"])
        matrix = build_fingerprint_matrix(smiles, n_bits=512)
        assert matrix.shape == (2, 512)


# ---------------------------------------------------------------------------
# Expression PCA tests
# ---------------------------------------------------------------------------


class TestExpressionPCA:
    """Tests for expression PCA pipeline."""

    @pytest.fixture()
    def synthetic_expression(self) -> pd.DataFrame:
        """Create a small synthetic expression matrix."""
        rng = np.random.default_rng(42)
        n_cells = 20
        n_genes = 100
        model_ids = [f"ACH-{i:06d}" for i in range(n_cells)]
        gene_names = [f"GENE{i} ({i})" for i in range(n_genes)]
        data = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
        return pd.DataFrame(data, index=model_ids, columns=gene_names)

    def test_fit_transform_shape(self, synthetic_expression: pd.DataFrame) -> None:
        pipeline = fit_expression_pipeline(synthetic_expression, n_components=10)
        transformed = transform_expression(pipeline, synthetic_expression)
        assert transformed.shape == (20, 10)
        assert transformed.dtype == np.float32

    def test_pipeline_attributes(self, synthetic_expression: pd.DataFrame) -> None:
        pipeline = fit_expression_pipeline(synthetic_expression, n_components=5)
        assert pipeline.n_components == 5
        assert len(pipeline.gene_columns) == 100
        assert 0.0 < pipeline.explained_variance_ratio_sum <= 1.0

    def test_n_components_capped_to_min_dimension(self) -> None:
        """If n_components > min(n_samples, n_features), it should be capped."""
        rng = np.random.default_rng(0)
        # Only 5 cells × 10 genes → max 5 components.
        small = pd.DataFrame(
            rng.standard_normal((5, 10)),
            index=[f"C{i}" for i in range(5)],
            columns=[f"G{i}" for i in range(10)],
        )
        pipeline = fit_expression_pipeline(small, n_components=256)
        assert pipeline.n_components == 5
        transformed = transform_expression(pipeline, small)
        assert transformed.shape == (5, 5)

    def test_leakage_free_transform(
        self, synthetic_expression: pd.DataFrame
    ) -> None:
        """Fit on a subset, transform a disjoint subset — should work."""
        train = synthetic_expression.iloc[:12]
        test = synthetic_expression.iloc[12:]

        pipeline = fit_expression_pipeline(train, n_components=8)
        test_transformed = transform_expression(pipeline, test)
        assert test_transformed.shape == (8, 8)

    def test_transform_preserves_order(
        self, synthetic_expression: pd.DataFrame
    ) -> None:
        """Transforming twice should give the same result."""
        pipeline = fit_expression_pipeline(synthetic_expression, n_components=5)
        t1 = transform_expression(pipeline, synthetic_expression)
        t2 = transform_expression(pipeline, synthetic_expression)
        np.testing.assert_array_equal(t1, t2)


class TestPathwayFeatures:
    """Tests for pathway activity feature construction."""

    def test_expression_column_to_symbol_strips_entrez_suffix(self) -> None:
        assert expression_column_to_symbol("TP53 (7157)") == "TP53"

    def test_pathway_scores_have_expected_shape(self) -> None:
        expression = pd.DataFrame(
            {
                "TP53 (7157)": [1.0, 2.0, 3.0],
                "MDM2 (4193)": [3.0, 2.0, 1.0],
                "CDKN1A (1026)": [1.0, 1.0, 1.0],
                "EGFR (1956)": [0.0, 1.0, 2.0],
            },
            index=["c1", "c2", "c3"],
        )
        definitions = [
            PathwayDefinition("P53_TEST", ("TP53", "MDM2", "CDKN1A")),
            PathwayDefinition("TOO_SMALL", ("EGFR",)),
        ]

        pipeline = fit_pathway_pipeline(expression, definitions, min_genes=2)
        scores = transform_pathways(pipeline, expression)

        assert pipeline.pathway_names == ["P53_TEST"]
        assert scores.shape == (3, 1)
        assert scores.dtype == np.float32
