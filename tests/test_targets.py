"""Tests for GDSC putative-target profiles and the B6 target ablations."""

import numpy as np
import pandas as pd
import pytest

from mcdrp.features.targets import (
    build_target_matrix,
    fit_target_pipeline,
    parse_putative_targets,
)
from mcdrp.models.ablation_b6 import assemble_ablation_features


class TestParsePutativeTargets:
    def test_gene_list_is_split_on_commas(self) -> None:
        assert parse_putative_targets("MEK1, MEK2") == ["MEK1", "MEK2"]

    def test_mechanism_phrase_is_kept_whole(self) -> None:
        assert parse_putative_targets("Microtubule destabiliser") == [
            "Microtubule destabiliser"
        ]

    def test_hdac_class_suffixes_are_dropped(self) -> None:
        tokens = parse_putative_targets("HDAC inhibitor Class I, IIa, IIb, IV")
        assert tokens == ["HDAC inhibitor Class I"]
        assert "IIa" not in tokens

    def test_missing_and_blank_are_empty(self) -> None:
        assert parse_putative_targets(None) == []
        assert parse_putative_targets("  ") == []
        assert parse_putative_targets(" ") == []

    def test_whitespace_and_duplicates_are_collapsed(self) -> None:
        assert parse_putative_targets("BCL2,  BCL-XL,  BCL2") == ["BCL2", "BCL-XL"]


class TestTargetPipeline:
    def test_vocabulary_comes_from_training_drugs_only(self) -> None:
        cohort = pd.DataFrame(
            {
                "pair_id": ["p1", "p2", "p3"],
                "drug_id": ["D1", "D2", "D3"],
                "putative_target": ["EGFR", "EGFR, MEK1", "PARP1"],
            }
        )
        pipeline, matrix = build_target_matrix(cohort, {"D1", "D2"})

        assert pipeline.target_names == ["EGFR", "MEK1"]
        assert "PARP1" not in pipeline.target_names
        # D3's unique target is not in the vocabulary, so the row is zeros.
        assert matrix[2].sum() == 0
        assert matrix[0, pipeline.target_names.index("EGFR")] == 1.0
        assert matrix[1, pipeline.target_names.index("MEK1")] == 1.0

    def test_empty_training_vocabulary_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty target vocabulary"):
            fit_target_pipeline(pd.Series([None, "  "], index=["D1", "D2"]))

    def test_missing_column_is_rejected(self) -> None:
        cohort = pd.DataFrame({"pair_id": ["p1"], "drug_id": ["D1"]})
        with pytest.raises(ValueError, match="putative_target"):
            build_target_matrix(cohort, {"D1"})


class TestAssembleTargets:
    def cohort(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "pair_id": ["p0", "p1"],
                "depmap_id": ["C0", "C1"],
                "drug_id": ["D0", "D1"],
            }
        )

    def test_targets_block_is_selected_without_morgan(self) -> None:
        cohort = self.cohort()
        target_matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        pca_map = {
            "C0": np.array([9.0, 8.0], dtype=np.float32),
            "C1": np.array([7.0, 6.0], dtype=np.float32),
        }
        assembled = assemble_ablation_features(
            cohort.iloc[[0]],
            ablation="targets_expression_pca",
            cohort=cohort,
            fp_matrix=np.ones((2, 4), dtype=np.float32),
            pca_feature_map=pca_map,
            pathway_feature_map={},
            target_matrix=target_matrix,
        )
        assert assembled.shape == (1, 4)
        np.testing.assert_array_equal(assembled[0, :2], [1.0, 0.0])
        np.testing.assert_array_equal(assembled[0, 2:], [9.0, 8.0])
