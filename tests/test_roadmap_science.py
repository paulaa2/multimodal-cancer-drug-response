"""Scientific contracts for harmonization, entity nulls and independent uncertainty."""

import json

import numpy as np
import pandas as pd
import pytest

from mcdrp.data.harmonize_endpoints import (
    dose_to_log10_um,
    fit_shared_auc,
    mapped_doses,
    unique_name_lookup,
)
from mcdrp.data.target_annotations import classify_token, gene_lookup
from mcdrp.features.embedding_audit import audit_embeddings
from mcdrp.features.permutations import permute_entity_features
from mcdrp.results.uncertainty import conformal_radius, interval_diagnostics, paired_bootstrap
from mcdrp.splits.calibration import reserve_calibration


def test_explicit_dose_units_and_curve_area():
    doses = np.linspace(-2, 1, 16)
    assert np.allclose(dose_to_log10_um(10**doses, "linear_um"), doses)
    assert np.array_equal(dose_to_log10_um(doses, "log10_um"), doses)
    with pytest.raises(ValueError, match="positive"):
        dose_to_log10_um(np.array([0]), "linear_um")
    # For a symmetric logistic around the integration midpoint, mean viability is 0.5.
    y = 1 / (1 + np.exp(2 * (doses + 0.5)))
    fitted = fit_shared_auc(doses, y, lower=-2, upper=1)
    assert fitted["auc_harmonized"] == pytest.approx(0.5, abs=1e-4)
    assert fit_shared_auc(doses, y, lower=-3, upper=1)["status"] == "insufficient_dose_coverage"
    flat = fit_shared_auc(doses, np.full(len(doses), 0.8), lower=-2, upper=1)
    assert flat["status"] == "flat" and flat["auc_harmonized"] == pytest.approx(0.8)


def test_names_do_not_erase_stereochemistry_or_choose_ambiguity():
    table = pd.DataFrame(
        {"name": ["(R)-drug", "(S)-drug", "AMB", "amb"], "id": ["r", "s", "1", "2"]}
    )
    assert unique_name_lookup(table, "name", "id") == {"(r)-drug": "r", "(s)-drug": "s"}


def test_changed_raw_curve_input_is_rejected_before_mapping(tmp_path):
    source = tmp_path / "curves.csv"
    source.write_text("corrupted input")
    with pytest.raises(ValueError, match="hash changed"):
        mapped_doses(
            {"path": str(source), "study": "prism", "sha256": "wrong"},
            cells={},
            drugs={},
            eligible=pd.DataFrame(),
        )


def test_entity_null_preserves_repeat_identity_and_partitions():
    ids = ["a", "b", "a", "c", "d", "d"]
    labels = ["train", "train", "validation", "test", "test", "test"]
    matrix = np.array([[1], [2], [1], [3], [4], [4]], dtype=np.float32)
    shuffled, donors = permute_entity_features(matrix, ids, labels, seed=7)
    assert np.array_equal(shuffled[0], shuffled[2])
    assert np.array_equal(shuffled[4], shuffled[5])
    signatures = donors.set_index("entity_id")["partition_signature"]
    for row in donors.itertuples():
        assert signatures[row.entity_id] == signatures[row.donor_id]
    broken = matrix.copy()
    broken[2] = 5
    with pytest.raises(ValueError, match="inconsistent"):
        permute_entity_features(broken, ids, labels, seed=7)


def test_hgnc_alias_is_unique_and_family_not_guessed():
    table = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "status": ["Approved", "Approved"],
            "alias_symbol": ["SHARED|OLD", "SHARED"],
            "prev_symbol": ["", ""],
        }
    )
    symbols, aliases = gene_lookup(table)
    assert classify_token("OLD", symbols, aliases)["normalized"] == "A"
    assert classify_token("SHARED", symbols, aliases)["resolution"] == "ambiguous_hgnc_alias"
    assert classify_token("PI3K", symbols, aliases)["kind"] == "family"


def test_pharmacological_family_has_priority_over_unique_gene_alias():
    symbols = {"pik3ca": "PIK3CA", "src": "SRC"}
    aliases = {"pi3k": {"PIK3CA"}, "akt": {"AKT1"}}
    assert classify_token("PI3K", symbols, aliases)["kind"] == "family"
    assert classify_token("AKT", symbols, aliases)["kind"] == "family"
    assert classify_token("PIK3CA", symbols, aliases)["kind"] == "gene"
    assert classify_token("SRC", symbols, aliases)["kind"] == "gene"


def test_calibration_entities_never_seen_by_training_or_validation():
    cohort = pd.DataFrame({"pair_id": range(12), "cell": list("abcdef") * 2, "y": range(12)})
    labels = pd.DataFrame(
        {"pair_id": range(12), "split": ["train"] * 6 + ["validation"] * 5 + ["test"]}
    )
    result = reserve_calibration(cohort, labels, group="cell", fraction=0.34, seed=42)
    merged = cohort.merge(result, on="pair_id")
    cal = set(merged.loc[merged["split"].eq("calibration"), "cell"])
    fitted = set(merged.loc[merged["split"].isin(["train", "validation"]), "cell"])
    assert not cal & fitted
    assert result.iloc[-1]["split"] == "test"
    changed = cohort.assign(y=-cohort["y"])
    pd.testing.assert_frame_equal(
        result, reserve_calibration(changed, labels, group="cell", fraction=0.34, seed=42)
    )


def test_stage_validator_accepts_calibration_and_rejects_entity_leakage(tmp_path):
    from mcdrp.results.validate_baseline_stage import validate_split

    cohort = pd.DataFrame(
        {
            "pair_id": range(6),
            "depmap_id": ["c"] * 6,
            "drug_id": ["a", "b", "v", "t", "cal", "unused"],
        }
    )
    assignments = pd.DataFrame(
        {
            "pair_id": range(6),
            "split": ["train", "train", "validation", "test", "calibration", "unused"],
        }
    )
    assignments.to_csv(tmp_path / "cold_drug.csv", index=False)
    assert validate_split(cohort, tmp_path, "cold_drug").status == "pass"
    cohort.loc[4, "drug_id"] = "a"
    assert validate_split(cohort, tmp_path, "cold_drug").status == "fail"


def test_conformal_uses_groups_and_finite_sample_order_statistic():
    residuals = np.arange(1, 21, dtype=float)
    groups = np.repeat(np.arange(10), 2)
    assert conformal_radius(residuals, groups, 0.2) == 18
    assert np.isinf(conformal_radius(residuals[:4], groups[:4], 0.1))
    report = interval_diagnostics(
        np.array([0.2, 0.5]), np.array([-3.0, 4.0]), np.array([1, 2]), 10, (0, 1)
    )
    assert report["mean_width"] == 1 and report["whole_group_coverage"] == 1


def test_paired_bootstrap_is_paired_and_reproducible():
    y = np.arange(12, dtype=float)
    groups = np.repeat(np.arange(4), 3)
    same = paired_bootstrap(y, y + 1, y + 1, groups, repeats=50)
    assert same == {"rmse_delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    better = paired_bootstrap(y, y, y + 1, groups, repeats=50)
    assert better["ci95_high"] < 0


def test_legacy_embedding_revision_cannot_pass_frozen_audit(tmp_path):
    cohort = tmp_path / "cohort.csv"
    embedding = tmp_path / "embedding.csv"
    manifest = tmp_path / "manifest.json"
    pd.DataFrame({"drug_id": ["1"], "canonical_smiles": ["CC"]}).to_csv(cohort, index=False)
    pd.DataFrame({"drug_id": ["1"]}).to_csv(embedding, index=False)
    manifest.write_text(json.dumps({"resolved_revision": "main"}))
    with pytest.raises(ValueError, match="checkpoint"):
        audit_embeddings(str(cohort), str(embedding), str(manifest))


def test_graph_only_ignores_cell_input():
    torch = pytest.importorskip("torch")
    from mcdrp.models.gnn_b4 import GraphBatch
    from mcdrp.models.graph_only import GraphOnlyRegressor

    torch.manual_seed(7)
    model = GraphOnlyRegressor(hidden_dim=8, layers=1).eval()
    graph = torch.ones(2, 3, 32)
    batch = GraphBatch(
        graph, torch.eye(3).repeat(2, 1, 1), torch.ones(2, 3), torch.zeros(2, 5), torch.zeros(2)
    )
    with torch.no_grad():
        before = model(batch)
        batch.cell_features = torch.randn(2, 5) * 1000
        assert torch.equal(before, model(batch))


def test_graph_aggregation_preserves_pair_mse():
    y = np.array([1.0, 3.0, 5.0, 6.0, 9.0])
    predictions = np.array([2.0, 7.0])
    pair_pred = predictions[[0, 0, 1, 1, 1]]
    means = np.array([y[:2].mean(), y[2:].mean()])
    weights = np.array([2, 3]) / 5
    constant = weights @ np.array([y[:2].var(), y[2:].var()])
    assert np.mean((y - pair_pred) ** 2) == pytest.approx(
        weights @ (means - predictions) ** 2 + constant
    )
