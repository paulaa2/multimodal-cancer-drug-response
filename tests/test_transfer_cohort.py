"""Tests for GDSC2 → PRISM transfer ingest and split construction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mcdrp.data.build_transfer_cohort import (
    build_transfer_assignments,
    build_transfer_cohort,
    validate_transfer_split,
)
from mcdrp.data.prism import collapse_prism_pairs, map_prism_drugs, smiles_to_inchikey
from mcdrp.results.validate_baseline_stage import validate_split
from mcdrp.splits.make_splits import TRANSFER_SPLIT

pytest.importorskip("rdkit")


def _inchikey(smiles: str) -> str:
    key = smiles_to_inchikey(smiles)
    assert key is not None
    return key


def test_prism_maps_on_full_inchikey_not_on_name() -> None:
    ethanol = _inchikey("CCO")
    gdsc = pd.DataFrame(
        {
            "drug_id": ["1003"],
            "inchi_key": [ethanol],
            "gdsc_drug_name": ["Camptothecin"],
        }
    )
    prism = pd.DataFrame({"prism_smiles": ["CCO", "c1ccccc1"]})
    mapping = map_prism_drugs(prism, gdsc)
    by_smiles = mapping.set_index("prism_smiles")
    assert by_smiles.loc["CCO", "drug_id"] == "1003"
    assert by_smiles.loc["CCO", "match_type"] == "full_inchikey"
    assert by_smiles.loc["c1ccccc1", "match_type"] == "unmatched"


@pytest.mark.parametrize("smiles", ["CCO, CCN", "CCN, CCO"])
def test_prism_rejects_distinct_identities_in_a_smiles_list(smiles) -> None:
    catalog = pd.DataFrame({"drug_id": ["1003"], "inchi_key": [_inchikey("CCO")]})
    mapped = map_prism_drugs(pd.DataFrame({"prism_smiles": [smiles]}), catalog)
    assert mapped.iloc[0]["match_type"] == "ambiguous_smiles"
    assert pd.isna(mapped.iloc[0]["drug_id"])
    assert smiles_to_inchikey(smiles) is None


def test_trailing_comma_on_prism_smiles_still_maps() -> None:
    ethanol = _inchikey("CCO")
    gdsc = pd.DataFrame(
        {"drug_id": ["1003"], "inchi_key": [ethanol], "gdsc_drug_name": ["ethanol"]}
    )
    mapping = map_prism_drugs(pd.DataFrame({"prism_smiles": ["CCO, CCO, CCO"]}), gdsc)
    assert mapping.iloc[0]["match_type"] == "full_inchikey"


def test_ambiguous_chemistry_is_not_assigned_by_catalog_order() -> None:
    key = _inchikey("CCO")
    catalog = pd.DataFrame({"drug_id": ["1", "2"], "inchi_key": [key, key]})
    for frame in (catalog, catalog.iloc[::-1]):
        mapping = map_prism_drugs(pd.DataFrame({"prism_smiles": ["CCO"]}), frame)
        assert mapping.iloc[0]["match_type"] == "unmatched"
        assert pd.isna(mapping.iloc[0]["drug_id"])


def test_collapse_keeps_median_auc_per_cell_drug() -> None:
    mapped = pd.DataFrame(
        {
            "depmap_id": ["ACH-1", "ACH-1", "ACH-2"],
            "drug_id": ["1003", "1003", "1003"],
            "auc": [0.2, 0.4, 0.5],
            "prism_name": ["a", "a", "a"],
            "prism_smiles": ["CCO", "CCO", "CCO"],
            "prism_inchikey": ["k", "k", "k"],
            "match_type": ["full_inchikey"] * 3,
        }
    )
    collapsed = collapse_prism_pairs(mapped)
    ethanol = collapsed.set_index("depmap_id")
    assert ethanol.loc["ACH-1", "auc"] == pytest.approx(0.3)
    assert len(collapsed) == 2


def test_transfer_split_holds_prism_out_of_train(tmp_path: Path) -> None:
    gdsc = pd.DataFrame(
        {
            "pair_id": ["pair_0", "pair_1", "pair_2"],
            "depmap_id": ["ACH-1", "ACH-1", "ACH-2"],
            "drug_id": ["1003", "1004", "1003"],
            "auc": [0.8, 0.7, 0.6],
            "inchi_key": [_inchikey("CCO")] * 3,
            "gdsc_drug_name": ["ethanol"] * 3,
            "putative_target": ["EGFR"] * 3,
            "canonical_smiles": ["CCO"] * 3,
            "oncotree_lineage": ["Lung"] * 3,
            "has_expression": [True] * 3,
        }
    )
    random_pair = pd.DataFrame(
        {
            "pair_id": ["pair_0", "pair_1", "pair_2"],
            "split": ["train", "validation", "test"],
        }
    )
    prism_rows = pd.DataFrame(
        {
            "pair_id": ["prism_0000000"],
            "depmap_id": ["ACH-1"],
            "drug_id": ["1003"],
            "auc": [0.4],
            "study": ["prism"],
        }
    )
    cohort, assignments = build_transfer_assignments(gdsc, prism_rows, random_pair)
    validate_transfer_split(cohort, assignments)
    labeled = cohort.merge(assignments, on="pair_id")
    assert set(labeled.loc[labeled["split"].eq("test"), "pair_id"]) == {"prism_0000000"}
    assert "pair_2" in set(labeled.loc[labeled["split"].eq("train"), "pair_id"])
    assert set(labeled.loc[labeled["split"].eq("validation"), "pair_id"]) == {"pair_1"}

    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    assignments.to_csv(split_dir / f"{TRANSFER_SPLIT}.csv", index=False)
    result = validate_split(cohort, split_dir, TRANSFER_SPLIT)
    assert result.status == "pass"


def test_build_transfer_cohort_writes_mapped_prism_test(tmp_path: Path) -> None:
    gdsc = pd.DataFrame(
        {
            "pair_id": ["pair_0", "pair_1", "pair_2"],
            "depmap_id": ["ACH-000001", "ACH-000001", "ACH-000002"],
            "drug_id": ["1003", "1004", "1003"],
            "auc": [0.8, 0.7, 0.6],
            "ln_ic50": [1.0, 2.0, 1.5],
            "inchi_key": [_inchikey("CCO"), _inchikey("CCN"), _inchikey("CCO")],
            "gdsc_drug_name": ["ethanol", "ethylamine", "ethanol"],
            "putative_target": ["EGFR", "EGFR", "EGFR"],
            "canonical_smiles": ["CCO", "CCN", "CCO"],
            "oncotree_lineage": ["Lung"] * 3,
            "has_expression": [True] * 3,
        }
    )
    random_pair = pd.DataFrame(
        {
            "pair_id": ["pair_0", "pair_1", "pair_2"],
            "split": ["train", "validation", "test"],
        }
    )
    prism = pd.DataFrame(
        {
            "depmap_id": ["ACH-000001", "ACH-000001"],
            "name": ["ethanol", "benzene"],
            "smiles": ["CCO", "c1ccccc1"],
            "auc": [0.41, 0.9],
            "ic50": [1.0, 2.0],
            "screen_id": ["MTS010", "MTS010"],
            "passed_str_profiling": [True, True],
        }
    )
    gdsc_path = tmp_path / "gdsc.csv"
    split_path = tmp_path / "random_pair.csv"
    prism_path = tmp_path / "prism.csv"
    gdsc.to_csv(gdsc_path, index=False)
    random_pair.to_csv(split_path, index=False)
    prism.to_csv(prism_path, index=False)

    cohort, assignments, report = build_transfer_cohort(
        gdsc_cohort=gdsc_path,
        prism_path=prism_path,
        random_pair_split=split_path,
        output_cohort=tmp_path / "transfer_pairs.csv",
        output_split=tmp_path / "transfer_prism.csv",
        drug_map=tmp_path / "map.csv",
        summary=tmp_path / "summary.json",
    )
    labeled = cohort.merge(assignments, on="pair_id")
    test = labeled.loc[labeled["split"].eq("test")]
    assert len(test) == 1
    assert test.iloc[0]["drug_id"] == "1003"
    assert test.iloc[0]["auc"] == pytest.approx(0.41)
    assert test.iloc[0]["putative_target"] == "EGFR"
    assert report["counts"]["test_drugs"] == 1
    assert report["counts"]["drug_overlap_train_test"] == 1
    assert report["mapping"]["unmatched"] == 1
