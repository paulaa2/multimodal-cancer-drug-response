"""Resolve complete chemical identities before partitioning drug records."""

from __future__ import annotations

import pandas as pd


def annotate_chemical_identity(cohort: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep all measurements, but group aliases by the complete standard InChIKey.

    This holds out exact chemical identities, including stereochemistry. It does
    not assert scaffold novelty or equivalence of different salts/protonation states.
    """
    from rdkit import Chem

    catalog = cohort[["drug_id", "canonical_smiles"]].drop_duplicates().copy()
    if catalog["drug_id"].duplicated().any():
        raise ValueError("A drug ID has inconsistent chemical structures.")
    keys = {}
    for row in catalog.itertuples(index=False):
        molecule = Chem.MolFromSmiles(str(row.canonical_smiles))
        key = Chem.MolToInchiKey(molecule) if molecule is not None else ""
        if not key:
            raise ValueError(f"Cannot resolve the chemical identity of drug {row.drug_id}.")
        keys[str(row.drug_id)] = key
    result = cohort.copy()
    result["chemical_id"] = result["drug_id"].astype(str).map(keys)
    catalog["chemical_id"] = catalog["drug_id"].astype(str).map(keys)
    aliases = [
        {"chemical_id": key, "drug_ids": sorted(group["drug_id"].astype(str).tolist())}
        for key, group in catalog.groupby("chemical_id", sort=True)
        if len(group) > 1
    ]
    return result, {
        "grouping": "full_standard_inchikey",
        "drug_ids": len(catalog),
        "chemical_identities": catalog["chemical_id"].nunique(),
        "aliases": aliases,
        "note": "Aliases remain as measured records but cannot cross cold-drug subsets.",
    }
