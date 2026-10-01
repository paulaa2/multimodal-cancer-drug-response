"""Map PRISM secondary-screen responses onto GDSC drugs by InChIKey.

This is the ingest for the GDSC2 → PRISM transfer experiment. Matching is
based on chemical identity: a PRISM compound joins a GDSC drug on a unique
full InChIKey, or otherwise a unique 14-character connectivity block. The
latter ignores stereochemistry. Ambiguous/unmatched compounds are dropped.

PRISM's own target annotations are ignored. Mapped rows inherit GDSC
``putative_target`` so the transfer tests the same drug representation that
won leave-drug-out on GDSC, not a second vocabulary.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PRISM_RESPONSE_PATH = "data/raw/secondary-screen-dose-response-curve-parameters.csv"
PREFERRED_SCREEN_ID = "MTS010"
STUDY_PRISM = "prism"
STUDY_GDSC = "gdsc2"


def _smiles_candidates(smiles: str) -> list[str]:
    """Split a PRISM SMILES cell into parseable candidates.

    The secondary-screen table stores repeated structures or candidates in the
    same comma-separated field. Distinct parsed identities are rejected.
    """

    text = str(smiles).strip().strip('"')
    return [part.strip() for part in text.split(",") if part.strip()]


def _smiles_inchikeys(smiles: str) -> set[str]:
    """Parse every distinct candidate without choosing by list order."""

    candidates = _smiles_candidates(smiles)
    if not candidates:
        return set()
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import inchi
    except ImportError as exc:  # pragma: no cover - optional extra.
        raise RuntimeError(
            "PRISM mapping requires RDKit. Install the baselines extra."
        ) from exc

    RDLogger.DisableLog("rdApp.error")
    keys: set[str] = set()
    for text in set(candidates):
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            continue
        try:
            key = inchi.MolToInchiKey(mol)
        except Exception:
            continue
        if key:
            keys.add(key)
    return keys


def smiles_to_inchikey(smiles: str) -> str | None:
    """Return one unambiguous identity, or None for invalid/ambiguous lists."""

    keys = _smiles_inchikeys(smiles)
    return next(iter(keys)) if len(keys) == 1 else None


def inchikey_connectivity(key: str) -> str:
    """Return the 14-character connectivity block of an InChIKey."""

    return key.split("-", 1)[0]


def _column(frame: pd.DataFrame, *candidates: str) -> str:
    """Resolve a column name case-insensitively."""

    lookup = {str(name).lower(): name for name in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(
        f"PRISM table is missing one of {candidates}. Columns: {list(frame.columns)}"
    )


def load_prism_responses(
    path: str = PRISM_RESPONSE_PATH,
    *,
    preferred_screen: str = PREFERRED_SCREEN_ID,
) -> pd.DataFrame:
    """Load PRISM dose-response rows, preferring the MTS010 technical redo."""

    raw = pd.read_csv(path, low_memory=False)
    depmap_col = _column(raw, "depmap_id")
    smiles_col = _column(raw, "smiles")
    auc_col = _column(raw, "auc")
    name_col = _column(raw, "name")

    frame = pd.DataFrame(
        {
            "prism_name": raw[name_col].astype(str),
            "depmap_id": raw[depmap_col].astype(str),
            "prism_smiles": raw[smiles_col],
            "auc": pd.to_numeric(raw[auc_col], errors="coerce"),
        }
    )
    if "ic50" in {str(c).lower() for c in raw.columns}:
        ic50_col = _column(raw, "ic50")
        frame["prism_ic50"] = pd.to_numeric(raw[ic50_col], errors="coerce")
    else:
        frame["prism_ic50"] = np.nan

    if "screen_id" in {str(c).lower() for c in raw.columns}:
        screen_col = _column(raw, "screen_id")
        frame["screen_id"] = raw[screen_col].astype(str)
        preferred_mask = frame["screen_id"].eq(preferred_screen)
        if preferred_mask.any():
            frame = frame.loc[preferred_mask].copy()
            logger.info("Keeping PRISM screen_id=%s (%d rows)", preferred_screen, len(frame))

    profiling_names = {str(c).lower() for c in raw.columns}
    if "passed_str_profiling" in profiling_names:
        flag_col = _column(raw, "passed_str_profiling")
        flags = raw.loc[frame.index, flag_col]
        keep = flags.isna() | flags.astype(str).str.lower().isin(("true", "1", "yes"))
        frame = frame.loc[keep.to_numpy()].copy()

    frame = frame.loc[np.isfinite(frame["auc"]) & frame["depmap_id"].str.startswith("ACH-")]
    logger.info("PRISM responses with AUC and DepMap IDs: %d", len(frame))
    return frame.reset_index(drop=True)


def map_prism_drugs(
    prism: pd.DataFrame,
    gdsc_drugs: pd.DataFrame,
) -> pd.DataFrame:
    """Attach GDSC ``drug_id`` to PRISM rows by InChIKey.

    ``gdsc_drugs`` must have ``drug_id``, ``inchi_key``, and a name column.
    """

    if "inchi_key" not in gdsc_drugs.columns:
        raise ValueError("GDSC drug table needs an inchi_key column.")

    name_col = "gdsc_drug_name" if "gdsc_drug_name" in gdsc_drugs.columns else "drug_name"
    keep_cols = ["drug_id", "inchi_key"]
    if name_col in gdsc_drugs.columns:
        keep_cols.append(name_col)
    catalog = gdsc_drugs.loc[gdsc_drugs["inchi_key"].notna(), keep_cols].drop_duplicates(
        "drug_id"
    )
    catalog = catalog.copy()
    if name_col != "gdsc_drug_name" and name_col in catalog.columns:
        catalog = catalog.rename(columns={name_col: "gdsc_drug_name"})
    if "gdsc_drug_name" not in catalog.columns:
        catalog["gdsc_drug_name"] = ""
    catalog["inchi_key"] = catalog["inchi_key"].astype(str).str.strip()
    catalog = catalog.loc[catalog["inchi_key"].str.len().gt(0)]
    catalog["connectivity"] = catalog["inchi_key"].map(inchikey_connectivity)
    full_pairs = catalog.loc[
        ~catalog["inchi_key"].duplicated(keep=False),
        ["drug_id", "inchi_key", "gdsc_drug_name"],
    ].itertuples(index=False, name=None)
    full_lookup = {
        key: (str(drug_id), str(name) if pd.notna(name) else "")
        for drug_id, key, name in full_pairs
    }
    # Connectivity omits stereochemistry: never choose among several candidates.
    connectivity_lookup: dict[str, tuple[str, str]] = {}
    conn_pairs = catalog.loc[
        ~catalog["connectivity"].duplicated(keep=False),
        ["drug_id", "connectivity", "gdsc_drug_name"],
    ].itertuples(index=False, name=None)
    for drug_id, conn, name in conn_pairs:
        connectivity_lookup.setdefault(
            conn, (str(drug_id), str(name) if pd.notna(name) else "")
        )

    unique_smiles = [str(s) for s in prism["prism_smiles"].dropna().unique()]
    smiles_keys: dict[str, set[str]] = {}
    for smiles in unique_smiles:
        smiles_keys[smiles] = _smiles_inchikeys(smiles)

    records = []
    for smiles, keys in smiles_keys.items():
        if len(keys) != 1:
            records.append(
                {
                    "prism_smiles": smiles,
                    "prism_inchikey": None,
                    "drug_id": None,
                    "gdsc_drug_name": None,
                    "match_type": "ambiguous_smiles" if keys else "unparseable",
                }
            )
            continue
        key = next(iter(keys))
        if key in full_lookup:
            drug_id, name = full_lookup[key]
            match_type = "full_inchikey"
        else:
            conn = inchikey_connectivity(key)
            if conn in connectivity_lookup:
                drug_id, name = connectivity_lookup[conn]
                match_type = "connectivity"
            else:
                drug_id, name, match_type = None, None, "unmatched"
        records.append(
            {
                "prism_smiles": smiles,
                "prism_inchikey": key,
                "drug_id": drug_id,
                "gdsc_drug_name": name,
                "match_type": match_type,
            }
        )
    mapping = pd.DataFrame.from_records(records)
    logger.info(
        "PRISM SMILES mapped: %s",
        mapping["match_type"].value_counts().to_dict(),
    )
    return mapping


def collapse_prism_pairs(mapped: pd.DataFrame) -> pd.DataFrame:
    """Keep one AUC per (cell, GDSC drug) by taking the median."""

    ready = mapped.loc[mapped["drug_id"].notna()].copy()
    if ready.empty:
        return ready
    grouped = ready.groupby(["depmap_id", "drug_id"], as_index=False).agg(
        auc=("auc", "median"),
        prism_name=("prism_name", "first"),
        prism_smiles=("prism_smiles", "first"),
        prism_inchikey=("prism_inchikey", "first"),
        match_type=("match_type", "first"),
        n_prism_rows=("auc", "size"),
    )
    grouped["drug_id"] = grouped["drug_id"].astype(str)
    grouped["depmap_id"] = grouped["depmap_id"].astype(str)
    grouped["study"] = STUDY_PRISM
    return grouped
