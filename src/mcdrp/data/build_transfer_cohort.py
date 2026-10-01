"""Build the GDSC2-train / PRISM-test transfer cohort.

Training and validation stay inside GDSC2. The test set is PRISM secondary-
screen AUC for compounds that map onto GDSC drugs by InChIKey and cell lines
that have DepMap expression. Early stopping therefore cannot see PRISM.
Overlapping drugs and cells are allowed: that is the assay-shift question,
not a cold-start split.

The regression target is ``auc``. GDSC ln(IC50) and PRISM IC50 live on
different concentration grids; AUC is the quantity both screens report.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mcdrp.data.prism import (
    PRISM_RESPONSE_PATH,
    STUDY_GDSC,
    STUDY_PRISM,
    collapse_prism_pairs,
    load_prism_responses,
    map_prism_drugs,
)
from mcdrp.splits.make_splits import TRANSFER_SPLIT
from mcdrp.splits.validation import validate_assignments

logger = logging.getLogger(__name__)

GDSC_COHORT_PATH = "data/processed/cohort_pairs.csv"
RANDOM_PAIR_SPLIT_PATH = "data/processed/splits/random_pair.csv"
TRANSFER_COHORT_PATH = "data/processed/transfer_pairs.csv"
TRANSFER_SPLIT_PATH = "data/processed/splits/transfer_prism.csv"
DRUG_MAP_PATH = "data/mappings/prism_gdsc_drug_map.csv"
SUMMARY_PATH = "data/reports/transfer_prism_summary.json"


def _drug_catalog(gdsc: pd.DataFrame) -> pd.DataFrame:
    """One row per GDSC drug with the identifiers PRISM matching needs."""

    columns = ["drug_id", "inchi_key", "gdsc_drug_name", "putative_target", "canonical_smiles"]
    present = [column for column in columns if column in gdsc.columns]
    catalog = gdsc[present].drop_duplicates("drug_id").copy()
    catalog["drug_id"] = catalog["drug_id"].astype(str)
    return catalog


def build_prism_pairs(
    prism_path: str | Path,
    gdsc: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return collapsed PRISM pairs and the per-SMILES mapping table."""

    responses = load_prism_responses(str(prism_path))
    catalog = _drug_catalog(gdsc)
    mapping = map_prism_drugs(responses, catalog)
    merged = responses.merge(mapping, on="prism_smiles", how="left")
    expression_ids = set(gdsc["depmap_id"].astype(str))
    merged = merged.loc[merged["depmap_id"].astype(str).isin(expression_ids)].copy()
    pairs = collapse_prism_pairs(merged)
    return pairs, mapping


def format_prism_cohort_rows(pairs: pd.DataFrame, gdsc: pd.DataFrame) -> pd.DataFrame:
    """Bring PRISM pairs onto the GDSC cohort schema."""

    catalog = _drug_catalog(gdsc).set_index("drug_id")
    cell_cols = [
        column
        for column in (
            "depmap_id",
            "sanger_model_id",
            "gdsc_cell_line_name",
            "depmap_cell_line_name",
            "depmap_stripped_cell_line_name",
            "oncotree_lineage",
            "oncotree_primary_disease",
            "oncotree_subtype",
            "cosmic_id",
            "has_expression",
        )
        if column in gdsc.columns
    ]
    cells = gdsc[cell_cols].drop_duplicates("depmap_id").set_index("depmap_id")

    rows = pairs.copy()
    for column in catalog.columns:
        rows[column] = rows["drug_id"].map(catalog[column])
    for column in cell_cols:
        if column == "depmap_id":
            continue
        rows[column] = rows["depmap_id"].map(cells[column])

    rows["study"] = STUDY_PRISM
    rows["ln_ic50"] = np.nan
    rows["has_response"] = rows["auc"].notna()
    if "canonical_smiles" in rows.columns:
        rows["has_smiles"] = rows["canonical_smiles"].notna() & rows[
            "canonical_smiles"
        ].astype(str).str.len().gt(0)
    rows = rows.sort_values(["depmap_id", "drug_id"], kind="stable").reset_index(drop=True)
    rows.insert(0, "pair_id", [f"prism_{i:07d}" for i in range(len(rows))])

    for column in gdsc.columns:
        if column not in rows.columns:
            rows[column] = pd.NA
    return rows


def build_transfer_assignments(
    gdsc: pd.DataFrame,
    prism_rows: pd.DataFrame,
    random_pair: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stack GDSC + PRISM rows and label PRISM as the only test set.

    GDSC ``random_pair`` validation stays validation so early stopping cannot
    see PRISM. GDSC train and the unused GDSC test fold become training rows.
    """

    gdsc = gdsc.copy()
    gdsc["study"] = STUDY_GDSC
    gdsc["drug_id"] = gdsc["drug_id"].astype(str)
    gdsc["depmap_id"] = gdsc["depmap_id"].astype(str)
    validate_assignments(gdsc, random_pair)
    gdsc = gdsc.merge(random_pair, on="pair_id", how="inner", validate="one_to_one")
    gdsc.loc[gdsc["split"].eq("test"), "split"] = "train"

    prism_rows = prism_rows.copy()
    prism_rows["split"] = "test"
    prism_rows["drug_id"] = prism_rows["drug_id"].astype(str)
    prism_rows["depmap_id"] = prism_rows["depmap_id"].astype(str)

    combined = pd.concat(
        [gdsc, prism_rows],
        ignore_index=True,
        sort=False,
    )
    assignments = combined[["pair_id", "split"]].copy()
    cohort = combined.drop(columns=["split"])
    return cohort, assignments


def validate_transfer_split(cohort: pd.DataFrame, assignments: pd.DataFrame) -> None:
    """Fail if PRISM leaked into train/validation or GDSC into test."""

    validate_assignments(cohort, assignments)
    merged = cohort.merge(assignments, on="pair_id", how="inner", validate="one_to_one")
    if not np.isfinite(merged["auc"].to_numpy(float)).all():
        raise ValueError("Transfer AUC values must be finite.")
    if "study" not in merged.columns:
        raise ValueError("Transfer cohort is missing a study column.")
    train_val = merged.loc[merged["split"].isin(("train", "validation"))]
    test = merged.loc[merged["split"].eq("test")]
    if train_val["study"].ne(STUDY_GDSC).any():
        raise ValueError("PRISM rows leaked into GDSC train or validation.")
    if test["study"].ne(STUDY_PRISM).any():
        raise ValueError("GDSC rows leaked into the PRISM test set.")
    if not all(merged["split"].eq(label).any() for label in ("train", "validation", "test")):
        raise ValueError("Transfer split is missing train/validation or test rows.")
    if merged["split"].eq("unused").any():
        raise ValueError("Transfer split cannot contain unused rows.")


def build_transfer_cohort(
    gdsc_cohort: str | Path = GDSC_COHORT_PATH,
    prism_path: str | Path = PRISM_RESPONSE_PATH,
    random_pair_split: str | Path = RANDOM_PAIR_SPLIT_PATH,
    output_cohort: str | Path = TRANSFER_COHORT_PATH,
    output_split: str | Path = TRANSFER_SPLIT_PATH,
    drug_map: str | Path = DRUG_MAP_PATH,
    summary: str | Path = SUMMARY_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Write the transfer cohort, split assignments, and mapping audit."""

    gdsc = pd.read_csv(gdsc_cohort)
    random_pair = pd.read_csv(random_pair_split)
    prism_pairs, mapping = build_prism_pairs(prism_path, gdsc)
    prism_rows = format_prism_cohort_rows(prism_pairs, gdsc)
    cohort, assignments = build_transfer_assignments(gdsc, prism_rows, random_pair)
    validate_transfer_split(cohort, assignments)

    Path(output_cohort).parent.mkdir(parents=True, exist_ok=True)
    Path(output_split).parent.mkdir(parents=True, exist_ok=True)
    Path(drug_map).parent.mkdir(parents=True, exist_ok=True)
    Path(summary).parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output_cohort, index=False)
    assignments.to_csv(output_split, index=False)
    mapping.to_csv(drug_map, index=False)

    merged = cohort.merge(assignments, on="pair_id")
    test = merged.loc[merged["split"].eq("test")]
    train = merged.loc[merged["split"].eq("train")]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_name": TRANSFER_SPLIT,
        "target": "auc",
        "hypothesis_tests": "none",
        "note": (
            "Train and validation are GDSC2 only. Test is PRISM AUC for "
            "InChIKey-mapped drugs and overlapping DepMap cell lines. "
            "Assay shift is the question; overlapping drugs are allowed."
        ),
        "inputs": {
            "gdsc_cohort": str(gdsc_cohort).replace("\\", "/"),
            "prism_path": str(prism_path).replace("\\", "/"),
            "random_pair_split": str(random_pair_split).replace("\\", "/"),
        },
        "outputs": {
            "cohort": str(output_cohort).replace("\\", "/"),
            "split": str(output_split).replace("\\", "/"),
            "drug_map": str(drug_map).replace("\\", "/"),
        },
        "mapping": mapping["match_type"].value_counts().to_dict(),
        "counts": {
            "gdsc_rows": int(merged["study"].eq(STUDY_GDSC).sum()),
            "prism_rows": int(merged["study"].eq(STUDY_PRISM).sum()),
            "train_rows": int(merged["split"].eq("train").sum()),
            "validation_rows": int(merged["split"].eq("validation").sum()),
            "test_rows": int(len(test)),
            "test_drugs": int(test["drug_id"].nunique()),
            "test_cells": int(test["depmap_id"].nunique()),
            "drug_overlap_train_test": int(
                len(
                    set(train["drug_id"].astype(str))
                    & set(test["drug_id"].astype(str))
                )
            ),
            "cell_overlap_train_test": int(
                len(
                    set(train["depmap_id"].astype(str))
                    & set(test["depmap_id"].astype(str))
                )
            ),
        },
    }
    with Path(summary).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return cohort, assignments, report


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Build the GDSC2-train / PRISM-test transfer cohort."
    )
    parser.add_argument("--gdsc-cohort", default=GDSC_COHORT_PATH)
    parser.add_argument("--prism", default=PRISM_RESPONSE_PATH)
    parser.add_argument("--random-pair-split", default=RANDOM_PAIR_SPLIT_PATH)
    parser.add_argument("--output-cohort", default=TRANSFER_COHORT_PATH)
    parser.add_argument("--output-split", default=TRANSFER_SPLIT_PATH)
    parser.add_argument("--drug-map", default=DRUG_MAP_PATH)
    parser.add_argument("--summary", default=SUMMARY_PATH)
    return parser


def main() -> None:
    """Command-line entry point."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = build_parser().parse_args()
    _cohort, _assignments, report = build_transfer_cohort(
        gdsc_cohort=args.gdsc_cohort,
        prism_path=args.prism,
        random_pair_split=args.random_pair_split,
        output_cohort=args.output_cohort,
        output_split=args.output_split,
        drug_map=args.drug_map,
        summary=args.summary,
    )
    counts = report["counts"]
    print(f"Wrote transfer cohort to {report['outputs']['cohort']}")
    print(
        f"  train={counts['train_rows']}  val={counts['validation_rows']}  "
        f"test={counts['test_rows']} PRISM pairs, {counts['test_drugs']} drugs"
    )


if __name__ == "__main__":
    main()
