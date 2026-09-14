"""Build the first integrated modelling cohort.

The cohort table keeps one row per GDSC drug-cell-line response pair and adds
the stable DepMap model identifier plus drug SMILES. It deliberately does not
join the full expression matrix into every response row; that would create a
large duplicated table and make split/debugging harder.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


GDSC_COLUMNS = [
    "DATASET",
    "CELL_LINE_NAME",
    "SANGER_MODEL_ID",
    "CANCER_TYPE",
    "DRUG_ID",
    "DRUG_NAME",
    "PUTATIVE_TARGET",
    "PATHWAY_NAME",
    "LN_IC50",
    "AUC",
]

METADATA_COLUMNS = [
    "ModelID",
    "SangerModelID",
    "CellLineName",
    "StrippedCellLineName",
    "OncotreeLineage",
    "OncotreePrimaryDisease",
    "OncotreeSubtype",
    "COSMICID",
]

STRUCTURE_COLUMNS = [
    "drug_id",
    "drug_name",
    "canonical_smiles",
    "isomeric_smiles",
    "inchi_key",
    "pubchem_cid",
    "mapping_status",
    "source",
]


def load_expression_ids(path: str | Path) -> set[str]:
    """Load DepMap model IDs present in the expression matrix."""

    ids = pd.read_csv(path, usecols=["ModelID"])["ModelID"].dropna().astype(str)
    return set(ids)


def load_metadata(path: str | Path, expression_ids: set[str]) -> pd.DataFrame:
    """Load DepMap model metadata and keep one row per Sanger model ID."""

    metadata = pd.read_csv(path, usecols=METADATA_COLUMNS)
    metadata = metadata.rename(
        columns={
            "ModelID": "depmap_id",
            "SangerModelID": "sanger_model_id",
            "CellLineName": "depmap_cell_line_name",
            "StrippedCellLineName": "depmap_stripped_cell_line_name",
            "OncotreeLineage": "oncotree_lineage",
            "OncotreePrimaryDisease": "oncotree_primary_disease",
            "OncotreeSubtype": "oncotree_subtype",
            "COSMICID": "cosmic_id",
        }
    )
    metadata = metadata.loc[metadata["sanger_model_id"].notna()].copy()
    metadata["depmap_id"] = metadata["depmap_id"].astype(str)
    metadata["sanger_model_id"] = metadata["sanger_model_id"].astype(str)
    metadata["has_expression"] = metadata["depmap_id"].isin(expression_ids)

    # Prefer the model with expression if a Sanger ID appears more than once.
    metadata = metadata.sort_values(
        ["sanger_model_id", "has_expression", "depmap_id"],
        ascending=[True, False, True],
    )
    return metadata.drop_duplicates("sanger_model_id", keep="first")


def load_drug_structures(path: str | Path) -> pd.DataFrame:
    """Load PubChem-derived drug structures."""

    structures = pd.read_csv(path, usecols=STRUCTURE_COLUMNS)
    structures = structures.rename(
        columns={
            "drug_id": "drug_id",
            "drug_name": "structure_drug_name",
            "source": "structure_source",
        }
    )
    structures["drug_id"] = structures["drug_id"].astype(str)
    structures["has_smiles"] = (
        structures["canonical_smiles"].notna()
        & structures["canonical_smiles"].astype(str).str.len().gt(0)
        & structures["mapping_status"].eq("matched")
    )
    return structures.drop_duplicates("drug_id", keep="first")


def load_gdsc(path: str | Path) -> pd.DataFrame:
    """Load the GDSC response table."""

    gdsc = pd.read_csv(path, usecols=GDSC_COLUMNS)
    gdsc = gdsc.rename(
        columns={
            "CELL_LINE_NAME": "gdsc_cell_line_name",
            "SANGER_MODEL_ID": "sanger_model_id",
            "CANCER_TYPE": "gdsc_cancer_type",
            "DRUG_ID": "drug_id",
            "DRUG_NAME": "gdsc_drug_name",
            "PUTATIVE_TARGET": "putative_target",
            "PATHWAY_NAME": "gdsc_pathway_name",
            "LN_IC50": "ln_ic50",
            "AUC": "auc",
            "DATASET": "gdsc_dataset",
        }
    )
    gdsc["sanger_model_id"] = gdsc["sanger_model_id"].astype(str)
    gdsc["drug_id"] = gdsc["drug_id"].astype(str)
    return gdsc


def build_cohort(
    gdsc_response: str | Path = "data/raw/GDSC2_fitted_dose_response_27Oct23.csv",
    depmap_metadata: str | Path = "data/raw/Model.csv",
    depmap_expression: str | Path = "data/raw/OmicsExpressionTPMLogp1HumanProteinCodingGenesStranded.csv",
    drug_structures: str | Path = "data/raw/drug_structures.csv",
    output: str | Path = "data/processed/cohort_pairs.csv",
    summary: str | Path = "data/reports/cohort_summary.json",
    *,
    require_expression: bool = True,
    require_smiles: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build and write the integrated cohort table."""

    inputs = {
        "gdsc_response": str(gdsc_response),
        "depmap_metadata": str(depmap_metadata),
        "depmap_expression": str(depmap_expression),
        "drug_structures": str(drug_structures),
    }
    outputs = {
        "cohort_pairs": str(output),
        "summary": str(summary),
    }
    filters = {
        "require_expression": require_expression,
        "require_smiles": require_smiles,
        "response_target": "LN_IC50",
    }

    expression_ids = load_expression_ids(depmap_expression)
    metadata = load_metadata(depmap_metadata, expression_ids)
    structures = load_drug_structures(drug_structures)
    gdsc = load_gdsc(gdsc_response)

    cohort = gdsc.merge(metadata, on="sanger_model_id", how="left")
    cohort = cohort.merge(structures, on="drug_id", how="left")

    cohort["has_response"] = cohort["ln_ic50"].notna()
    cohort["is_model_ready"] = cohort["has_response"]
    if require_expression:
        cohort["is_model_ready"] &= cohort["has_expression"].fillna(False)
    if require_smiles:
        cohort["is_model_ready"] &= cohort["has_smiles"].fillna(False)

    model_ready = cohort.loc[cohort["is_model_ready"]].copy()
    model_ready = model_ready.sort_values(
        ["depmap_id", "drug_id", "ln_ic50"], kind="stable"
    ).reset_index(drop=True)
    model_ready.insert(0, "pair_id", [f"pair_{i:07d}" for i in range(len(model_ready))])

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_ready.to_csv(output_path, index=False)

    summary_data = make_summary(inputs, outputs, filters, cohort, model_ready, expression_ids)
    summary_path = Path(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_data, handle, indent=2)

    return model_ready, summary_data


def make_summary(
    inputs: dict[str, str],
    outputs: dict[str, str],
    filters: dict[str, Any],
    full_cohort: pd.DataFrame,
    model_ready: pd.DataFrame,
    expression_ids: set[str],
) -> dict[str, Any]:
    """Summarize the integrated cohort."""

    missing_expression = full_cohort["has_expression"].fillna(False).eq(False)
    missing_smiles = full_cohort["has_smiles"].fillna(False).eq(False)
    missing_response = full_cohort["has_response"].eq(False)

    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "inputs": inputs,
        "outputs": outputs,
        "filters": filters,
        "counts": {
            "gdsc_response_rows": int(len(full_cohort)),
            "depmap_expression_models": int(len(expression_ids)),
            "model_ready_rows": int(len(model_ready)),
            "model_ready_cell_lines": int(model_ready["depmap_id"].nunique()),
            "model_ready_sanger_cell_lines": int(
                model_ready["sanger_model_id"].nunique()
            ),
            "model_ready_drugs": int(model_ready["drug_id"].nunique()),
            "rows_missing_expression": int(missing_expression.sum()),
            "rows_missing_smiles": int(missing_smiles.sum()),
            "rows_missing_response": int(missing_response.sum()),
        },
        "columns": list(model_ready.columns),
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description="Build the integrated cohort table.")
    parser.add_argument(
        "--gdsc-response",
        default="data/raw/GDSC2_fitted_dose_response_27Oct23.csv",
        help="Path to the GDSC response table.",
    )
    parser.add_argument(
        "--depmap-metadata",
        default="data/raw/Model.csv",
        help="Path to DepMap model metadata.",
    )
    parser.add_argument(
        "--depmap-expression",
        default="data/raw/OmicsExpressionTPMLogp1HumanProteinCodingGenesStranded.csv",
        help="Path to DepMap expression matrix.",
    )
    parser.add_argument(
        "--drug-structures",
        default="data/raw/drug_structures.csv",
        help="Path to the drug structures table.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/cohort_pairs.csv",
        help="Output cohort CSV.",
    )
    parser.add_argument(
        "--summary",
        default="data/reports/cohort_summary.json",
        help="Output JSON summary.",
    )
    parser.add_argument(
        "--allow-missing-expression",
        action="store_true",
        help="Do not require expression availability for model-ready rows.",
    )
    parser.add_argument(
        "--allow-missing-smiles",
        action="store_true",
        help="Do not require SMILES availability for model-ready rows.",
    )
    return parser


def main() -> None:
    """Command-line entry point."""

    args = build_parser().parse_args()
    cohort, summary = build_cohort(
        gdsc_response=args.gdsc_response,
        depmap_metadata=args.depmap_metadata,
        depmap_expression=args.depmap_expression,
        drug_structures=args.drug_structures,
        output=args.output,
        summary=args.summary,
        require_expression=not args.allow_missing_expression,
        require_smiles=not args.allow_missing_smiles,
    )
    counts = summary["counts"]
    print(f"Wrote {len(cohort)} model-ready rows")
    print(f"Unique cell lines: {counts['model_ready_sanger_cell_lines']}")
    print(f"Unique drugs: {counts['model_ready_drugs']}")
    print(f"Cohort path: {summary['outputs']['cohort_pairs']}")
    print(f"Summary path: {summary['outputs']['summary']}")


if __name__ == "__main__":
    main()
