# F0 Data Audit

The next implementation step is not model training. It is building a clean,
inspectable cohort from several biological and chemical resources.

## What You Need To Download Manually

Put raw files in `data/raw/`. This directory is ignored by git because the files
can be large or redistributable only under source-specific terms.

Start with four files:

```text
data/raw/GDSC2_fitted_dose_response_27Oct23.xlsx
data/raw/GDSC2_fitted_dose_response_27Oct23.csv
data/raw/Model.csv
data/raw/OmicsExpressionTPMLogp1HumanProteinCodingGenesStranded.csv
data/raw/drug_structures.csv
```

The first three files are enough to begin the audit. `drug_structures.csv` will
be created later from PubChem or ChEMBL after we inspect which GDSC drugs appear
in the response table.

`GDSC2_fitted_dose_response_27Oct23.csv` is a local conversion of the official
Excel file. Keeping the CSV copy makes repeated audits much faster.

## Build Drug Structures

After `drug_mapping_template.csv` exists, create the first SMILES table with:

```powershell
python -m mcdrp.data.drug_structures
```

This queries PubChem by GDSC drug name, writes
`data/raw/drug_structures.csv`, and caches lookup responses under
`data/interim/pubchem_smiles_cache.jsonl`.

Rows marked as `not_found`, `error`, or `missing_smiles` should be reviewed
manually before using molecular fingerprints or graph models.

## First Audit Command

After installing the project, run:

```powershell
python -m mcdrp.data.audit
```

The command writes:

```text
data/reports/data_audit_report.json
data/mappings/cell_line_mapping_template.csv
data/mappings/drug_mapping_template.csv
```

At first, the report may simply tell you which files are missing. That is useful:
it makes the next action explicit.

## What The Audit Checks

- whether each expected raw file exists
- table shape: rows and columns
- whether key identifier columns can be detected
- candidate cell-line mappings
- candidate drug-to-SMILES mappings

## Why This Matters

Drug-response projects are easy to break silently. A model can appear strong if
cell-line or drug identifiers are mismatched, duplicated, or leaked across
splits. The audit phase keeps those decisions visible before any training code
exists.
