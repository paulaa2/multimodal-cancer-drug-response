# F0 Data Audit

Inputs, identifiers and response conventions must be explicit before modelling.
The audited local input bytes now have SHA-256 hashes in
`data/reports/data_audit_report.json`; a hash identifies a snapshot but does not
recover an unrecorded download release or validate biological annotations.

## What You Need To Download Manually

Put raw files in `data/raw/`. This directory is ignored by git because the files
can be large or redistributable only under source-specific terms.

Start with these files:

```text
data/raw/GDSC2_fitted_dose_response_27Oct23.xlsx
data/raw/GDSC2_fitted_dose_response_27Oct23.csv
data/raw/Model.csv
data/raw/OmicsExpressionTPMLogp1HumanProteinCodingGenesStranded.csv
data/raw/drug_structures.csv
data/raw/secondary-screen-dose-response-curve-parameters.csv
```

The first three files are enough to begin the audit. `drug_structures.csv` will
be created later from PubChem or ChEMBL after we inspect which GDSC drugs appear
in the response table.

`GDSC2_fitted_dose_response_27Oct23.csv` is a local conversion of the official
Excel file. Keeping the CSV copy makes repeated audits much faster.

The PRISM file is the secondary-screen dose-response table from the
[PRISM Repurposing 19Q4](https://doi.org/10.6084/m9.figshare.9393293.v4)
release (figshare file 20237739). It is required only for the GDSC2 → PRISM
transfer split. The SMILES column can repeat structures or list candidates
comma-separated. The mapper parses all distinct candidates and rejects lists
with multiple InChIKeys, then tries a unique full-key
match or a unique connectivity match. Duplicate GDSC keys are rejected;
connectivity matches remain a stereochemistry caveat.

```powershell
python -m mcdrp.data.build_transfer_cohort
```

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

## Default expression profiles and refreshed transfer

The local DepMap matrix contains 738 rows for the cohort's 717 models, including
21 alternative profiles. `load_expression_matrix` selects
`IsDefaultEntryForModel=Yes`, verifies one row per requested model and rejects
ambiguous/missing profiles. Earlier results weighted alternatives in PCA and
could overwrite a cell's feature vector by CSV order; those require reruns.

The strict transfer artifact is separate from the historical one:
`data/processed/transfer_checked_pairs.csv`,
`data/processed/transfer_checked/transfer_prism.csv`, and
`data/mappings/prism_gdsc_drug_map_strict.csv`. There are 44 full-key and two
unique connectivity matches, with 46 drugs and 14,200 external test pairs.
The strict builder retains PRISM mapping provenance columns in the cohort.

To re-audit frozen inputs without replacing mapping templates:

```powershell
python -m mcdrp.data.audit --no-write-templates
```

The shipped `configs/data_audit.example.json` specifies the actual input paths.
See `configs/experiments/transfer_prism_multiseed.json` for the refreshed transfer
pipeline. Published AUC remains unharmonized to a common dose range.
