# Project Plan

This plan translates the design document into implementation phases that can be
worked through incrementally.

## F0 - Biology and Data Audit

Goal: create a clean, frozen cohort table before modelling.

Outputs:

- source data summary with exact file names, URLs, versions, and checksums
- GDSC response table standardized to one target, initially `ln(IC50)`
- cell-line mapping table from GDSC names/IDs to DepMap IDs
- drug mapping table from GDSC compound IDs/names to canonical SMILES
- overlap report with included/excluded cell lines, drugs, and response pairs

Key rule: no model training starts until the cohort and mappings are explicit.

After the audit and SMILES lookup, build the first model-ready cohort:

```powershell
python -m mcdrp.data.build_cohort
```

This writes `data/processed/cohort_pairs.csv`, a lightweight table of valid
GDSC drug-cell-line pairs with DepMap IDs and canonical SMILES. The expression
matrix remains separate and is linked by `depmap_id`.

## F1 - Classical Baselines

Goal: measure how far simple models go.

Outputs:

- leakage-safe random-pair and cold-drug splits
- Morgan fingerprint features
- selected-gene or PCA cell-line features
- Ridge / Elastic Net baseline
- XGBoost baseline
- first metrics table

Before modelling, generate the initial split assignment files:

```powershell
python -m mcdrp.splits.make_splits
```

This writes one CSV per split under `data/processed/splits/` and a summary at
`data/reports/splits_summary.json`. The cold-cell split checks that held-out
cell lines do not appear in train. The cold-drug split checks that held-out drugs
do not appear in train.

## F2 - Multimodal GNN

Goal: train a reproducible neural model that combines transcriptomics and drug
graphs.

Outputs:

- RDKit molecular graph conversion
- cell-line MLP encoder
- GIN/GINE drug encoder
- concatenation fusion head
- training loop with early stopping
- matched comparison against fingerprint MLP baseline

## F3 - Cold-Start Evaluation

Goal: make the generalization story scientifically meaningful.

Outputs:

- random-pair split
- cold-cell split
- cold-drug split
- scaffold split
- table reporting metrics and unique drugs/cell lines/pairs per split
- plot showing random-to-cold performance degradation

## F4 - Pathways and Ablations

Goal: test whether pathway-level representations improve robustness and
interpretability.

Outputs:

- pathway activity matrix
- comparison of selected genes vs PCA vs pathway scores
- ablation table under identical splits

## F5 - Interpretation and Polish

Goal: turn the implementation into a portfolio-ready scientific artifact.

Outputs:

- 2-3 biological case studies
- gene/pathway attribution analysis
- optional drug substructure explanations
- architecture and data-integration figures
- final README results section
- short report or GitHub release summary
