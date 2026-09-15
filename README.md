# Robust Multimodal Cancer Drug Response Prediction

Independent computational biology portfolio project combining cancer cell-line
transcriptomics with molecular graph representations to predict continuous drug
response.

The project is intentionally built around realistic generalization: random
drug-cell-line splits are useful for debugging, but the core scientific question
is how much performance survives when the model sees unseen cell lines, unseen
drugs, or unseen chemical scaffolds.

## Research Question

Can a multimodal model that combines transcriptomic state and molecular graph
structure predict cancer drug response robustly under realistic distribution
shifts, and do biologically informed pathway representations improve
generalization and interpretability?

## Project Status

Current phase: **F0 - repository and data-audit scaffolding**.

The repository currently contains the project design document and the initial
implementation structure. The next milestone is to build a reproducible data
audit that integrates:

- GDSC response labels
- DepMap / CCLE transcriptomic profiles
- ChEMBL or PubChem chemical structures
- optional pathway resources such as MSigDB

No model results should be trusted until identifier mappings, data-audit
summaries, and leakage-safe splits are in place.

## Scientific Setup

Each training example is a measured response for one cancer cell line and one
drug:

```text
(cell line c, drug d) -> response y_cd
```

The model receives:

- a transcriptomic vector for the cell line, `x_cell`
- a molecular representation of the drug, `G_drug`
- a continuous response target, recommended first target: `ln(IC50)`

The prediction task is regression:

```text
y_hat = f_theta(x_cell, G_drug)
```

Classification into sensitive/resistant groups can be added later, but it is not
the primary task because response thresholds introduce extra assumptions.

## Planned Data Sources

| Resource | Role | Main object |
| --- | --- | --- |
| GDSC | Drug-response labels | drug-cell-line pair -> ln(IC50) / AUC |
| DepMap / CCLE | Transcriptomic profiles and cell-line metadata | cell line -> RNA-seq expression |
| ChEMBL or PubChem | Canonical chemical structures | drug -> canonical SMILES |
| MSigDB / pathway resource | Optional pathway representation | gene set -> pathway definition |

The project should use one frozen release of each resource. Mixing different
release versions inside the same experiment can silently create inconsistent
labels, features, or identifiers.

## Evaluation Splits

The evaluation design is the most important part of this project.

| Split | Held out | Question answered |
| --- | --- | --- |
| Random pair | drug-cell-line pairs | Can the model interpolate among known drugs and known cell lines? |
| Cold cell | entire cell lines | Can the model generalize to a new biological context? |
| Cold drug | entire drugs | Can the model generalize to unseen compounds? |
| Scaffold | chemical scaffolds | Can the model generalize to structurally novel chemistry? |
| Cold both | drugs and cell lines | Can it generalize when both modalities are novel? |

Feature selection, scaling, PCA, and pathway-derived transformations must be fit
only on training data for each split.

## Model Roadmap

The project should grow from simple, auditable baselines to multimodal deep
learning.

| ID | Model | Cell representation | Drug representation |
| --- | --- | --- | --- |
| B0 | Mean baseline | none | none |
| B1 | Ridge / Elastic Net | selected genes or PCA | Morgan fingerprint |
| B2 | XGBoost | selected genes or PCA | Morgan fingerprint |
| B3 | Multimodal MLP | gene, PCA, or pathway encoder | Morgan fingerprint |
| M1 | Multimodal GNN | gene encoder | molecular graph |
| M2 | Pathway + GNN | pathway encoder | molecular graph |

If Morgan fingerprints plus XGBoost match or beat the GNN, that is still a valid
scientific result. The goal is rigorous comparison, not forcing a preferred
model to win.

## Repository Structure

```text
multimodal-cancer-drug-response/
├── README.md
├── pyproject.toml
├── data/
│   ├── README.md
│   ├── reports/
│   └── mappings/
├── docs/
│   └── project_plan.md
├── src/
│   └── mcdrp/
│       ├── data/
│       ├── models/
│       ├── splits/
│       └── metrics.py
└── tests/
```

Reusable logic belongs in `src/mcdrp/`. Generated data, split files, and result
tables are written under ignored folders such as `data/processed/` and
`results/`.

## Setup

Create and activate a Python environment, then install the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Optional modelling dependencies are declared as extras in `pyproject.toml`.
Install `.[baselines]` for RDKit/XGBoost and `.[gpu]` for PyTorch-based GPU
training when the relevant phase needs them.

## First Implementation Milestones

1. **Data audit**
   - collect raw source files outside git
   - standardize cell-line, drug, and gene identifiers
   - create explicit mapping tables with status flags
   - write a short cohort summary

   Start here:

   ```powershell
   python -m mcdrp.data.audit
   ```

   This creates an audit report in `data/reports/` and mapping templates in
   `data/mappings/`. See `docs/data_audit.md` for the expected raw files and
   workflow.

   Once `drug_mapping_template.csv` exists, build the initial SMILES table:

   ```powershell
   python -m mcdrp.data.drug_structures
   ```

   Then build the first integrated cohort:

   ```powershell
   python -m mcdrp.data.build_cohort
   ```

2. **Leakage-safe splits**
   - random pair split
   - cold-cell split
   - cold-drug split
   - scaffold split after SMILES are available

   Start with the first three splits:

   ```powershell
   python -m mcdrp.splits.make_splits
   ```

3. **Classical baselines**
   - mean baseline
   - Ridge / Elastic Net with Morgan fingerprints
   - XGBoost with Morgan fingerprints
   - MLP neural baseline with the same engineered features

   Start with the B0 mean baselines:

   ```powershell
   python -m mcdrp.models.baseline_b0
   ```

   Run the first neural baseline on one split:

   ```powershell
   python -m mcdrp.models.baseline_b3 --splits random_pair --device cuda
   ```

   B3 is intentionally simple: it uses the same Morgan fingerprint + expression
   PCA features as B1/B2, but fits an MLP. This tells us whether a basic neural
   model helps before adding a graph molecular encoder.
   Use `--device auto` to use CUDA when available and CPU otherwise. B2 XGBoost
   supports the same device option; B1 Ridge/Elastic Net remains CPU-based.

   After running any baseline, consolidate every available result with the same
   command:

   ```powershell
   python -m mcdrp.results.compare_baselines
   ```

   Optional controlled hyperparameter tuning:

   ```powershell
   python -m mcdrp.models.tune_b1 --splits random_pair
   python -m mcdrp.models.tune_b2 --splits random_pair --device cuda
   python -m mcdrp.models.tune_b3 --splits random_pair --device cuda
   ```

   `compare_baselines` automatically includes tuning outputs when they exist and
   keeps only the validation-selected candidates for the final comparison.

   ```powershell
   python -m mcdrp.results.compare_baselines
   ```

   Before moving to GNN work, run the baseline-stage quality gate:

   ```powershell
   python -m mcdrp.results.validate_baseline_stage
   ```

   This checks cohort/schema integrity, split leakage, metric files, tuning
   selection rows, and consistency between the comparison CSVs and JSON summary.

4. **Multimodal neural model**
   - transcriptomic encoder
   - molecular graph encoder
   - fusion regression head

   Start with the lightweight pure-PyTorch GNN baseline:

   ```powershell
   python -m mcdrp.models.gnn_b4 --splits random_pair --device cuda
   python -m mcdrp.results.compare_baselines
   ```

   For a fast smoke run, reduce epochs:

   ```powershell
   python -m mcdrp.models.gnn_b4 --splits random_pair --device cuda --max-epochs 5
   ```

5. **Biological interpretation**
   - pathway representation
   - gene/pathway attribution
   - 2-3 case studies with known drug biology

## Leakage Controls

- Create train/validation/test groups before fitting any preprocessing step.
- Fit scalers, PCA, and feature selection only on training data.
- Keep all rows for a held-out drug out of training during cold-drug evaluation.
- Keep all rows for a held-out cell line out of training during cold-cell
  evaluation.
- Use scaffold grouping instead of naive random drug grouping for chemical
  generalization.
- Report unique drugs, unique cell lines, and total response pairs for every
  split.

## Final Portfolio Deliverables

- reproducible GitHub repository
- documented data-integration audit
- model comparison table across random and cold-start splits
- plot showing random-to-cold performance drop
- architecture figure and data-integration figure
- at least one biological interpretation figure or case study
- short report or release summary

## Limitations

This project should be presented as a preclinical cell-line modelling and
research-engineering project, not as a clinically validated precision-medicine
system. Cell-line drug response is an experimental proxy, and model attributions
should not be interpreted as causal biomarkers.
