# Robust Multimodal Cancer Drug Response Prediction

Predicting how cancer cell lines respond to drugs, from transcriptomic state and
molecular structure — and measuring honestly how much of that ability survives
when the model meets an unseen cell line, an unseen tissue, or an unseen
compound.

## Research question

The current reproducible release is [stable v1](docs/stable_v1.md): preprocessing,
training, test and results for B0/RF/B6/B7, five chemically disjoint cold-drug folds
and GDSC→PRISM. Chemical aliases stay together (238 IDs, 229 identities).
Start with the [stable notebook](notebooks/stable_v1_results.ipynb). Other
architectures and older runs remain experimental. The stable guide contains
the isolated reproduction command and local release manifests.

The [bounded improvement follow-up](docs/improvement_v1.md) compares a
validation-selected ensemble, small residual neural towers and explicitly
supervised PRISM adaptation. It preserves stable v1 as its reference and keeps
source-only transfer separate from experiments using destination-domain labels.

> Can a multimodal model combining transcriptomic state and molecular structure
> predict cancer drug response robustly under realistic distribution shifts, and
> do biologically informed pathway representations improve generalization and
> interpretability?

The evaluation design is the point of this project, not the architecture. Recent
critical reviews of this field find that about half of published models do not
significantly outperform a baseline that only knows the average response of each
drug and each cell line, and that confirmed data leakage affects roughly 70% of
audited methods. So the project is built around a strong feature-free reference,
leakage-safe splits, and metrics that separate real signal from mean-effect
memorization. See [`docs/related_work.md`](docs/related_work.md).

## Task

Each example is one measured response for a (cell line, drug) pair:

```text
y_hat = f(x_cell, G_drug)     target: ln(IC50)
```

`x_cell` is a transcriptomic vector or pathway expression profile, `G_drug` is a
molecular graph, Morgan fingerprint, or pretrained embedding. Regression, not
classification, because sensitive/resistant thresholds add assumptions.

## Status

The model ladder B0–B7 and the six GDSC splits are implemented. The
2026-09-30 audit found issues that affect reproducibility and interpretation:
expression contained 21 alternative profiles alongside the 717 default cell
profiles; GNN initialization was seeded after constructing the model; B3 used
pair-based internal early stopping in cold settings; and PRISM matching chose
arbitrarily among duplicate chemical identities. These are corrected. Historical
model numbers using the old preprocessing require reruns before confirmation.
See [`docs/repository_audit.md`](docs/repository_audit.md) for scope and evidence.

The earlier exploratory audit checkpoint preserves these controls:

- `configs/experiments/cold_drug_checked.json`: five existing cold-drug folds,
  default expression profiles, target + PCA and Morgan + PCA against fold-fitted
  mean effects, with fixed hyperparameters and no cross-partition tuning file.
- `configs/experiments/transfer_prism_multiseed.json`: strict chemical matching,
  default expression profiles, and three fitting/permutation seeds (42, 7, 13).
  Test has **46 drugs, 335 cells, and 14,200 pairs**; GDSC remains the sole
  training/validation study. This is assay transfer, with known cells and drugs.
- `mcdrp.results.transfer_diagnostics`: within-drug assay concordance and
  leave-drug-out error stratified by each fold's training annotation vocabulary.

Global scores, ranking, and calibration answer different questions. Negative
within-drug R² does **not** imply absent ranking: refreshed PRISM target + PCA
predictions have mean within-drug Spearman 0.2115 despite R² −0.2026.
The original transfer stress test uses published GDSC/PRISM AUC. A separate
cohort now recomputes viable-fraction AUC over one shared concentration window;
published and harmonized endpoints are compared on identical retained pairs.
These are exploratory preclinical results, with `mean_effects` the prespecified
reference. Findings and next actions belong in
[`docs/research_roadmap.md`](docs/research_roadmap.md).

The roadmap 2–6 protocols, frozen data sources, Random Forest, modality controls,
annotation/embedding provenance and independent calibration are documented in
[`docs/roadmap_2_6.md`](docs/roadmap_2_6.md). Keep these refreshed outputs separate
from historical architecture comparisons.

The earlier exploratory [roadmap-results notebook](notebooks/final_results.ipynb) presents
16 publication-quality figures covering all refreshed variants, B0, modality
controls, assay transfer and calibration. It runs from compact versioned
reports without raw data or a GPU. See [notebook instructions](notebooks/README.md)
for reproduction and PNG/SVG/PDF/HTML exports.

## Quickstart

```powershell
conda create -n mcdrp python=3.12
conda activate mcdrp
pip install -e ".[dev,baselines]"
pytest
```

The package uses a `src/` layout and must be installed to import as `mcdrp`.
Always invoke modules as `python -m mcdrp.<module>`; see
[`docs/workflow.md`](docs/workflow.md) for why the `src.mcdrp.` form is a trap.

Raw data is not in the repository. Follow
[`docs/data_audit.md`](docs/data_audit.md) to place the GDSC, DepMap, and
structure files, then:

```powershell
python -m mcdrp.data.audit             # identifier mappings and audit report
python -m mcdrp.data.drug_structures   # canonical SMILES
python -m mcdrp.data.build_cohort      # model-ready pair table
python -m mcdrp.splits.make_splits     # the six leakage-checked splits
python -m mcdrp.data.build_transfer_cohort  # GDSC2-train / PRISM-test, target auc
python -m mcdrp.models.baseline_b0     # feature-free reference
python -m mcdrp.results.compare_baselines
```

Larger stages run from versioned configs, so an experiment is reproducible from
one file:

```powershell
python -m mcdrp.experiments.run_experiment configs/experiments/b5_hybrid_full.json --dry-run
python -m mcdrp.experiments.run_experiment configs/experiments/b5_hybrid_full.json
```

`--dry-run` prints the commands without executing them. Every run writes a
manifest to `results/experiments/<name>/manifest.json`.

## Evaluation design

Six grouped splits plus one cross-study transfer split, each answering one
question. Bracketed names are the conventional ones in the literature.

| Split | Held out | Question |
| --- | --- | --- |
| `random_pair` [LPO] | individual pairs | can it fill gaps in a measured matrix? |
| `cold_cell` [LCO] | whole cell lines | a new biological context? |
| `cold_tissue` [LTO] | whole tissues | a new cancer type? (repurposing) |
| `cold_drug` [LDO] | whole drugs | an unseen compound? |
| `cold_scaffold` | Bemis-Murcko scaffolds | structurally novel chemistry? |
| `cold_both` | drugs and cell lines | both novel at once? |
| `transfer_prism` | assay (GDSC2 → PRISM AUC) | do GDSC-fitted models transfer across screens? |

Metrics come in three families, because global RMSE and Pearson are dominated by
differences in mean potency between drugs: global, mean-effect **normalized**,
and **stratified** per drug and per cell line. The full rationale, the leakage
rules, and the reporting requirements are in
[`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) — read this before
interpreting any number in this repository.

## Model ladder

Each rung isolates one question, and a simple model winning is a valid result.

| ID | Model | Cell features | Drug features |
| --- | --- | --- | --- |
| B0 | mean baselines | none | none |
| B1 | Ridge / ElasticNet | genes or PCA | Morgan |
| B2 | XGBoost | genes or PCA | Morgan |
| B3 | MLP | genes, PCA, or pathways | Morgan |
| B4 | lightweight GNN | expression PCA | graph |
| B5 | hybrid GNN | expression PCA | graph + Morgan |
| B6 | modality ablation and controls | PCA / pathways / matched genes | Morgan / annotated or Ridge-predicted targets |
| B7 | pretrained-drug benchmark | PCA / pathways | frozen embedding |

## Repository layout

```text
├── configs/experiments/   versioned experiment definitions
├── data/
│   ├── reports/           small JSON run summaries (tracked)
│   └── mappings/          identifier mapping tables (tracked)
├── docs/                  design, protocol, and literature
├── src/mcdrp/
│   ├── data/              audit, structures, cohort assembly
│   ├── features/          fingerprints, expression, pathways, graphs, embeddings
│   ├── splits/            leakage-checked split construction
│   ├── models/            B0-B7 and tuning drivers
│   ├── results/           comparison, cold-start, and ablation reporting
│   ├── experiments/       config-driven pipeline runner
│   └── metrics.py         metrics and the mean-effects reference
└── tests/
```

Generated data and result tables are written to gitignored folders
(`data/raw/`, `data/processed/`, `results/`, `figures/`).

## Documentation

| Document | Contents |
| --- | --- |
| [`evaluation_protocol.md`](docs/evaluation_protocol.md) | splits, metrics, leakage rules, reporting bar |
| [`related_work.md`](docs/related_work.md) | literature and the reasoning behind the design |
| [`research_roadmap.md`](docs/research_roadmap.md) | prioritized next steps |
| [`repository_audit.md`](docs/repository_audit.md) | corrections, refreshed results, verification and remaining limits |
| [`workflow.md`](docs/workflow.md) | environment, quality gates, git conventions |
| [`project_plan.md`](docs/project_plan.md) | phases F0-F7 with commands |
| [`experiment_design.md`](docs/experiment_design.md) | model ladder and ablation rationale |
| [`data_audit.md`](docs/data_audit.md) | expected raw files and audit workflow |
| [`external_embeddings.md`](docs/external_embeddings.md) | B7 embedding format and generation |

## Limitations

A preclinical cell-line modelling and research-engineering project, not a
clinically validated precision-medicine system. Cell-line drug response is an
experimental proxy, model attributions are not causal biomarkers, and the
literature shows transfer from cell lines to patient data is currently poor.
Grouped folds share training data; their spread is descriptive. Transfer fitting
seeds reuse one partition and external cohort, so they do not estimate uncertainty
over independent screens. Historical B3/B4/B5 comparisons need refreshed training.
