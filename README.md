# Robust Multimodal Cancer Drug Response Prediction

Predicting how cancer cell lines respond to drugs, from transcriptomic state and
molecular structure — and measuring honestly how much of that ability survives
when the model meets an unseen cell line, an unseen tissue, or an unseen
compound.

## Research question

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

`x_cell` is a transcriptomic vector or pathway activity profile, `G_drug` is a
molecular graph, Morgan fingerprint, or pretrained embedding. Regression, not
classification, because sensitive/resistant thresholds add assumptions.

## Status

Baselines through B7 are implemented and have been run across the split suite.
The headline comparison, on the `random_pair` test set:

| Model | Features | RMSE | vs. `mean_effects` |
| --- | --- | --- | --- |
| B0 `mean_effects` | none | 1.203 | reference |
| B5 hybrid GNN | graph + Morgan + expression PCA | **0.931** | −22.6% |

The `mean_effects` baseline reaches a global Pearson correlation of 0.906 using
no features at all, which is why it is the reference the models are measured
against. B5 beating it by 22.6% is evidence the modalities genuinely contribute.

Per split, best model against the strongest B0 baseline (test sets):

| Split | Best model | RMSE | Improvement |
| --- | --- | --- | --- |
| `random_pair` | B5 hybrid GNN | 0.931 | 22.6% |
| `cold_cell` | B2 tuned XGBoost | 1.299 | 12.6% |
| `cold_tissue` | *not yet run* | — | — |
| `cold_drug` | B2 tuned XGBoost | 2.218 | 16.4% |
| `cold_scaffold` | B5 hybrid GNN | 2.736 | 19.3% |
| `cold_both` | B7 pretrained + pathways | 2.375 | 10.7% |

**These are single-seed runs and carry no dispersion estimate, so small
differences between models are not yet interpretable.** The runner is
`python -m mcdrp.experiments.multiseed`; the experiment that counts is
`configs/experiments/multiseed_b0_b2_full.json`. Until those replications
exist, the ranking above is provisional. Note also the expected
pattern: the models hold up best where interpolation is possible and degrade
sharply on unseen chemistry.

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

Six splits, each answering one question. Bracketed names are the conventional
ones in the literature.

| Split | Held out | Question |
| --- | --- | --- |
| `random_pair` [LPO] | individual pairs | can it fill gaps in a measured matrix? |
| `cold_cell` [LCO] | whole cell lines | a new biological context? |
| `cold_tissue` [LTO] | whole tissues | a new cancer type? (repurposing) |
| `cold_drug` [LDO] | whole drugs | an unseen compound? |
| `cold_scaffold` | Bemis-Murcko scaffolds | structurally novel chemistry? |
| `cold_both` | drugs and cell lines | both novel at once? |

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
| B6 | modality ablation | PCA / pathways | Morgan |
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
Results here are single-seed and single-source (GDSC2).
