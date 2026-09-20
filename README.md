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
On the `random_pair` test set, where a model may have seen both the cell line
and the drug before, the modalities clearly help:

| Model | Features | RMSE | vs. `mean_effects` |
| --- | --- | --- | --- |
| B0 `mean_effects` | none | 1.203 | reference |
| B5 hybrid GNN | graph + Morgan + expression PCA | **0.931** | −22.6% |

That 22.6% is the easiest setting in the suite, and it does not carry over.
The result below is the one that constrains what this project can claim.

### On unseen drugs, nothing beats predicting the cell-line mean

A single group-level partition suggested the models transferred to new
chemistry: on the seed-42 `cold_drug` split the best model reached 2.196 RMSE
against a 2.662 baseline, which reads as a 17% improvement. It does not
survive cross-validation. Testing every one of the 238 drugs across 5 folds
instead of the 36 that one partition happened to hold out, the same model
reaches 2.748 and **loses to the baseline**.

Because all models share the folds, the comparison that settles this is paired
within each fold, not overlapping error bars: the fold-to-fold spread of
±0.42 is mostly fold difficulty, which every model sees alike.

| Model on `cold_drug` | RMSE vs B0 `mean_effects` | Folds won |
| --- | --- | --- |
| B6 expression PCA | ±0.000 | 2 of 5 |
| B6 pathways | +0.002 | 0 of 5 |
| B6 Morgan + expression + pathways | +0.086 | 0 of 5 |
| B2 XGBoost | +0.091 | 1 of 5 |
| B6 Morgan | +0.131 | 0 of 5 |

`cold_scaffold` repeats the pattern against a 2.665 baseline. Two conclusions
follow. A model given only expression ties the baseline to three decimals,
because with an unseen drug the best it can do is predict the cell-line mean.
And **adding Morgan fingerprints makes every variant worse**, in both splits:
structural similarity does not transfer to new molecules on this cohort, it
adds noise. Reproduce with `python -m mcdrp.results.grouped_cv_report`.

### What is not yet settled

| Split | Status |
| --- | --- |
| `random_pair` | B5 hybrid GNN 0.931 against a 1.203 baseline, single seed |
| `cold_cell` | grouped CV in progress; single partition gave 1.299 |
| `cold_tissue` | grouped CV in progress; single partition gave 1.372 |
| `cold_drug` | settled above: no model beats the baseline |
| `cold_scaffold` | settled above: no model beats the baseline |
| `cold_both` | single partition only, 2.365 with B2 |

The `mean_effects` baseline reaches a global Pearson of 0.906 using no
features, which is why it is the reference rather than a naive mean. Splits
still marked single-partition carry no dispersion estimate, so differences
between models on them are not yet interpretable, and the `cold_drug` result
above is a direct warning about reading them as findings.

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
