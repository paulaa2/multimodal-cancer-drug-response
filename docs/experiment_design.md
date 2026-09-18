# Experiment Design

This project should be framed as a representation-learning study for cancer
drug-response prediction under distribution shift.

## Core question

Which molecular and transcriptomic representations generalize best when the
test set contains unseen biological contexts, unseen compounds, or unseen
chemical scaffolds?

## Evaluation protocol

Defined in [`evaluation_protocol.md`](evaluation_protocol.md), which is the
single source of truth for splits, metrics, baselines, leakage rules, and the
reporting bar. This document covers only the model and ablation design.

## Model hierarchy

Use the following hierarchy to make conclusions easy to defend:

- B0: mean baselines.
- B1: linear Morgan + expression PCA.
- B2: XGBoost Morgan + expression PCA.
- B3: MLP Morgan + expression PCA.
- B4: lightweight GNN + expression PCA.
- B5: hybrid GNN + Morgan + expression PCA.
- B6: modality ablations with Morgan, expression PCA, and pathway activity.
- B7: pretrained molecular embedding benchmark with biological cell features.

B4 answers whether a graph encoder helps over fingerprints. B5 answers whether
a stronger multimodal neural model can improve interpolation and cold-start
performance.

## Ablations to add next

The most valuable future ablations are:

- expression only
- Morgan only
- graph only
- Morgan + expression
- graph + expression
- graph + Morgan + expression
- pretrained molecular embedding + expression
- pretrained molecular embedding + pathway features

The goal is not only to find the best model; it is to explain which information
source is responsible for each gain.

The first implemented ablation layer is B6:

```powershell
python -m mcdrp.experiments.run_experiment configs/experiments/b6_ablation_random_pair.json --dry-run
python -m mcdrp.experiments.run_experiment configs/experiments/b6_ablation_random_pair.json
```

B6 compares Morgan fingerprints, PCA expression, pathway scores, and their
combinations under the same XGBoost model family. It can use either the built-in
compact cancer pathway panel or an external GMT file such as MSigDB Hallmark.

The next representation layer is B7:

```powershell
python -m mcdrp.experiments.run_experiment configs/experiments/b7_pretrained_random_pair.json --dry-run
python -m mcdrp.experiments.run_experiment configs/experiments/b7_pretrained_random_pair.json
```

B7 consumes frozen external molecular embeddings and compares them alone and in
combination with expression PCA and pathway activity. This directly tests
whether learned molecule representations add value over handcrafted Morgan
fingerprints and graph encoders in the same split protocol.

## High-impact next upgrades

Prioritized in [`research_roadmap.md`](research_roadmap.md). The short version:
per-row predictions are in place; multi-seed dispersion is implemented as
`mcdrp.experiments.multiseed` and still needs to be run before any new
modality, because without those numbers no comparison between models is
interpretable.

## Claim discipline

This is a preclinical cell-line modelling project. Claims should be phrased as
model generalization and representation-learning findings, not as clinical
precision-medicine validation. See
[`evaluation_protocol.md`](evaluation_protocol.md) section 6.
