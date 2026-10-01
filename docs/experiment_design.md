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
- B6: modality ablations with Morgan, expression PCA, and pathway mean expression.
- B7: pretrained molecular embedding benchmark with biological cell features.

B4 answers whether a graph encoder helps over fingerprints. B5 answers whether
a stronger multimodal neural model can improve interpolation and cold-start
performance.

## Implemented representation controls

The implemented model/ablation modules cover:

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

The pretrained representation layer is B7:

```powershell
python -m mcdrp.experiments.run_experiment configs/experiments/b7_pretrained_random_pair.json --dry-run
python -m mcdrp.experiments.run_experiment configs/experiments/b7_pretrained_random_pair.json
```

B7 consumes frozen external molecular embeddings and compares them alone and in
combination with expression PCA and pathway mean expression. This directly tests
whether learned molecule representations add value over handcrafted Morgan
fingerprints and graph encoders in the same split protocol.

## High-impact next upgrades

The [research roadmap](research_roadmap.md) is authoritative for completed and
pending experiments. The 2026-09-30 audit corrected preprocessing and fitting;
[repository audit](repository_audit.md) identifies which historical model
comparisons need refreshing. Assay-endpoint harmonization now precedes new
encoders. The shared-window endpoint, entity nulls, graph-only control, RF,
annotation views and independent calibration are described in
[roadmap 2–6](roadmap_2_6.md). Built-in pathway scores are unweighted expression summaries, not
validated signaling activity estimates.

## Claim discipline

This is a preclinical cell-line modelling project. Claims should be phrased as
model generalization and representation-learning findings, not as clinical
precision-medicine validation. See
[`evaluation_protocol.md`](evaluation_protocol.md) section 6.
