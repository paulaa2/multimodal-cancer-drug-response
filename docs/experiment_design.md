# Experiment Design

This project should be framed as a representation-learning study for cancer
drug-response prediction under distribution shift.

## Core question

Which molecular and transcriptomic representations generalize best when the
test set contains unseen biological contexts, unseen compounds, or unseen
chemical scaffolds?

## Evaluation protocol

The main comparison should report metrics for:

- `random_pair`: interpolation among known drugs and cell lines.
- `cold_cell`: new cell lines with known drugs.
- `cold_drug`: new drugs with known cell lines.
- `cold_scaffold`: new chemical scaffolds, the strongest chemistry split.
- `cold_both`: new drugs and new cell lines simultaneously.

All preprocessing must be fit inside each split using training rows only:
expression PCA, feature scaling, target scaling, and any feature selection.

## Model hierarchy

Use the following hierarchy to make conclusions easy to defend:

- B0: mean baselines.
- B1: linear Morgan + expression PCA.
- B2: XGBoost Morgan + expression PCA.
- B3: MLP Morgan + expression PCA.
- B4: lightweight GNN + expression PCA.
- B5: hybrid GNN + Morgan + expression PCA.

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
python -m src.mcdrp.experiments.run_experiment configs/experiments/b6_ablation_random_pair.json --dry-run
python -m src.mcdrp.experiments.run_experiment configs/experiments/b6_ablation_random_pair.json
```

B6 compares Morgan fingerprints, PCA expression, pathway scores, and their
combinations under the same XGBoost model family. It can use either the built-in
compact cancer pathway panel or an external GMT file such as MSigDB Hallmark.

## High-impact next upgrades

1. Add pathway-level transcriptomic features from MSigDB Hallmark, Reactome, or
   PROGENy.
2. Add pretrained molecular embeddings from a model such as ChemBERTa or
   MolFormer.
3. Report mean and standard deviation across multiple seeds.
4. Add uncertainty estimation with deep ensembles or conformal prediction.
5. Build figures for architecture, random-to-cold performance drop, ablations,
   and scaffold-level errors.

## Claim discipline

This is a preclinical cell-line modelling project. Claims should be phrased as
model generalization and representation-learning findings, not as clinical
precision-medicine validation.
