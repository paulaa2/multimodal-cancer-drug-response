# Related Work

Why this project is designed the way it is, and which published findings each
design decision responds to. The emphasis is deliberately on the evaluation
literature rather than on architectures, because the evidence is that evaluation
choices, not architectures, explain most reported progress in this field.

## Benchmarks and critical evaluations

**Critical evaluation of drug response prediction models with DrEval.**
*Nature Communications*, 2026. <https://www.nature.com/articles/s41467-026-72903-w>
(preprint: <https://doi.org/10.1101/2025.05.26.655288>, code:
<https://github.com/daisybio/drevalpy>)

The most directly relevant reference. It identifies six recurring obstacles:
non-reproducible models, data leakage, pseudoreplication, biased evaluation
metrics, missing ablation studies, and inconsistent viability data. Its headline
results:

- About half of tested models do not significantly outperform a naive predictor
  using only drug and cell-line mean effects.
- In the leave-cell-line-out setting, no model surpasses a properly tuned random
  forest.
- All models explain under 20% of drug-sensitivity variance for unseen cell
  lines (normalized R2 < 0.20), dropping below 0.08 for unseen tissues.
- All tested models fail entirely for unseen drugs (R2 ~ 0).
- Most variance in drug response comes from drug identity, so a model that
  memorizes per-drug means explains most of the variance while learning nothing
  generalizable. The paper frames this as a form of Simpson's paradox.

What this project takes from it: the `mean_effects` baseline, the normalized and
per-drug/per-cell stratified metrics, the LPO/LCO/LTO/LDO split vocabulary, the
`cold_tissue` split, and the requirement that baselines be tuned as carefully as
models.

**The specification game: rethinking the evaluation of drug response prediction
for precision oncology.** *Journal of Cheminformatics*, 2025.
<https://doi.org/10.1186/s13321-025-00972-y>

Argues that the splitting strategy must match the intended application, and
documents how inconsistently this is done in practice. Establishes the ordering
of difficulty used in [`evaluation_protocol.md`](evaluation_protocol.md):
random pairs, then unseen cell lines, then unseen drugs, then both.

**Widespread data leakage inflates accuracy and corrupts biomarker discovery in
cancer drug response prediction.** bioRxiv, 2026.
<https://www.biorxiv.org/content/10.64898/2026.02.05.704016v2.full-text>
(code: <https://github.com/AsiaeeLab/drug-response-leakage>)

Quantifies one leakage mode — supervised feature screening on all samples before
cross-validation — across 265 drugs and 1,462 cell lines. Leakage-free CV raises
mean squared error by 16.6% on average, and 83% of drugs show inflated
performance under the leaked pipeline. A code-level audit finds confirmed
leakage in 23 of 32 published methods. Critically, the inflation is comparable
in size to the improvements those methods typically claim over elastic-net
baselines.

Its five-mode leakage taxonomy is the checklist used in
[`evaluation_protocol.md`](evaluation_protocol.md) section 2:

1. preprocessing fit on all samples before splitting
2. test data used for early stopping or model selection
3. pair-level splits inconsistent with cold-start claims
4. target-domain adaptation using test samples
5. post-hoc selection of the best test metric

The paper also reports that leakage inflates "stable" biomarker counts roughly
eightfold (18.1 vs 2.2 features per drug) without improving recovery of known
drug targets — a caution that applies directly to the interpretation phase
planned here.

## Multimodal architectures

The architecture literature is much less conclusive than it appears, which is
why this repository treats fingerprint baselines as serious competitors rather
than as strawmen.

**DTLCDR.** 2025. <https://pubmed.ncbi.nlm.nih.gov/41050110/> — Combines
chemical descriptors, molecular graphs, predicted protein target profiles, and
expression with a pretrained single-cell language model. Its ablations single
out *predicted target information* as the largest contributor to generalization
for unseen drugs. This is the strongest available argument for adding a
target-profile modality.

**BioGDR.** *npj Digital Medicine*, 2026.
<https://www.nature.com/articles/s41746-026-02735-x> — Finds that differential
expression features contribute most, and notably that removing *structural*
features degrades performance less than removing the biological features. A
direct warning against assuming the graph encoder is the important part.

**XMR.** *Frontiers in Bioinformatics*, 2023.
<https://doi.org/10.3389/fbinf.2023.1164482> — Reports that a larger GNN
chemical encoder keeps improving performance while a larger biologically
structured genomic encoder overfits. Useful for capacity allocation between the
two towers.

Set against DrEval's ablation finding that additional modalities and more
advanced representations changed little for the model they examined, and that
randomizing molecular fingerprints only mattered in the unseen-drug setting, the
reasonable prior is: expression carries most of the signal for `random_pair` and
`cold_cell`, and drug representation only becomes decisive for `cold_drug` and
`cold_scaffold`. The B6 and B7 ablations in this repository are designed to test
exactly that.

## Data sources

| Resource | Role |
| --- | --- |
| GDSC2 | drug response labels, `ln(IC50)` |
| DepMap / CCLE | transcriptomics and cell-line lineage metadata |
| PubChem / ChEMBL | canonical SMILES |
| MSigDB / Reactome / PROGENy | pathway gene sets |

One frozen release per resource, per the audit in
[`data_audit.md`](data_audit.md). DrEval notes that technical differences
between screens are substantial enough that naively pooling datasets is
counterproductive, which is why cross-study validation is scoped as a separate
milestone rather than as extra training data.

## Deliberate differences from DrEval

This repository is not a DrEval reimplementation, and where it diverges the
reason is scope rather than disagreement:

- **Single train/validation/test split per setting**, not k-fold nested CV.
  Cheaper, but it means no dispersion estimates, which is the main outstanding
  weakness.
- **`cold_scaffold` split**, which DrEval does not include. Bemis-Murcko
  scaffold grouping is a stricter chemical generalization test than holding out
  drug IDs.
- **Graph encoders written in plain PyTorch** rather than PyTorch Geometric, to
  keep the dependency surface small.

Using `drevalpy` directly as an external cross-check would be a strong addition
and is listed in [`research_roadmap.md`](research_roadmap.md).
