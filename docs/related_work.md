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
target-profile modality. On this cohort the claim splits: GDSC `putative_target`
annotations transfer under leave-drug-out grouped CV, but a leakage-safe map
from Morgan fingerprints onto that same vocabulary does not recover the gain.

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
| GDSC2 | drug response labels, `ln(IC50)` and AUC |
| PRISM Repurposing 19Q4 | secondary-screen AUC for the `transfer_prism` split |
| DepMap / CCLE | transcriptomics and cell-line lineage metadata |
| PubChem / ChEMBL | canonical SMILES |
| MSigDB / Reactome | pathway gene sets for mean-expression features |
| PROGENy | weighted perturbation footprints for a future activity scorer |

One frozen release per resource, per the audit in
[`data_audit.md`](data_audit.md). DrEval notes that technical differences
between screens are substantial enough that naively pooling datasets is
counterproductive, which is why `transfer_prism` is a held-out test rather
than extra training data. Refreshed target features improve absolute error
against B0 and Morgan; within-drug R² remains negative while within-drug
Spearman is modestly positive. See the [audit](repository_audit.md) for the
corrected three-seed cohort/results and the roadmap for remaining comparisons.

## Deliberate differences from DrEval

This repository is not a DrEval reimplementation, and where it diverges the
reason is scope rather than disagreement:

- **Grouped outer folds with a separate validation group**, rather than full
  nested CV, are implemented for cold-cell/tissue/drug/scaffold. Random-pair,
  double-cold and assay-transfer replication have different estimands. Fitting
  seeds on one external cohort do not measure dispersion over new screens.
- **`cold_scaffold` split**, which DrEval does not include. Bemis-Murcko
  scaffold grouping is a stricter chemical generalization test than holding out
  drug IDs.
- **Graph encoders written in plain PyTorch** rather than PyTorch Geometric, to
  keep the dependency surface small.

Using `drevalpy` directly as an external cross-check would be a strong addition
and is listed in [`research_roadmap.md`](research_roadmap.md).

## Literature added during the 2026-09-30 audit

**Chen et al. — Harmonization and integration of pharmacogenomics screens.**
*Bioinformatics* 42(7), btag382 (2026).
<https://doi.org/10.1093/bioinformatics/btag382>

Compares curve fits, response metrics and concentration windows across screens.
Its strongest agreement uses sigmoid-derived AUC on a truncated dose interval.
This directly motivates obtaining raw curves and harmonizing endpoints before
treating published GDSC and PRISM AUC as the same scale. The current local
fitted-response summaries do not implement this procedure.

**Xia et al. — A cross-study analysis of drug response prediction in cancer
cell lines.** *Briefings in Bioinformatics* 23(1), bbab356 (2022; online 2021).
<https://doi.org/10.1093/bib/bbab356>

Evaluates five screens and recomputes response summaries from viability curves.
Its fixed-concentration-range AUC comparison reinforces the need to distinguish
assay portability from generalization to new biological or chemical entities.
Useful for designing concordance diagnostics and a harmonized external cohort.

**Corsello et al. — Discovering the anticancer potential of non-oncology drugs
by systematic viability profiling.** *Nature Cancer* 1, 235–248 (2020).
<https://doi.org/10.1038/s43018-019-0018-6>

The primary PRISM repurposing reference. Use its screen definitions and source
release metadata when documenting pooled cell-line viability, secondary
concentration-response profiling, technical replicates and QC decisions.
Chemical connectivity matches need separate sensitivity analysis because they
ignore stereochemical distinctions.

**Schubert et al. — Perturbation-response genes reveal signaling footprints in
cancer gene expression.** *Nature Communications* 9, 20 (2018).
<https://doi.org/10.1038/s41467-017-02391-6>

PROGENy infers pathway activity from perturbation-response footprints rather
than averaging expression of pathway members. The current GMT pathway means
are an exploratory feature block, not PROGENy. A faithful extension requires
signed weighted coefficients and documented gene alignment.

Chen et al. provide [derived profiles](https://zenodo.org/records/20620409),
[CTRP inputs](https://zenodo.org/records/17981848) and
[GDSC/PRISM replicate-level inputs](https://zenodo.org/records/17196025), plus
[analysis code](https://github.com/digitaltumors/Pharmacogenomics_Screens_Harmonization).
The GDSCv2/PRISM raw inputs are now checksum-verified locally and used in the
separate shared-window experiment described in [roadmap 2–6](roadmap_2_6.md).
CTRP and derived profiles have not been incorporated. The local fitter is a
documented 3PL protocol, rather than an exact copy of the reference analysis.

**Angelopoulos and Bates — A Gentle Introduction to Conformal Prediction and
Distribution-Free Uncertainty Quantification.**
[arXiv:2107.07511](https://arxiv.org/abs/2107.07511).
Use the independent calibration sample and finite-sample quantile construction.
The current implementation uses entity-maximum residuals to respect the
evaluation unit; it reports empirical coverage under assay shift.

**Barber et al. — Conformal prediction beyond exchangeability.**
*Annals of Statistics* 51(2), 816–845 (2023).
[doi:10.1214/23-AOS2276](https://doi.org/10.1214/23-AOS2276).
This motivates explicitly qualifying interval guarantees under distribution
drift. The local baseline does not implement that paper's weighted/randomized
extension or claim that ordinary conformal intervals solve arbitrary assay shift.

These papers support protocol decisions; they do not validate this repository's
numerical results. A failed small Ridge target predictor does not refute the
more extensive pretraining or data used by published target-profile models.
