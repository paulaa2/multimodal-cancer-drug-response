# Roadmap items 2–6: protocols and execution

The current status is maintained in [research roadmap](research_roadmap.md).
The small, versioned result artifact is
[`roadmap_2_6_summary.json`](../data/reports/roadmap_2_6_summary.json).
Large source data and predictions remain in ignored data/results directories.
These experiments are exploratory: PRISM labels were examined before this phase.

## 2. Shared assay endpoint

`configs/public_sources.json` freezes URLs, checksums and local paths. The actual
raw GDSCv2 and PRISM inputs are in [Zenodo record 17196025](https://zenodo.org/records/17196025).
This is distinct from the CTRP-only record 17981848. Downloads verify checksums
before atomic replacement and reuse only verified cached files.

`configs/harmonization.json` specifies GDSC doses in linear µM and PRISM doses
in log10 µM. PRISM is restricted to MTS010. Identity comes from unique
within-study names and the existing strict chemical map; ambiguous aliases are
discarded. Dose repeats are summarized within replicates before pooling
replicates. The response convention is viable fraction, with a decreasing
three-parameter logistic curve: top=1, bottom in [0,1], positive slope.
The slope multiplies log10-dose inside the logistic function; the equivalent
Hill coefficient is slope/ln(10).
Raw noisy viability is not clipped before fitting. Flat/nonimproving curves
receive a prespecified constant fallback; failed fits are excluded.

Normalized area is integrated over **log10(µM) [-1.9, 0.9]**, approximately
0.0126–7.943 µM. At least four measured doses and complete window coverage are
required. There is no extrapolation outside observed dose support. Published
AUC is retained in `published_auc`; `auc` holds the new endpoint in a separate
cohort. This is a documented 3PL protocol, not an exact reimplementation of a
publication's fitter. Exposure duration and assay biology remain unresolved.
Published protocol context is recorded separately in
`configs/assay_protocol_context.json`: [GDSC2's nominal 72-hour assay](https://depmap.sanger.ac.uk/documentation/datasets/drug-sensitivity/)
and [PRISM repurposing's nominal five-day incubation](https://pmc.ncbi.nlm.nih.gov/articles/PMC7328899/).
The imported CSVs do not verify timing per curve, and viable-fraction AUC is
not corrected for the different exposure periods or readouts.

The retained cohort has 9,197 GDSC training and 1,550 validation pairs, covering
23 drugs; PRISM has 10,785 test pairs, 335 cells and 38 drugs. Predictions are
stratified into drugs seen and unseen in this restricted GDSC sample. Both
endpoint comparisons use **identical retained observations and partitions**.
The concordance diagnostic also uses identical cell-drug intersections.
Changing sample composition cannot be credited as an endpoint improvement.

The curve family is informed by the authors' [fitting source at the frozen
reference commit](https://github.com/digitaltumors/Pharmacogenomics_Screens_Harmonization/blob/144210bb1860089c82d038c9accbee4e6ec45c5e/DrugResponseProfileFitCalculation.py).
Its fitting bounds, truncation window and extrapolation choices are distinct
from this repository's protocol.

## 3. Random Forest and DrEval

RF has a fixed budget of 200 trees and four validation-selected candidates:
`max_features` in {sqrt, 0.1} and `min_samples_leaf` in {5,20}. Bootstrap is
enabled. Each cold-drug fold fits PCA and the target vocabulary on training
data; only training-constant columns are removed. Test labels never select a
candidate. Target+PCA and Morgan+PCA are compared with the same fold-fitted B0.

The cross-check calls the **official DrEvalPy 1.5.1** dataset and evaluation
APIs. The wheel checksum is recorded in
[`dreval_crosscheck_summary.json`](../data/reports/dreval_crosscheck_summary.json).
It checks identity-validated prediction metrics and cold-drug entity separation.
This establishes metric/dataset interoperability, not an independently trained
DrEval benchmark. DrEval assigns zero correlation to constant predictions;
this repository marks their correlation undefined. This convention difference
is recorded explicitly rather than treated as an agreement or implementation bug.
The reproducible optional dependency is `pip install -e ".[validation]"`.

## 4. Entity modality nulls and graph-only

Nulls permute whole cell or drug feature vectors with seeds 7,13,42. Every pair
for an entity receives the same donor. Donors share the entity's complete
partition-membership signature, preventing train-only entities from receiving
held-out-only donors. Multiple drug modalities use the same donor mapping.
Fixed points and singleton strata are reported, and donor CSVs are retained.
These are negative controls; outcomes are not evidence of biological causality.
The legacy unrestricted target shuffle remains a separately named diagnostic.

The drug-only graph control reuses the molecular encoder, has no cell encoder
and never reads expression. Dropout is zero. It minimizes a count-weighted MSE
on drug means: this equals pair-level MSE up to the constant within-drug
variance. Training is full-batch, seed precedes weight initialization, and
validation selects the early-stopping checkpoint. Graph-only predictions are
constant within a drug and therefore cannot rank its cell sensitivities.

## 5. Annotations, embeddings and chemical sensitivity

The local HGNC snapshot is dated 2026-09-30 and protected by SHA-256. Its live
download URL is mutable; retain the checked local snapshot for reproduction.
Approved symbols take precedence; previous symbols and aliases resolve only
when unique. Mechanisms/families are not expanded into guessed genes. The
normalization table retains raw tokens, resolution status and source hashes.
Curated pharmacological families take precedence over HGNC aliases: a generic
PI3K/AKT label must not become one specific family member. Exact approved gene
symbols remain genes. There are 287 resolved gene tokens and 159 drugs with at
least one gene. The
original annotation's curation date is unknown. Raw, gene-only, mechanism-only
and typed views are separate feature ablations with training-only vocabularies.
Unique HGNC aliases (including TOP2 and ESR) remain nomenclature resolutions:
they do not establish isoform-selective pharmacology. Primary binding evidence
and expert review are needed before using these annotations mechanistically.
The first annotation normalization treated some family aliases as genes. Those
initial view results are retained as diagnostics; the current report uses the
corrected refresh in `results/roadmap_2_6/curation_corrected`. It verifies the
annotation hash before including any target view in the ranking.

ChemBERTa embeddings were regenerated offline from cached checkpoint
`DeepChem/ChemBERTa-77M-MLM@ed8a5374f2024ec8da53760af91a33fb8f6a15ff`,
with mean pooling, maximum length 256 and 384 dimensions. The audit requires
unique complete IDs, matching SMILES, finite values, a resolved commit and
matching cohort/output hashes. B7 accepts `--embedding-manifest` and rejects
inconsistent frozen artifacts. Runs without one explicitly report unverified
provenance. The regenerated vectors numerically reproduce the legacy vectors
(maximum absolute difference zero), while preserving both files.

Full-InChIKey sensitivity evaluates frozen strict-transfer predictions on
13,597 test pairs and 44 drugs, excluding the two connectivity-only matches.
This is a test-subset sensitivity, not a newly trained model or a selection
criterion.

## 6. Calibration and uncertainty

Fifteen percent of training entities are reserved using IDs and a fixed seed,
without consulting responses. Cold-drug uses drugs; transfer uses cells.
Selected entities are removed from training and excluded from validation if
they occurred there. The outer test remains unchanged. PCA/vocabularies fit
only the reduced training sample. Calibration predictions are generated after
model selection. The original early-stopping validation cannot serve as the
calibration sample.

Raw and validation-fitted nonnegative affine predictions are both reported;
test results do not choose between them. Calibration scores are the maximum
absolute residual per reserved entity. Radius uses the finite-sample order
statistic `ceil((n_groups+1)*(1-alpha))`; insufficient groups produce an infinite
radius explicitly. Reports include row and whole-group coverage, interval
width, RMSE, bias, and diagnostic calibration slope/intercept. Shared viable
fraction AUC intervals are clipped to [0,1].

Paired bootstrap resamples held-out drugs jointly for a candidate and B0,
reporting conditional 95% intervals for row-weighted RMSE differences. It does
not refit the model or turn overlapping folds into independent studies.
Transfer interval coverage is **empirical**, with no ordinary conformal
guarantee: assay shift, different drug support and dependent measurements
violate exchangeability. See
[Angelopoulos and Bates](https://arxiv.org/abs/2107.07511) and
[Barber et al.](https://arxiv.org/abs/2202.13415).

## Commands

Run from the repository with `PYTHONPATH=src`, using an environment containing
the required extras. Graph controls/embeddings need PyTorch; embeddings also
need Transformers. The main ablation runner explicitly requires CUDA.

```text
python -m mcdrp.data.public_sources
python -m mcdrp.data.target_annotations
python -m mcdrp.data.harmonize_endpoints
python -m mcdrp.features.create_drug_embeddings --revision ed8a5374f2024ec8da53760af91a33fb8f6a15ff --output data/external/drug_embeddings_pinned.csv --summary data/reports/drug_embeddings_pinned_summary.json
python -m mcdrp.features.embedding_audit --embeddings data/external/drug_embeddings_pinned.csv --manifest data/reports/drug_embeddings_pinned_summary.json --output data/reports/drug_embeddings_pinned_audit.json
python -m mcdrp.experiments.roadmap --phase forests
python -m mcdrp.experiments.roadmap --phase ablations
python -m mcdrp.experiments.roadmap --phase harmonized
python -m mcdrp.experiments.annotation_views
python -m mcdrp.experiments.endpoint_comparison
python -m mcdrp.experiments.run_experiment configs/experiments/roadmap_graph_only.json
python -m mcdrp.experiments.run_experiment configs/experiments/roadmap_b7_pinned.json
python -m mcdrp.results.roadmap_report
```

With the `validation` extra installed, check each fold with the official
DrEval API, for example (substitute fold 1–4 for the other checks):

```text
python -m mcdrp.results.dreval_crosscheck --predictions results/predictions/cv_checked/cold_drug/fold_0/b0.csv results/roadmap_2_6/ablation/fold_0/standard/predictions.csv results/roadmap_2_6/random_forest/fold_0/predictions.csv --split-dir data/processed/splits/cv/cold_drug/fold_0 --output results/roadmap_2_6/reports/dreval_fold_0.json
```

## Executed evidence

The canonical [summary](../data/reports/roadmap_2_6_summary.json) contains 23
five-fold model/variant aggregates, 30 donor diagnostics, 105 paired bootstrap
comparisons, independent calibration diagnostics, matched endpoint comparisons
and full-key sensitivity. The fold-level CSV and stratified interval files
are retained under `results/roadmap_2_6`; large predictions remain ignored by Git.
The [validation record](../data/reports/roadmap_2_6_validation.json) records the
182 passing tests, static checks, artifact counts and verified hashes.

Mean cold-drug RMSE is 2.434 for target RF, 2.469 for Morgan RF, 2.515 for
standard target XGBoost, 2.632 for frozen B7 and 2.726 for graph-only.
The raw transfer intervals cover every tested drug response in only 17.9% of
PRISM cells for Morgan and 24.8% for targets, against a nominal group target
of 90%. High row coverage therefore cannot be interpreted as successful
simultaneous cell coverage. The [research roadmap](research_roadmap.md) records
the remaining experiments and the limits of these exploratory comparisons.

Roadmap phases retain hashes and skip completed cases only when their recorded
inputs/code match. Use a new `--root` after a fitting-code or data change; do not
mix stale tuning/predictions into a refreshed ranking. The initial accidental
CPU execution was interrupted and preserved in
`results/roadmap_2_6_cpu_diagnostic`; the published comparison uses the corrected
CUDA runs. Prediction identity and complete-subset coverage are checked before
scoring. The authoritative status does not claim that tests prove scientific
conclusions or that these experiments establish clinical efficacy.
