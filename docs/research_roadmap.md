# Research Roadmap

Prioritized next steps, ordered by how much each one strengthens the
conclusions rather than by how interesting it is to build. Rationale and
citations are in [`related_work.md`](related_work.md).

## Where the project stands

The model ladder B0-B7 is implemented. Every model writes per-row predictions,
reporting computes mean-effect normalized and stratified metrics, and the
leave-tissue-out split is wired into the default split suite and full
experiment configs. B4/B5 have a four-point validation tuner; B6 reuses
validation-selected B2 XGBoost hyperparameters. A multi-seed runner rebuilds
splits and models under several seeds and reports mean and spread, without
p-values.

The first thing that machinery revealed, on the `random_pair` test set:

| Model | Features | Pearson (global) | Pearson (per drug) | R2 (per drug) |
| --- | --- | --- | --- | --- |
| B0 `mean_effects` | none | 0.906 | 0.640 | 0.363 |
| B2 tuned XGBoost | Morgan + expression PCA | 0.913 | 0.668 | 0.396 |

Read globally, XGBoost adds 0.007 Pearson over a model with no features — which
would look like the modalities contribute nothing. Read per drug, it lifts R2
from 0.363 to 0.396, about a 9% relative gain in explained within-drug variance.
Both readings are of the same run. The global number is the misleading one, and
this is why the reporting rules in
[`evaluation_protocol.md`](evaluation_protocol.md) exist.

## Tier 1 — required before any result is a finding

### 1. Multi-seed runs with dispersion — code done, numbers pending

`python -m mcdrp.experiments.multiseed` rebuilds splits and retrains under
several seeds, then writes mean and spread. Split construction and model
initialization share the same seed. Canonical seed-42 artifacts are not
overwritten. The runner does not compute p-values: pairs are not independent,
and the units of replication are cell lines and drugs.

The experiment that counts is `configs/experiments/multiseed_b0_b2_full.json`.
Until those numbers exist, the README ranking remains provisional.

### 1b. Grouped cross-validation for the cold splits — code done, numbers pending

A single group-level partition cannot support a cold-split claim.
`cold_drug` holds out 36 of 238 drugs and validates on a different 36, and
because drug potency dominates ln(IC50), the two subsets measure different
things: under seed 42 the validation mean is 1.924 and the test mean is 3.463,
a gap of 1.54 against a target standard deviation of about 2.7.
`cold_scaffold` shows the same gap with the sign reversed. That single number
is what tuning was selecting against, which is why tuning B4 made test RMSE
worse on exactly those two splits and why XGBoost early-stopped at iteration 0
on `cold_scaffold`.

`python -m mcdrp.splits.grouped_cv` builds K folds over the grouping entity so
every group is tested exactly once. Each fold is written as its own directory
mirroring the flat split layout, `splits/cv/cold_drug/fold_0/cold_drug.csv`, so
models consume a fold unchanged through `--split-dir`. No model or tuner knows
that cross-validation exists.

`python -m mcdrp.results.grouped_cv_report` computes metrics per fold and then
averages them. Per fold, not pooled: each fold has its own training rows, so
`r2_normalized` is only defined against that fold's `mean_effects` reference,
and pooling first would silently compare every fold against one arbitrary
reference. As with multi-seed, no p-values: folds share training data and the
units of replication are still drugs and cell lines.

Five folds over `cold_drug` test all 238 drugs and cut the validation/test mean
gap from 1.54 to 0.51 on average. The measurement this replaces was worse than
it looked: B0 `mean_effects` reads 2.653 RMSE on the seed-42 partition, but
across folds it ranges from 2.200 to 3.204, giving 2.662 +/- 0.424. Any cold
split comparison decided by a tenth of an RMSE was reading noise.

### 2. Persist per-row predictions — done

B0–B7 write tidy prediction files (`pair_id`, `stage`, `model`, `variant`,
`split_name`, `subset`, `y_true`, `y_pred`) under `results/predictions/`.
`metrics_report` computes all metric families from those files, so a new metric
no longer requires retraining.

`variant` separates a model's default-hyperparameter run from its tuner's
validation-selected run. A tuner reuses the stage and model labels of the model
it tunes, so without it `b4.csv` and `b4_tuned.csv` grouped together and every
B4 metric was computed over both runs at once. Files written before the column
existed keep working: the variant is recovered from the `_tuned` filename stem.

### 3. Normalized metrics for every model — done

Mean-effect normalized metrics, including `r2_normalized`, are computed in
reporting from the prediction artifacts and the split-wise `mean_effects`
reference. They are no longer limited to B0.

Reports must select the winning row whole, via
`mcdrp.results.selection.lowest_rmse_rows`, and never with
`sort_values("rmse").groupby(keys).first()`. pandas reduces each column
independently and skips NaN, so the second spelling took the winner's RMSE but
filled each hyperparameter column from whichever other stage first had a value
there. Stages report disjoint hyperparameter columns, so that happened on
essentially every "best model" row.

### 3b. Guard the expression pipeline against near-constant genes — done

`fit_expression_pipeline` drops genes whose training standard deviation falls
below `MIN_GENE_STD` and clips scaled values to `MAX_ABS_ZSCORE` on both the
fit and transform paths. Without this, `StandardScaler` divides by almost zero:
`SMIM10L3 (122526779)` has standard deviation 6.3e-08 across the 501 training
cell lines of the seed-13 `cold_cell` split, which gave the held-out line
ACH-000470 a z-score of 3.05e7, a PCA component of 6.2e5, and a B1 ridge
prediction of 28269 against a target range of roughly [-9, 12]. Tree models
hide the problem because they never extrapolate; linear models and MLPs do not.

The guard is a safety net, not a change of representation. On the seed-42
splits it drops 17 of 19215 genes and moves explained variance from 0.8045 to
0.8047, leaving the canonical artifacts comparable.

### 4. Close the `cold_tissue` gap — code done, numbers pending

`cold_tissue` is in `DEFAULT_SPLITS` and in the full B4–B7 experiment and
tuning configs. The remaining hole is generating the trained-model numbers on
that split.

## Tier 2 — strengthens the scientific contribution

### 5. Real pathway resources

The built-in pathway panel validates the pipeline but is not a serious
biological resource. Move to MSigDB Hallmark, Reactome, or PROGENy via the
existing `--gene-sets` option. Pathway scaling is already fit on training cell
lines only per split.

### 6. Randomization controls for modalities

Stronger than an ablation: keep the architecture fixed and permute one view.
Shuffle the drug features across drugs, then the cell features across cell
lines, and measure the damage. If permuting a modality costs nothing, that
modality is decorative.

The literature's expectation is specific and worth testing directly: expression
carries nearly all the signal for `random_pair` and `cold_cell`, while drug
representation only matters for `cold_drug` and `cold_scaffold`. B6 and B7 can
answer this on this cohort.

### 7. Drug target profiles as a modality

The clearest architectural signal in the recent literature is that predicted
protein-target profiles improve generalization to unseen drugs more than better
structural encoders do, and that removing structural features hurts less than
removing biological ones. Since `cold_drug` is where every model fails, this is
the most promising direction for the hardest split. GDSC already ships a
`putative_target` field, which is a cheap first version.

### 8. Cross-study validation

Train on GDSC2, test on CTRP, CCLE, or PRISM after harmonizing the response
target. The caveat from the literature is important: technical differences
between screens are large, so treat this as a transfer experiment, not as extra
training data.

## Tier 3 — polish and depth

### 9. Uncertainty

Cold-start predictions should come with a confidence signal. Deep ensembles,
Monte Carlo dropout, or conformal intervals — conformal being the cheapest
defensible option since it needs only a calibration split.

### 10. Multi-omics cell features

Mutation status for driver genes, copy-number summaries, and methylation or
proteomics if a clean matched release exists. Implement as feature blocks with
train-only fitting and evaluate through the same protocol.

### 11. Interpretation

Pathway-level importance, scaffold-level error analysis, and two or three case
studies on known targeted therapies. Treat attributions cautiously: leakage is
known to inflate apparent biomarker counts several-fold without improving
recovery of known drug targets, so any biomarker claim needs a stability check
such as bootstrap selection frequency.

### 12. External benchmark cross-check

Run `drevalpy` on the same cohort as an independent implementation of the
evaluation protocol. Agreement would be strong evidence the numbers here are
sound; disagreement would be informative.

## Explicit non-goals

- Clinical or patient-level claims. This is preclinical cell-line modelling.
- Chasing the best `random_pair` RMSE. It is an imputation setting.
- Adding architectures without an ablation that isolates their contribution.
