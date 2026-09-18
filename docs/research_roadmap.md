# Research Roadmap

Prioritized next steps, ordered by how much each one strengthens the
conclusions rather than by how interesting it is to build. Rationale and
citations are in [`related_work.md`](related_work.md).

## Where the project stands

The model ladder B0-B7 is implemented and has been run across the split suite.
The evaluation machinery now includes a `mean_effects` reference, mean-effect
normalized metrics, per-drug and per-cell-line stratified metrics, and a
leave-tissue-out split.

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

### 1. Multi-seed runs with dispersion

Every number in the repository is a single run. Differences of a few percent
between models are currently uninterpretable, which means the model ranking in
the README is provisional.

Run each model/split with several seeds and report mean and spread. Seeds must
vary both the split construction and model initialization, since split
composition is a large variance source for cold settings.

Because drug-cell-line pairs are not independent — the units of replication are
cell lines and drugs — any significance test must account for that grouping.
Until then, make no significance claims. This is the pseudoreplication trap.

### 2. Persist per-row predictions

Models currently write only aggregate metrics, so no new metric can be computed
without retraining, and error analysis is impossible.

Have every model write a tidy predictions file (`pair_id`, `split_name`,
`subset`, `model`, `y_true`, `y_pred`), then compute all metric families in one
reporting step. This makes evaluation uniform by construction instead of by
convention, and unlocks tiers 2 and 3 cheaply. It also removes the largest
source of duplication in `src/mcdrp/models/`, where the metric-row builder,
argument parser, and summary writer are copy-pasted across a dozen files.

### 3. Normalized metrics for every model

`mean_effects` normalization is currently wired into B0 only, because the
reference has to be fitted on the training rows of each split and the model
scripts do not thread that context into their metric builders. Item 2 makes this
a non-issue: normalize during reporting, where the split is known.

### 4. Close the `cold_tissue` gap

The leave-tissue-out split exists and has baselines, but no trained model has
been run on it. It is the setting relevant to drug repurposing and the one where
the literature reports the sharpest degradation, so leaving it empty is a
conspicuous hole.

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
