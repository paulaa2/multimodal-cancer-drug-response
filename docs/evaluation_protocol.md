# Evaluation Protocol

This is the authoritative description of how models in this repository are
evaluated. Other documents describe *what* is implemented; this one describes
what counts as evidence. If a result is not produced under this protocol, it
should not be reported.

The protocol follows the practices recommended in recent critical reviews of
drug-response prediction, summarized in [`related_work.md`](related_work.md).
Three conclusions from that literature drive the design:

1. Most published gains are inflated by data leakage, and the most common leak
   is fitting preprocessing on all samples before splitting.
2. The split must match the claim. A random split over drug-cell-line pairs
   cannot support a claim about unseen cell lines or unseen drugs.
3. Global RMSE and Pearson correlation are dominated by differences in mean
   potency between drugs, so they hide whether a model learned anything about
   drug-cell-line interaction.

## 1. Splits

Each split holds out whole entities, and every split answers one question. The
names on the left are used in code and file names; the names in brackets are the
conventional names in the literature.

| Split | Held out | Question | Application |
| --- | --- | --- | --- |
| `random_pair` [LPO] | individual pairs | Can it fill gaps in a partly measured matrix? | imputing missing screen entries |
| `cold_cell` [LCO] | whole cell lines | Can it generalize to a new biological context? | personalized medicine |
| `cold_tissue` [LTO] | whole tissues of origin | Can it generalize to a new cancer type? | drug repurposing |
| `cold_drug` [LDO] | whole drugs | Can it generalize to an unseen compound? | drug design |
| `cold_scaffold` | whole Bemis-Murcko scaffolds | Can it generalize to novel chemistry? | drug design, stricter |
| `cold_both` | drugs and cell lines | Can it generalize when both are novel? | hardest setting |
| `transfer_prism` | assay | Do GDSC-fitted models transfer to PRISM AUC? | cross-study |

`random_pair` is a debugging and imputation setting, not evidence of
generalization. Report it, but never lead with it.

`transfer_prism` is not a cold split. Train and validation stay inside GDSC2
(validation is the existing `random_pair` validation fold, so early stopping
cannot see PRISM). Test is PRISM secondary-screen AUC for compounds matched to
GDSC drugs by InChIKey and cell lines that already appear in the GDSC cohort.
Overlapping drugs and cells are allowed: the question is assay shift, not
unseen chemistry. The target is `auc`, not `ln_ic50`, because the two screens
do not share a concentration grid. PRISM SMILES cells can list isomers
comma-separated; each candidate is parsed and the first InChIKey that matches
a GDSC drug is kept. Build with `python -m mcdrp.data.build_transfer_cohort`.

`cold_tissue` is stricter than `cold_cell` because held-out cell lines share no
lineage with training cell lines. `cold_scaffold` is stricter than `cold_drug`
because holding out a drug ID still leaves close structural analogues in
training.

`cold_both` labels mixed held-out blocks as `unused`, so train, validation, and
test have disjoint cell lines *and* disjoint drugs. Training scripts ignore
`unused` rows. This costs a large fraction of the data, which is why its
absolute numbers are not comparable to the other splits.

Build them all with:

```powershell
python -m mcdrp.splits.make_splits
```

`make_splits` asserts the absence of leakage for each split rather than trusting
the construction, and writes the evidence to `data/reports/splits_summary.json`.

## 2. Leakage rules

These are not stylistic preferences. Violating any of them invalidates the
numbers.

- Assign train/validation/test groups **before** fitting anything.
- Fit scalers, PCA, feature selection, pathway scoring, and target scaling on
  training rows only, separately for every split.
- Never use test rows for early stopping, checkpoint selection, or
  hyperparameter choice. Select on validation, then report test once.
- Select hyperparameters **independently on each split**. A setting chosen on
  `random_pair` is not valid for a cold-start claim. The `*_tuning_random_pair`
  configs are smoke tests; the `*_tuning_full` configs are the result.
- Never report the best test metric across epochs, seeds, or configurations.
  The selection rule must be fixed before looking at test.
- Keep every row of a held-out entity out of training, not just some rows.

Candidates are ranked on validation, the selected candidate is marked
`selected_by_validation`, and only that candidate enters the test comparison.
The B4/B5 runners currently compute candidate test predictions internally;
the tuners do not use those scores to choose parameters. Test-ranked "best"
tables are descriptive diagnostics and must never determine the selected model.
Same-named splits from different folds/seeds are different partitions: do not
import tuning decisions from another partition. Refreshed CV uses fixed settings
and `--no-b2-tuned` to make this explicit.

## 3. Baselines

Every model is compared against feature-free baselines, computed by
`mcdrp.models.baseline_b0`:

| Baseline | Prediction |
| --- | --- |
| `global_mean` | training mean |
| `drug_mean` | mean response of that drug |
| `cell_mean` | mean response of that cell line |
| `tissue_mean` | mean response of that lineage |
| `mean_effects` | `mu_cell + mu_drug - mu` |

`mean_effects` is the one that matters, and it is a demanding reference. In this
cohort it reaches a global Pearson correlation of **0.906** on the
`random_pair` test set using no transcriptomics and no chemical structure at all.
Any model that does not clearly beat it has not demonstrated that its input
modalities contribute anything.

For unseen entities the reference degrades gracefully: an unknown cell line or
drug falls back to the global mean, so `mean_effects` collapses onto `drug_mean`
in `cold_cell` and onto `cell_mean` in `cold_drug`. That is the correct
behaviour, and it is why the achievable performance differs so much per split.

## 4. Metrics

`mcdrp.metrics.regression_metrics` reports three families. Reporting only the
first is the single most common way to overstate a model.

**Global.** `rmse`, `mae`, `pearson`, `spearman`, `r2`. Useful, but inflated by
between-drug variance in potency.

**Mean-effect normalized.** `r2_normalized = 1 - SSE_model / SSE_reference`.
The denominator is the squared error of the train-fitted `mean_effects`
reference, not the centered variance of residuals. Normalized Pearson/Spearman
correlate `y - reference` with `prediction - reference`. Residuals can reflect
interactions, omitted main effects, measurement noise and calibration bias;
they are not a purified causal biological signal.

**Stratified.** `pearson_per_drug`, `r2_per_drug`, `pearson_per_cell`,
`r2_per_cell`, computed within each entity and averaged over entities. Groups
with fewer than two rows are skipped, and `n_groups_per_drug` records how many
contributed.

The reference is always fit on training rows only, so normalization introduces
no leakage.

Within-drug R² measures absolute prediction accuracy. Negative R² means more
squared error than that evaluation group's response mean; it does not imply
zero ranking skill. Report within-drug Spearman separately. Each metric can
have a different number of eligible groups when a target or prediction is
constant. Published GDSC/PRISM AUC values are not automatically comparable:
dose windows, units, viability readouts and curve fits require harmonization.
The separate shared-window cohort implements this endpoint comparison on
identical retained observations; its exposure duration remains study-specific.
See [roadmap 2–6](roadmap_2_6.md) for the fitting and eligibility protocol.

### Why this matters, measured in this cohort

B0 test-set results, illustrating the gap between global and honest metrics:

| Split | Model | Pearson (global) | Pearson (per drug) | R2 (per drug) |
| --- | --- | --- | --- | --- |
| `random_pair` | `drug_mean` | 0.846 | undefined | −0.013 |
| `random_pair` | `mean_effects` | 0.906 | 0.640 | 0.363 |
| `cold_cell` | `mean_effects` | 0.853 | undefined | −0.016 |
| `cold_tissue` | `mean_effects` | 0.860 | undefined | −0.144 |
| `cold_drug` | `mean_effects` | 0.311 | 0.624 | −4.835 |

`drug_mean` predicts one constant per drug, so it achieves a global Pearson of
0.846 while explaining *none* of the variation within any drug. Its per-drug
correlation is undefined because its per-drug predictions have zero variance,
and its per-drug R2 is slightly negative. A reader who sees only the 0.846 would
conclude the model is good. It is a lookup table.

The same effect explains why `cold_cell` and `cold_tissue` keep a high global
correlation: with cell effects unavailable, the model falls back to drug means,
which still tracks the dominant between-drug variance while explaining nothing
within a drug.

## 5. Reporting requirements

A result table is complete when it has all of:

- every split in section 1, with unique drugs, unique cell lines, and pair
  counts per split
- the B0 baselines alongside the models, on identical splits
- both global and normalized metrics
- an explicit statement of the validation-based selection rule used
- the random-to-cold degradation, not just the best split

Known gaps in the current results, which should be closed before any of this is
written up as a finding:

- **Uneven replication.** The corrected primary LDO comparison has five outer
  folds; strict PRISM has three fitting seeds on one fixed partition. Other
  architectures/splits still have historical results that require refreshed
  training. Dispersion over folds or seeds does not establish significance.
- **Pseudoreplication.** Pairs are not independent; the units of replication are
  cell lines and drugs. Significance claims must account for the grouping, so no
  significance is claimed.
- **Cross-study is one screen.** `transfer_prism` uses published `auc`, with
  GDSC-only training/validation and overlapping drugs/cells in PRISM. Corrected
  targets reduce absolute error but retain negative within-drug R² and modest
  ranking. Seeds on that screen do not estimate independent-study uncertainty;
  measured assay concordance is a diagnostic, not an accuracy upper bound.

## 6. Claim discipline

This is a preclinical cell-line modelling project. Cell-line drug response is an
experimental proxy for clinical response, model attributions are not causal
biomarkers, and cross-study transfer in this field is known to be poor. Claims
should be phrased as representation-learning and generalization findings.

## 7. Reserved calibration and negative controls

Calibration is an optional fourth role, carved out of training entities without
using outcomes. Its entities are excluded from training and validation, and
`unused` records validation rows removed for that purpose. Test stays fixed.
The scoring contract requires complete predictions and matching outcomes for
each requested role; calibration labels never select a model or early stop.
B0, B6 and RF can predict this role. Named cold splits also require calibration
groups to obey their entity separation policy.

Validation-fitted affine postprocessing and group-maximum conformal residuals
use distinct datasets. Report row coverage, whole-entity coverage, width and
bias together; apparent row coverage does not imply coverage of an entire
cell/drug response profile. Assay-shift coverage is empirical. Paired entity
bootstrap compares frozen predictions, conditional on fitted models, and does
not represent uncertainty over a new independent study.

Entity nulls permute features within complete partition-membership signatures,
with consistent donors across repeated pairs and drug modalities. Save donor
maps and fixed-point counts. Graph-only has no expression/cell encoder; its
within-drug ranking is undefined by construction. Chemical matching and
descriptor expressiveness are distinct: strict InChIKey matching does not make
canonical-SMILES descriptors or the current graph encoder stereochemistry-aware.
