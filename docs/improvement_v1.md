# Bounded exploratory improvement phase

This follow-up addresses the three limitations of stable v1: cold-drug error,
GDSC-to-PRISM transfer and the absence of a competitive neural comparison on the
corrected chemical-identity folds. Stable v1 and tag `v0.2.0-stable` remain the
reference. The new recipe is deliberately experimental and does not inherit
the certification of the stable Python 3.13 environment.

## Fixed protocol

[`configs/improvement_v1.json`](../configs/improvement_v1.json) declares eight
cases: five chemical-identity cold-drug folds and the three original transfer
endpoint comparisons. All use the existing stable cohort, assignments and
training-fitted expression/target preprocessing. No chemical aliases cross a
cold-drug partition. No PRISM outcomes enter source-model selection or fitting.
The stable artifact manifest is checked before the first new fit.

* **Convex ensemble:** RF-targets, RF-Morgan and frozen-embedding B7; 15 weights
  on a simplex with step 0.25. Choose minimum validation RMSE separately in each
  case. The simplex contains single models, so combining them is optional.
* **Residual neural towers:** drug and cell encoders, additive drug/cell heads
  and an interaction head receiving drug, cell and their elementwise product.
  Two representations: targets + frozen ChemBERTa; Morgan + targets + frozen
  ChemBERTa. Width 64, dropout 0.2, AdamW learning rate 0.001 and decay 0.01,
  batch size 2,048, up to 120 epochs with patience 15. Fit both seeds 42 and 43
  for each candidate. Select the representation by validation RMSE of the two
  best-checkpoint predictions averaged together. Neither seed is selected on
  test. Continuous features are scaled on unique training entities only;
  binary features stay binary. Target mean/scale are fitted on training pairs.
  AUC predictions are clipped to [0,1] after averaging. This is a new architecture,
  not a retroactive recertification of historical B3/B4/B5.
* **Error diagnostics:** per-drug bias and centered error of stable RF-targets,
  nearest-training-drug Morgan Tanimoto and known target-token count. These are
  descriptive diagnostics, not features selected with test labels. Cold-drug
  is exact chemical-identity separation, not scaffold novelty.

All neural candidate weights and learning histories are retained. Predictions
are checked after restoring saved models and scalers. Candidate and selection
fingerprints exclude calibration/test outcomes. Fitted-model bootstrap intervals
resample chemical identities for cold-drug and cells for transfer; they do not
include retraining uncertainty or make the reused datasets confirmatory.
Independent scoring reads saved prediction floats with pandas' `round_trip`
parser: its default fast parser can move nearly tied neural predictions by one
floating-point unit and change rank metrics. No verification tolerance is relaxed.

## Executed results and decision

All 32 neural fits, eight ensemble selections and four adapters completed.
Independent verification rescored complete predictions, checked selection on
validation, reloaded all neural models and reproduced target-domain adapters
and interval diagnostics. The complete Torch-enabled suite passed 196 tests;
Ruff and mypy passed. The notebook has ten executed code cells and eight
figures, each exported as PNG, SVG and PDF.

On five cold-drug folds, mean RMSE is 2.300128 for the neural seed ensemble,
2.362372 for RF-targets and 2.369393 for the convex ensemble. The neural recipe
wins three folds against RF-targets, improving mean RMSE by 2.63%, but its mean
within-drug Spearman is 0.628029 versus 0.650130 for RF-targets. All five folds
choose targets + frozen embeddings; adding Morgan is not selected by validation.
RF-targets therefore remains a reference for within-drug ranking, rather than
being replaced indiscriminately by the lower-RMSE neural recipe.

For source-only transfer, neural RMSE is 0.137812 on published AUC versus
RF-targets 0.133865; 0.171788 on harmonized AUC versus RF-targets 0.171932;
and 0.141677 on matched published AUC versus RF-targets 0.138175. There is no
consistent source-only transfer gain.

On 168 reserved PRISM evaluation cells / 5,419 pairs, supervised adaptation
reduces neural RMSE from 0.171148 to 0.144575 (15.53%). The paired-cell bootstrap
delta is -0.026573, with conditional 95% interval [-0.028966, -0.024021]. Whole-cell
coverage is 87.5% and mean clipped interval width is 0.656775 AUC units. Coverage
and width remain limitations despite lower point error. The adapters largely
correct compound response levels; they do not learn a new within-drug ranking.
The notebook includes clipping-only controls: the neural source predictor is
already within [0,1], so its adaptation gain is not a clipping artifact.

Descriptively, 76.57% of RF-targets' pooled cold-drug squared error is the
drug-specific mean-bias component, using the exact decomposition into squared
bias and centered error. This diagnoses potency estimation as a priority.
99 of 238 held-out drug records have zero tokens in their fold's training
target vocabulary; this includes absent annotations and out-of-vocabulary
annotations and should not be interpreted as 99 biologically targetless drugs.

These results justify keeping the neural recipe as an experimental comparator
and retaining RF-targets as the stable reference. Better pharmacological evidence,
identically featured tabular controls and a fresh external cohort are still
needed before claiming robust generalization. The three limitations have been
investigated concretely; they have not all been solved by this experiment.

## Separate, supervised PRISM adaptation

The original harmonized PRISM evaluation cohort is split by complete cell ID,
independently of outcomes, using seed 20261001: 20% adaptation training, 10%
validation, 20% calibration, 50% evaluation. This is a new experimental question
using labels in the destination domain. The previously reported source-only
transfer results remain intact.

For each predetermined predictor (RF-targets, RF-Morgan, convex ensemble and
neural ensemble), fit a nonnegative affine map with optional chemical-specific
residual offsets. Offset shrinkage is chosen from 0, 10, 50, 200, alongside the
affine-only option, on development validation cells. Refit the selected adapter
on the full 30% development population. Unknown chemicals receive zero offset;
chemical aliases share offsets. Clip point predictions to [0,1].

Calibrate intervals on the independent 20% PRISM calibration cells using maximum
absolute residual per cell and the finite-sample conformal quantile at alpha 0.1.
Report both row coverage and simultaneous complete-cell coverage, together with
interval width. Both adapted and source-only *point* predictors use PRISM labels
for these new intervals; neither interval variant is zero-shot calibration.
The conformal guarantee assumes exchangeable cells and is not established by
one exploratory partition.

This evaluates new PRISM cells for drugs represented in the destination domain,
not novel drugs or a wholly unexamined study. Reserved PRISM cells may already
have source-domain GDSC observations; this is not a global cold-cell experiment.
Offsets and a shared positive affine map cannot improve within-drug rank order.
A lower AUC RMSE after adaptation therefore does not by itself demonstrate
better biological sensitivity prediction.

The neural predictor also combines different input blocks from RF-targets.
Any measured gain is a gain of the complete recipe, not an isolated causal
effect of its architecture. An identically featured tabular comparison and
new representation ablations remain necessary to separate those effects.

## Reproduce

Use an environment with PyTorch/CUDA and the versions recorded in the compact
report. The executed follow-up uses Python 3.11.14, NumPy 1.26.4, pandas 2.3.3,
scikit-learn 1.7.2, scipy 1.16.3, joblib 1.5.2 and PyTorch 2.9.1+cu128. The
source features were computed under the original stable environment; the neural
profile is separately recorded and does not claim a hash-locked reinstall.

From the repository root:

```powershell
$env:PYTHONPATH = 'src'
python -m mcdrp.experiments.improvement_v1 --stage all
python scripts/build_improvement_notebook.py
python -m nbconvert --to notebook --execute --inplace notebooks/improvement_v1_results.ipynb
```

The run refuses configuration changes in an existing output directory and source
changes when resuming the same stage. To repeat independently, copy the config
to a new file and change `output_root` to a fresh directory. Completed cases can
be resumed without overwriting their fit; failed/incomplete candidates remain
visible. Notebook construction requires a complete checksum-verified manifest.
Plotting needs only the compact snapshot and notebook dependencies, not models,
PyTorch, raw inputs or a GPU.

## Scientific motivation

[DrEval (2026)](https://www.nature.com/articles/s41467-026-72903-w) motivates
strong simple baselines, group-aware evaluation and modest claims when deep
architectures do not consistently outperform them.
[DTLCDR (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12491712/) motivates
testing target and molecular representations together. Our small towers do not
reproduce DTLCDR's target-affinity predictor or single-cell pretraining.

The notebook and machine-readable snapshot report all completed outcomes,
including negative findings. New model promotion requires convincing gains
under the same task, not a visually better plot or a changed endpoint.
