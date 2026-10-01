# Stable v1 protocol

This release freezes the complete preprocessing, training, validation, test and
results workflow for **B0, Random Forest, B6 and B7**. Other architectures and
historical splits remain experimental. Software stability does not imply
clinical utility or successful transfer between assays.

## Scope

The primary comparison uses five complete-chemical-identity-disjoint folds,
seed 42. The 238 drug IDs contain **229 standard full InChIKey identities**;
aliases stay together in training, validation, test and calibration. All 149,364
GDSC pairs are tested exactly once per model. Exact identity disjointness does
not establish scaffold novelty. The six specifications are B0 mean
effects, RF targets/PCA, RF Morgan/PCA, B6 targets/PCA, B6 Morgan/PCA and B7
frozen ChemBERTa/PCA. RF selects from a fixed four-candidate validation grid;
XGBoost uses validation early stopping and the selected iteration range.
Test labels never enter selection.

The historical drug-ID folds separated different IDs of the same molecule:
7,353 test pairs had a structure present in training. Those folds cannot support
an unseen-chemical claim. `data/reports/stable_v1_chemical_audit.json` preserves
the nine alias pairs and overlap counts; the interrupted numerical-validation
run is at `results/stable_v1_failed_id_disjoint/`. Stable v1 reconstructs new
folds using complete InChIKey groups, without choosing folds on test accuracy.
Chemical identity is also the resampling unit for paired bootstrap and the
reservation/scoring unit for cold-drug calibration. Macro per-drug metrics
retain the original record IDs and are labeled accordingly.

The target representation retains the original GDSC `putative_target` tokens,
which mix gene symbols, families and mechanism phrases. The HGNC audit records
this distinction; lexical normalization is not evidence of pharmacological
binding to a particular isoform. The stable comparison does not reinterpret
these tokens as experimentally verified protein targets.

The secondary comparison transfers GDSC→PRISM with strict chemical mapping
(44 full InChIKey matches and two unambiguous connectivity matches):
published AUC, shared-window AUC and published AUC on the same curve-eligible
pairs. Connectivity matches do not establish stereochemical identity. Published
and shared-window AUC have different definitions; compare each model with B0
within the same endpoint, rather than interpreting their raw RMSE difference
as better transfer. This is assay transfer with overlapping cells/drugs, not another
cold-drug experiment. Twelve separate B6 fits reserve calibration entities
independently of fitting and validation across five folds and harmonized
transfer. There are **60 final persisted model fits** in total.

## Reproduce from frozen raw inputs

The certified target is **Windows x86_64, CPython 3.13.5**, with CUDA for
XGBoost. All 78 runtime/development dependencies and wheel hashes are pinned
in `requirements/stable-v1-win-py313.lock.txt`. B7 uses a hash-verified external
embedding table and checkpoint manifest; this run does not regenerate the
encoder and does not require PyTorch.

From the repository root, supply an installed Python 3.13.5 executable:

```powershell
.\scripts\run_stable_v1.ps1 -Python "C:\path\to\python.exe" -Output "results/stable_v1_reproduction"
```

The script creates an isolated environment, installs with `--require-hashes`,
checks dependencies, rebuilds all derived inputs, fits models, executes the
notebook and verifies the complete artifact inventory. Output/environment
names must be new. Failed runs retain `failure.json` and logs; partial output
is never implicitly reused or labeled complete.

The frozen raw files must be available at the paths in `configs/stable_v1.json`.
The verified local archive
`results/releases/stable_v1/stable_v1_inputs.zip` preserves them with a manifest.
Extract it into the repository or another input root. In the locked environment:

```powershell
$env:PYTHONPATH = "src"
python -m mcdrp.experiments.stable_v1 run --inputs-root "C:\frozen-inputs" --output "results/new_run"
python -m mcdrp.experiments.stable_v1 verify --output "results/new_run"
```

Public curve download URLs are recorded in `configs/public_sources.json`.
The exact DepMap release of the previously acquired local files is not
established: checksums identify the frozen inputs. Reconstruction from those
files does not certify authenticated acquisition from a new account. A live
HGNC download cannot silently replace the frozen snapshot.

## Outputs and acceptance

The certified run is `results/stable_v1/`:

- `data/`: newly reconstructed cohorts, chemical mapping, curve fits, folds,
  transfer and independent calibration assignments.
  `cohort.csv` preserves the exact raw reconstruction contract;
  `chemical_cohort.csv` adds identity groups without dropping observations.
- `features/`: train-fitted StandardScaler/PCA, target vocabulary and maps.
  Reloaded preprocessing is checked against original expression transformations.
  Scaling and PCA use float64 with a fixed internal 8-thread BLAS profile;
  final model inputs use float32. Preprocessing
  may be shared within a run only when training cell sets are exactly equal.
- `fits/`: fitted models, complete predictions, metrics, validation grids,
  selected iterations and prediction replay checks.
- `reports/stable_v1_summary.json`: all metrics, compact plotting inputs,
  paired bootstrap, calibration and provenance.
- `stable_v1_results.ipynb`, `stable_v1_results.html`, `figures/stable_v1/`:
  executed report and PNG 300 dpi / SVG / PDF exports.
- `manifest.json`: complete file inventory with sizes and SHA-256.
- `reports/source_code.json`: hashes of source, scripts and dependency files
  recorded when the report is constructed after training.

Acceptance requires exact cohort reconstruction, full out-of-fold coverage,
all 60 models reproducing predictions after reload, reloaded preprocessing
reproducing features, successful notebook execution and artifact verification.
Raw-expression projections must agree under external 8- and 16-thread BLAS
profiles (internally fixed at eight threads) within
an absolute tolerance of **1e-7**, for every distinct training cell set.
Future reproductions compare all six primary mean-fold RMSE values with the
hash-pinned reference in `data/reports/stable_v1_golden.json`, using the
**predeclared absolute tolerance 0.005** and the same scientific configuration.
The same absolute tolerance checks each of the 60 individual test RMSE values,
including transfer and calibration fits. Reconstructed cohorts, curve-fit
tables and assignments must also match the reference CSV hashes exactly.
These are numerical regression checks, not significance thresholds or selection
criteria. The reference was established only after the complete revised recipe
passed validation. Bitwise training
reproducibility across arbitrary hardware/CUDA versions is not claimed.

`scripts/verify_stable_repeat.py` supplies an additional targeted diagnostic:
from another locked environment, refit fold 0's RF Morgan winner and B6 targets
with newly reconstructed raw expression/drug features and compare predictions
within 1e-7. It repeats the fixed recipe without choosing settings on test; it
does not constitute a second full run of all 60 models.

The first clean run using the historical float32 PCA failed the original 0.005
checkpoint tolerance for RF Morgan. Its results and failure remain at
`results/stable_v1_failed_float32/`; the audit is preserved in
`data/reports/stable_v1_numerical_audit.json`. An expression-only investigation
found sensitivity to BLAS thread count and motivated float64 preprocessing.
An additional float64 run stopped before training because the harmonized
scenario's final float32 cast differed by 2.38e-7 across thread profiles,
despite only 4.70e-11 difference in the float64 projection. It remains at
`results/stable_v1_failed_float64_unfixed_blas/`. Fixing the internal BLAS
profile in fitted preprocessing also freezes the reduction order; every
distinct training cell set is checked again.
The tolerance was not widened. Historical scores use different precision and
drug-ID folds; they are migration diagnostics, not the golden reference for
the revised numerical recipe and chemical-disjoint evaluation.

The executed notebook is copied to `notebooks/stable_v1_results.ipynb`, and its
compact report to `data/reports/stable_v1_summary.json`. Regenerating figures
needs neither raw data nor GPU; select a Python kernel with the locked notebook
dependencies. `final_results.ipynb` preserves the earlier, broader exploratory
roadmap checkpoint, including modality and annotation controls.

## Release and future changes

`scripts/freeze_stable_v1.py` verifies the run, creates separate local raw-input
and fitted-artifact ZIP archives, reads every member back and checks hashes.
`data/reports/stable_v1_release.json` records archive checksums. Code, configs
and reports are frozen by local tag **v0.2.0-stable**. Archives are ignored by
Git and are local copies, not an off-device backup. Datasets are not published.

Preserve this tag; use new output directories for changes. New architectures,
curation policies, splits or endpoints require new validation. Overlapping
training folds are not independent studies. Entity bootstrap conditions on
fitted models. A shared dose window does not resolve unknown exposure duration
or assay biology. Poor simultaneous coverage in PRISM remains a negative
scientific result; the release does not conceal it or tune on test to fix it.
