# Results notebooks

The current release is [stable_v1_results.ipynb](stable_v1_results.ipynb). It
uses `data/reports/stable_v1_summary.json` and needs neither raw inputs nor GPU
for plotting. See the [stable protocol](../docs/stable_v1.md) for full retraining.
It contains 14 executed figures and exports PNG (300 dpi), SVG and PDF under
the repository's ignored `figures/stable_v1/` directory. Its numerical reference
uses float64 scaling/PCA; the earlier float32 checkpoint is kept separately.
The new exploratory follow-up is
[`improvement_v1_results.ipynb`](improvement_v1_results.ipynb), with cold-drug,
neural learning, error diagnostics, original source-only transfer and separately
labelled PRISM-supervised adaptation. Its portable snapshot needs no GPU.
See [its protocol](../docs/improvement_v1.md) before comparing adaptation results
with the original transfer experiment.

Select the locked Python kernel for the stable notebook, or execute explicitly:

```powershell
python -m nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3 --inplace notebooks/stable_v1_results.ipynb
```

## Earlier exploratory checkpoint

Open [`final_results.ipynb`](final_results.ipynb). The executed notebook contains
16 figures with Spanish captions and interpretations, plus complete metric and
provenance tables. It covers the final roadmap 2–6 checkpoint: all 23 refreshed
model/variant comparisons, five B0 references, permutations, corrected target
views, frozen B7, dose-window harmonization, assay transfer, chemical matching
sensitivity, bootstrap comparisons and independent calibration.

Historical architectures/splits that still need corrected retraining are
excluded from the refreshed ranking. The notebook explicitly distinguishes
fold standard deviations from bootstrap intervals, mean fold RMSE from pooled
RMSE, row coverage from complete-group coverage and published from harmonized
AUC. These are exploratory preclinical results.

## Reproduce the figures

From a Python environment inside the repository:

```powershell
pip install -e ".[notebooks]"
python -m nbconvert --to notebook --execute --inplace notebooks/final_results.ipynb
```

Alternatively, open the notebook in a Jupyter-compatible editor and run all
cells. Select the Python kernel of the environment containing those extras.
Run from the repository root or `notebooks/`; the notebook locates the root.
Raw data, model weights, XGBoost, PyTorch and a GPU are unnecessary for plotting.
If large experiment artifacts are present, their recorded hashes are checked.
The compact snapshot is also checked against the canonical fold aggregates.

The notebook exports:

- `figures/final_results/01_*.png` through `16_*.png`: 300 dpi PNG figures.
- Matching SVG and PDF figures with editable/vector text.
- Complete metrics, 105 bootstrap comparisons, calibration diagnostics and
  source-artifact SHA-256 values as CSV, plus a figure manifest.

The verified local execution also produced the self-contained, code-free
`results/reports/final_results/final_results.html`. To export a new HTML report:

```powershell
python -m nbconvert --to html --no-input --output final_results --output-dir results/reports/final_results notebooks/final_results.ipynb
```

Generated figures and HTML remain ignored by Git; the notebook's executed
outputs and compact plotting inputs are versioned.

## Update after new experiments

Regenerate the canonical final report first, preserving the existing source
and prediction-integrity checks. Then rebuild the compact plotting snapshot:

```powershell
python scripts/export_final_results_plot_data.py
```

This export command requires the local final experiment artifacts, including
the corrected B0 fold metrics/predictions. It checks five-fold coverage,
nonduplicated identities, identical outer-test targets and complete 149,364-pair
coverage for the three prediction-density maps. Those maps include every pair
using identical 70×70 bins without trimming outliers or aesthetic sampling.
It records hashes of every source file used by the snapshot.

Reexecute the notebook after rebuilding. Do not present a stale canonical
report alongside new predictions. See the [research roadmap](../docs/research_roadmap.md)
for remaining experiments and the scope of the current scientific claims.
