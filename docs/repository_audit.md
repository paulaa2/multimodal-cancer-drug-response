# Repository audit — 2026-09-30

## Outcome and scope

The repository contains a functioning B0–B7 model ladder, six GDSC evaluation
splits, grouped outer folds and a GDSC-to-PRISM assay-transfer workflow. The
audit reviewed data assembly, features, fitting/tuning, split construction,
metrics, reports, configurations, tests and scientific claims. Several defects
affected reproducibility or evaluation integrity; these were corrected and the
central target-versus-structure comparisons were rerun on real data.

This is a code and research-protocol checkpoint, not proof that every model is
correct or every historical result is reproducible. Full B3/B4/B5 retraining,
all other cold splits, external embedding provenance and endpoint harmonization
remain explicit work in the [roadmap](research_roadmap.md). No existing local
changes or historical result directories were discarded.

## Corrections that affect results

- **Expression identity.** The cohort's 717 DepMap models had 738 expression
  records, including 21 alternative profiles. Previously, PCA included the
  duplicate profiles and the final model-to-feature mapping depended on CSV
  order. The loader now selects the declared default profile, requires unique
  model IDs and rejects missing or ambiguous requested models. PCA/scaling
  remain fitted on training cells only. Expression-based historical comparisons
  need refreshed predictions.
- **GNN initialization.** B4/B5 now seed torch before constructing model
  weights. Setting a seed only inside the later training function did not make
  initialization reproducible. Historical graph-model rankings require reruns.
- **B3 early stopping.** Both sklearn and torch backends now use the split's
  declared validation groups. The sklearn backend trains with `partial_fit`
  and restores the best validation checkpoint; it no longer makes an internal
  random pair split when a cold validation set was supplied. A standalone fit
  without external validation still supports the documented internal fallback.
- **Linear tuning.** ElasticNet uses `copy_X=True`, preventing one candidate
  from changing a shared design matrix used by subsequent candidates.
- **Chemical identity.** PRISM full-InChIKey and connectivity lookups reject
  duplicated GDSC identities rather than choosing whichever record occurred
  first/last. Comma-separated PRISM structures are also rejected when they
  encode multiple InChIKeys; repeated identical structures remain valid.
  The strict external cohort contains 44 unique full-key matches
  and two unique connectivity matches. Connectivity ignores stereochemistry;
  a full-key-only sensitivity analysis remains useful. PRISM matching provenance
  is preserved in the assembled cohort.
- **Prediction persistence.** B1/B2/B3/B7 tuners now export the selected
  model's validation/test predictions with the `tuned` variant. Previously,
  normalized-report instructions promised files these tuners did not write.

## Integrity and maintainability

- A shared split-assignment validator checks unique nonnull pair IDs, known
  subset labels and exact cohort coverage. Model loaders and downstream reports
  reuse this contract instead of silently accepting incomplete merges.
- The artifact validator now understands `unused` pairs in double-cold splits,
  verifies tissue/scaffold separation and checks validation–test as well as
  training overlaps. Cold group CV requires at least three folds, so training
  cannot disappear when validation and test each consume one fold.
- Metrics reject nonfinite, empty, mismatched or non-vector inputs. The full
  report also rejects duplicate/stale predictions, wrong target values, wrong
  subsets and missing evaluation pairs. Fold and seed summaries reject
  duplicate model rows and incomplete comparisons; shuffled variants stay
  separate.
- `B0_mean_effects` is the prespecified reporting reference whenever present.
  Choosing a reference after examining test scores would alter the comparison.
  Test-set rankings are descriptive and must not select the next model.
- B2 tuning records a selection fingerprint over train/validation pair labels,
  targets and feature identities, plus expression-file hash, PCA settings and
  default-profile policy. B6 checks that fingerprint before reusing parameters;
  legacy or mismatched tuning requires a rerun or explicit `--no-b2-tuned`.
  Test labels do not contribute to this fingerprint. Saved tuning parameters
  also retain maximum rounds and stopping patience.
- External embedding blocks must be finite and requested cell-feature maps
  complete. Morgan fingerprints now use RDKit's current generator and cache
  unique SMILES. Molecular graph charge features handle both bounds; graph
  tests verify padding invariance and finite, nonzero gradients.
- Raw-data auditing now has an actual default config, records file size and
  SHA-256, writes to `data/reports/`, and supports `--no-write-templates` to
  avoid replacing curated mappings. The five configured raw inputs were found.
- Experiment manifests record config snapshots/hashes, source hashes including
  local edits, package/Python versions and step status. Failed runs preserve
  diagnostics; dry runs write a separate manifest. Changes made after a run
  started do not retroactively change its embedded configuration.

## Refreshed experiments

### Leave-drug-out: five existing folds

The checked config uses the same outer-fold identities, default expression
profiles, 256-component training-fitted PCA and fixed XGBoost settings
(`700` estimators, depth `6`, learning rate `0.05`, validation patience `40`).
It explicitly disables reuse of tuning artifacts from another partition. This
is a controlled refresh, not a complete nested hyperparameter search.

- B0 mean effects: mean fold RMSE **2.6618**.
- Targets + expression PCA: **2.5199**, paired mean delta **−0.1419**, better
  RMSE in **5/5** folds; mean normalized R² **0.1056**.
- Morgan + expression PCA: **2.7355**, paired mean delta **+0.0737**, better
  RMSE in **1/5** folds; mean normalized R² **−0.0648**.

This supports an absolute-error advantage for this annotated-target feature
block under this protocol. It does not support better within-drug ranking than
B0: mean within-drug Spearman is **0.6086** for targets versus **0.6228** for
B0. Within-drug R² remains negative. Target annotations include mechanism
classes as well as gene names; the result is not a causal biomarker claim.

Coverage diagnostics test all **238** drugs exactly once. Of these, **141**
have at least one annotation token in their fold's training vocabulary, **18**
have no annotations, and **79** only have unseen tokens. Pair-weighted target
RMSE deltas against B0 are **−0.2546**, **−0.0097** and **+0.0141**, respectively.
The mean of per-drug deltas differs from the pair-weighted estimand and is
reported separately. These strata are descriptive, not randomized interventions.

### PRISM transfer: strict cohort, three fitting seeds

Only GDSC trains/validates the models; PRISM supplies held-out published AUC.
The fixed external cohort has **46 drugs, 335 cells and 14,200 pairs**.
The same GDSC partition is used with seeds **42, 7 and 13**.

- B0 mean effects: mean RMSE **0.14508**.
- Targets + PCA: **0.13722 ± 0.00024**, improving RMSE by **5.4%** and winning
  **3/3** fitting seeds; normalized R² **0.1054**.
- Morgan + PCA: **0.13951 ± 0.00011**, winning **3/3** seeds against B0.
- Whole-drug target permutation: **0.14686 ± 0.00478**, winning **1/3** seeds.

The ± values are sample standard deviations across fitting/permutation seeds,
not external-assay confidence intervals. Targets retain modest within-drug
ranking (mean Spearman **0.2115**) despite negative within-drug R² (**−0.2026**).
The earlier approximately 8% improvement came from a different, ambiguous
49-drug cohort and must not be compared as if only the model changed.

Measured GDSC/PRISM concordance uses **13,263 shared measured pairs**, fewer
than the full transfer test because some cell–drug pairs have no GDSC label.
Across the 46 drugs, median Spearman is **0.2943** and median Pearson **0.3726**.
This exposes assay disagreement; it is not a mathematical upper bound on
prediction quality. Published AUC values have not been integrated over a common
concentration window.

## Evidence and reproduction

Small tracked checkpoints:

- `data/reports/repository_audit_summary.json`: verification/runtime checkpoint.
- `data/reports/repository_audit_validation.json`: cohort and six split audits.
- `data/reports/config_validation_summary.json`: 23 configs / 84 parsed commands.
- `data/reports/cold_drug_checked_summary.json`: paired fold metrics.
- `data/reports/transfer_prism_strict_summary.json`: strict mapping/cohort counts.
- `data/reports/transfer_prism_replication_summary.json`: per-seed scores and deltas.
- `data/reports/transfer_diagnostics_summary.json`: concordance and coverage.
- `data/reports/data_audit_report.json`: frozen local input fingerprints.

Large generated tables, predictions and logs remain ignored. Corrected outputs
live in `results/cv_checked/`, `results/transfer_strict/` and
`results/reports/diagnostics/`, separate from historical results. Intermediate
49-drug/default-profile refreshes are not the final strict comparison.

With the raw inputs and original grouped folds available:

```powershell
python -m mcdrp.experiments.run_experiment configs/experiments/transfer_prism_multiseed.json
python -m mcdrp.experiments.run_experiment configs/experiments/cold_drug_checked.json
```

Run transfer first: the cold-drug config's final diagnostics read its strict
cohort. The original fold directories are built by `mcdrp.splits.grouped_cv`.
The final aggregation/diagnostic steps were also run separately after training;
the training manifests retain the exact configs used when those runs began.

Local quality gates passed: **167 tests with no skips**, `ruff check src tests`,
`mypy` on **52 source files**, and `git diff --check`. Config parsing checked
**23 configs / 84 commands**. The full local test run used Python 3.11 with
torch 2.9.1; XGBoost 3.2/CUDA experiments used Python 3.13. No global environment
was modified. CI now
installs CPU torch on its Python 3.12 matrix entry to keep graph regressions
  covered; this CI configuration has not yet run on GitHub in this audit.

An additional CPU smoke run on 180 real cohort rows exercised B2 tuning/export
and B6 parameter reuse. Matching provenance was accepted; changing the training
partition was rejected before fitting. Current strict transfer prediction IDs,
targets and saved RMSE were also rechecked for all three seeds after the final
cohort rebuild. The optional-torch-absent suite passed 156 tests, with 11 neural
or neural-config checks skipped as designed.

## Conceptual boundaries and literature

Lower error, ranking and calibration are distinct outcomes. Negative R² does
not mean zero ranking. Normalized R² is `1 − SSE_model / SSE_reference`, not
the ratio of centered residual variances. Mean-effect residuals are a
statistical decomposition, not isolated biological interactions. Likewise,
mRNA/target products and GMT means are exploratory representations, not a
validated mechanistic law or PROGENy activity.

The priority is common dose-window endpoint harmonization, guided by
[Chen et al. (2026)](https://doi.org/10.1093/bioinformatics/btag382) and
[Xia et al. (2022)](https://doi.org/10.1093/bib/bbab356). The PRISM protocol
comes from [Corsello et al. (2020)](https://doi.org/10.1038/s43018-019-0018-6).
Faithful PROGENy would require signed weighted footprints as in
[Schubert et al. (2018)](https://doi.org/10.1038/s41467-017-02391-6).
See [related work](related_work.md) for the reading rationale and the
[roadmap](research_roadmap.md) for the single authoritative next-step list.
