# Research roadmap

The frozen software/evaluation release is [stable v1](stable_v1.md), covering
B0/RF/B6/B7, five full-chemical-identity cold-drug folds and strict GDSC→PRISM transfer. Reproduce
that release before extending architectures. The broader exploratory controls
below remain a separate checkpoint; historical architectures are excluded
from the stable ranking.

Stable v1 uses float64 scaling/PCA with float32 final features, after a failed
float32 reproduction and a data-only numerical audit. Its hash-pinned reference
belongs to that revised recipe; see the [acceptance criteria](stable_v1.md).
Historical float32 scores remain a separate checkpoint.
The historical drug-ID folds also allow nine chemical alias pairs to cross
subsets; they cannot establish unseen-chemical performance. The stable
protocol groups all 238 IDs into 229 complete chemical identities for folds,
calibration and bootstrap, while preserving every measured pair.

This is the authoritative list of project status and next experiments. The
[evaluation protocol](evaluation_protocol.md) defines measurement;
[related work](related_work.md) supplies the literature rationale;
[project plan](project_plan.md) is a command/reference guide to implemented phases.

## Audit checkpoint — 2026-09-30

The full source tree and existing artifact contracts were reviewed, with unit
checks, static checks and real-data diagnostics. See
[repository audit](repository_audit.md). Code passing tests is not proof that
all scientific conclusions are correct.

Historical results remain available, but expression-based comparisons predate
the default-profile correction. GNN comparisons also predate the initialization
seed correction; B3 predates grouped external early stopping. Do not mix these
artifacts with refreshed predictions in one ranking.

## Implemented and executed in this audit

1. **Data and scoring integrity.** Select the default DepMap profile for each
   model; reject missing/duplicate profiles, incomplete split assignments,
   duplicate/stale predictions, inconsistent targets, wrong subsets and
   nonfinite scores. Allow `unused` in the double-cold audit and independent
   calibration reservation; verify tissue/scaffold and validation–test group
   separation, including calibration groups when present.
2. **Reproducible fitting.** Seed GNN construction; give B3 the declared
   validation groups in both backends; prevent ElasticNet tuning from mutating
   a shared feature matrix. B1/B2/B3/B7 tuners now persist validation-selected
   predictions for normalized reporting. Experiment manifests survive failures
   and contain config snapshots, package versions and source hashes.
3. **Annotation coverage diagnostic (§7b of the previous roadmap).** Score
   held-out drugs as covered, unannotated, or annotated only with tokens absent
   from that fold's training vocabulary. Report both pair-weighted squared-error
   aggregation and mean drug deltas; they are different estimands.
4. **Assay concordance (§8c).** Match the same cells within each drug and report
   Pearson/Spearman and published-AUC offsets. Low concordance is a diagnostic,
   not a mathematical upper bound on model accuracy. Negative R² diagnoses
   absolute prediction error; Spearman diagnoses ranking.
5. **Transfer fitting replication (§8b).** Three seeds, fixed GDSC partition,
   strict unambiguous InChIKey matching and default expression. Targets, Morgan
   and a whole-drug target permutation share model settings. Fitting replication
   is not independent external validation.
6. **Refresh the main LDO comparison.** Five existing cold-drug folds, target +
   PCA and Morgan + PCA against B0, with explicitly fixed settings. This checks
   the central claim after correcting preprocessing. Other architectures and
   splits remain historical until retrained.

## Requested steps 2–6 — implemented and executed

The [implementation and commands](roadmap_2_6.md) define the protocols. The
canonical machine-readable evidence is
[`roadmap_2_6_summary.json`](../data/reports/roadmap_2_6_summary.json);
prediction artifacts are under `results/roadmap_2_6`. This checkpoint supersedes
the pending endpoint, baseline, modality, provenance and uncertainty tasks
listed before this execution. Historical artifacts retain their original status.

2. **Shared-window endpoints.** Verified raw GDSC/PRISM dose-response files,
   explicit dose units, decreasing curves and integration over log10 µM
   [-1.9, 0.9], without extrapolation. Retained published AUC separately and
   fitted both endpoints on identical eligible pairs. The harmonized cohort
   contains 9,197 training, 1,550 validation and 10,785 PRISM test pairs.
   Exact per-curve exposure durations are unavailable; nominal assay durations
   are documented rather than treated as a time correction. On the same 22
   drugs/4,548 cell–drug pairs, mean drug Spearman rises from 0.343 to 0.367,
   while absolute between-assay AUC discrepancy increases. Harmonization does
   not remove assay shift; absolute RMSE across endpoint definitions is not a
   fair improvement comparison.
3. **Simple baselines and independent metric check.** Validation-selected
   random forests with targets or Morgan plus training-fitted expression PCA,
   all five cold-drug folds. Mean target-RF RMSE is 2.434, against 2.515 for the
   refreshed target XGBoost and 2.662 for B0 mean effects. These are exploratory
   test comparisons, not selection on an untouched confirmatory cohort.
   Official DrEvalPy 1.5.1 checks 450 global metric values: 410 agree numerically
   and 40 differ only by the explicitly recorded constant-correlation convention.
   This checks metric interoperability and group separation, not independent
   retraining of the whole DrEval benchmark.
4. **Modality controls.** Three seeds of consistent whole-cell and whole-drug
   feature permutations, restricted by partition-membership strata, for both
   target and Morgan models on all five folds. Donor identity and unchanged
   feature vectors are reported. Five graph-only fits ignore expression and
   yield mean RMSE 2.726. Null results diagnose representation dependence;
   they do not establish causal mechanisms.
5. **Annotation and representation provenance.** Frozen HGNC snapshot,
   separate genes/families/mechanisms/unresolved tokens, three refreshed target
   views and explicit missingness. Gene-only RMSE is 2.655, mechanism/family
   RMSE 2.537 and typed-view RMSE 2.491; collapsing everything to genes loses
   predictive information. ChemBERTa embeddings have a resolved checkpoint
   commit, source/output hashes and a successful audit; B7 was refreshed on
   all five folds (RMSE 2.632). Full-InChIKey-only transfer sensitivity retains
   the target advantage on 44 drugs/13,597 pairs. HGNC aliases are nomenclature
   matches, not experimental binding evidence; pharmacological specificity
   still needs expert curation.
6. **Independent calibration and uncertainty.** Reserved training entities
   without using responses; fitted models/early stopping exclude them. Executed
   five cold-drug calibration fits and harmonized transfer calibration, reporting
   raw and validation-affine predictions, group-max residual intervals and
   paired drug bootstrap comparisons. PRISM raw interval coverage by complete
   cell is only 17.9% for Morgan and 24.8% for targets, despite row coverage of
   93.4% and 94.6%. A nominal 90% simultaneous group guarantee is unsupported
   under this shift. Stratified diagnostics and interval widths are retained.

Validation: 182 tests passed; Ruff and mypy passed. Source, prediction identity,
annotation hashes and complete five-fold coverage are checked before reporting.

## Next priorities

Repeat the modality permutations and annotation-view controls on the frozen
chemical-identity folds before transferring their historical conclusions to
the stable comparison. Their previous drug-ID results remain exploratory.

1. **Refresh remaining comparisons.** Rerun corrected `cold_scaffold`,
   `cold_cell`, `cold_tissue`, B3/B4/B5 and random-pair replications in isolated
   output directories. Keep hyperparameter selection inside each fold, including
   PCA dimension if it is tuned. Do not reuse a tuning CSV solely because its
   split name matches: the actual training/validation identities must match.
2. **Biological representations and interpretation.** Hallmark/Reactome GMT
   files work with the current unweighted mean-expression scorer. PROGENy needs
   signed weighted perturbation footprints and a separate implementation; it
   cannot be reproduced by passing its genes as an ordinary GMT. mRNA-target
   products do not imply a mechanistic sensitivity law.
3. **Additional omics and target evidence.** Mutation/CNV views should use
   frozen releases and train-only fitting, evaluated with ablations. Curate
   pharmacological family specificity against primary binding data before
   mechanistic interpretation. Predicted targets need a documented source and
   validation protocol; failure of one Ridge map does not rule out the approach.
4. **Fresh external confirmation and calibration under shift.** Select the
   protocol using development data, then obtain an untouched external screen.
   Investigate transfer calibration with a separately reserved target-domain
   sample and explicit assumptions; the current PRISM result supplies a
   negative coverage diagnostic, not an interval guarantee.

## Claim limits

No clinical, causal or universal architecture claims. Five overlapping folds
are not five independent studies. The held-out PRISM screen has already been
examined; further adaptations to its labels would require a fresh external
cohort for confirmation. A useful negative result and a strong simple baseline
are valid outcomes.
