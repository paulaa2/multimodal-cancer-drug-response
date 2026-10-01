# Experiment configs

`stable_v1.json` is the frozen release contract. Its dedicated runner rebuilds
raw preprocessing, train-fitted features, models, test reports and notebook in
a new output directory. See [stable v1](../docs/stable_v1.md). Configurations
under `experiments/` also include historical/experimental architectures and
are not all part of this release.

JSON files in this directory define reproducible experiment pipelines. Run them
with:

```powershell
python -m mcdrp.experiments.run_experiment configs/experiments/<config>.json --dry-run
python -m mcdrp.experiments.run_experiment configs/experiments/<config>.json
```

Use `--dry-run` first to inspect the exact commands before launching expensive
GPU training.

Suggested order:

Smoke tests (`*_random_pair.json`) check that a pipeline runs. They are not
the evaluation result. Hyperparameters are selected independently on each
split; a setting chosen on `random_pair` cannot be reused for a cold-start
claim. The `*_full.json` configs are the ones that count.

1. `experiments/splits_full.json`
2. `experiments/b1_tuning_full.json`
3. `experiments/b2_tuning_full.json`
4. `experiments/b3_tuning_full.json`
5. `experiments/b4_gnn_full.json`
6. `experiments/b4_tuning_full.json`
7. `experiments/b5_hybrid_full.json`
8. `experiments/b5_tuning_full.json`
9. `experiments/b6_ablation_full.json`
10. `experiments/drug_embeddings_chemberta.json`
11. `experiments/b7_pretrained_full.json`
12. `experiments/b7_tuning_full.json`
13. `experiments/reports_f3_f4.json`
14. `experiments/multiseed_b0_b2_full.json`

B4/B5 tuning searches a 4-point grid of learning rate × dropout and selects on
validation RMSE **per split**. B6 ablations reuse the validation-selected B2
XGBoost hyperparameters for that same split when
`results/baselines/b2_tuning_metrics.csv` exists, so B2 must be tuned on every
split that B6 will report. Full configs include `cold_tissue`.

The `*_random_pair.json` configs remain as optional smoke tests.

Multi-seed configs rebuild splits under seeds 42, 7, and 13 and retrain B0–B2
with the same seed for initialization. They write mean and spread to
`results/multiseed/` and do not compute p-values. Canonical seed-42 artifacts
are left in place.

The B7 configs require an external embedding table at
`data/external/drug_embeddings.csv`; see `docs/external_embeddings.md` for the
expected format.

## Corrected audit checkpoint

`transfer_prism_multiseed.json` builds the strict transfer cohort and evaluates
three fitting seeds in `results/transfer_strict/`. Run it before
`cold_drug_checked.json`, whose final diagnostics read that strict cohort.
The cold-drug config refreshes five existing folds in `results/cv_checked/`
with default expression profiles and fixed model settings. It needs folds built
by `mcdrp.splits.grouped_cv`, and explicitly uses `--no-b2-tuned`.

These configs and small JSON reports preserve the exploratory audit checkpoint. Other configs
remain executable reference pipelines; their old output numbers have not all
been refreshed. Reusing a tuning file requires identical training/validation
pair identities, not just a matching split name. The selected prediction files
from B1/B2/B3/B7 tuning now support the normalized metrics report.

B6 enforces this using B2's `selection_fingerprint`, including selection
targets, expression-file hash, PCA settings and default-profile policy. Old
tuning CSVs without that provenance are rejected; rerun B2 tuning or explicitly
use `--no-b2-tuned` with prespecified settings.

`configs/data_audit.example.json` is the default raw-data audit config.
Use `python -m mcdrp.data.audit --no-write-templates` when retaining curated
mapping files. Experiment manifests include the actual config snapshot and
source hashes, including per-step hashes; dry runs have a separate manifest.

## Roadmap 2–6

`public_sources.json` freezes raw GDSCv2/PRISM inputs and the local HGNC snapshot
by checksum. `harmonization.json` declares units, MTS010 eligibility, the shared
log-dose integration window and output paths. See
[`docs/roadmap_2_6.md`](../docs/roadmap_2_6.md) for assumptions and ordered commands.

`roadmap_graph_only.json` evaluates the graph-only control on cold-drug folds;
`roadmap_b7_pinned.json` requires the verified pinned embedding manifest. The
`mcdrp.experiments.roadmap` phases execute five-fold RF selection, three seeds
per entity modality null, target-view comparisons and independent calibration.
Each phase has isolated outputs and input/code hashes. Changed inputs/code need
a new output root rather than a mixed resumed ranking.
