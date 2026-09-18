# Experiment configs

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
