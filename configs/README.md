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

1. `experiments/splits_full.json`
2. `experiments/b4_gnn_full.json`
3. `experiments/b4_tuning_random_pair.json`
4. `experiments/b4_tuning_full.json`
5. `experiments/b5_hybrid_random_pair.json`
6. `experiments/b5_hybrid_full.json`
7. `experiments/b5_tuning_random_pair.json`
8. `experiments/b5_tuning_full.json`
9. `experiments/b6_ablation_random_pair.json`
10. `experiments/b6_ablation_full.json`
11. `experiments/drug_embeddings_chemberta.json`
12. `experiments/b7_pretrained_random_pair.json`
13. `experiments/b7_pretrained_full.json`
14. `experiments/b7_tuning_random_pair.json`
15. `experiments/b7_tuning_full.json`
16. `experiments/reports_f3_f4.json`

B4/B5 tuning searches a 4-point grid of learning rate × dropout and selects on
validation RMSE. B6 ablations reuse the validation-selected B2 XGBoost
hyperparameters when `results/baselines/b2_tuning_metrics.csv` exists. Full
configs include `cold_tissue`.

The B7 configs require an external embedding table at
`data/external/drug_embeddings.csv`; see `docs/external_embeddings.md` for the
expected format.
