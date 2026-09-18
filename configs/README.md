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
2. `experiments/b5_hybrid_random_pair.json`
3. `experiments/b5_hybrid_full.json`
4. `experiments/b6_ablation_random_pair.json`
5. `experiments/b6_ablation_full.json`
6. `experiments/drug_embeddings_chemberta.json`
7. `experiments/b7_pretrained_random_pair.json`
8. `experiments/b7_pretrained_full.json`
9. `experiments/b7_tuning_random_pair.json`
10. `experiments/b7_tuning_full.json`
11. `experiments/reports_f3_f4.json`

The B7 configs require an external embedding table at
`data/external/drug_embeddings.csv`; see `docs/external_embeddings.md` for the
expected format.
