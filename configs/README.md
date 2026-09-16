# Experiment configs

JSON files in this directory define reproducible experiment pipelines. Run them
with:

```powershell
python -m src.mcdrp.experiments.run_experiment configs/experiments/<config>.json --dry-run
python -m src.mcdrp.experiments.run_experiment configs/experiments/<config>.json
```

Use `--dry-run` first to inspect the exact commands before launching expensive
GPU training.

Suggested order:

1. `experiments/splits_full.json`
2. `experiments/b5_hybrid_random_pair.json`
3. `experiments/b5_hybrid_full.json`
4. `experiments/b6_ablation_random_pair.json`
5. `experiments/b6_ablation_full.json`
6. `experiments/reports_f3_f4.json`
