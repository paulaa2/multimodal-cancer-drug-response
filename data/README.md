# Data Directory

Large datasets should not be committed to git.

The frozen v1 input contract is `configs/stable_v1.json`. Its new derived
datasets live inside `results/stable_v1/data/`, with a verified local raw-input
archive under `results/releases/stable_v1/`. See `docs/stable_v1.md` for exact
reproduction. Earlier `data/processed/` files remain exploratory checkpoints;
the stable runner reconstructs its own cohorts and splits from frozen raw files.

Recommended local layout:

```text
data/
├── raw/          # downloaded source files, ignored by git
├── external/     # third-party resources such as gene sets, ignored by git
├── interim/      # partially cleaned files, ignored by git
├── processed/    # model-ready matrices/tables, ignored by git
├── reports/      # small audit/cohort/split summaries, tracked by git
└── mappings/     # identifier mapping tables and audit reports, tracked by git
```

The first real project milestone is a data audit that creates explicit mapping
tables with status flags. Avoid silent fuzzy matching.

Every generated dataset should have a short JSON summary in `data/reports/`.
