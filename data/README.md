# Data Directory

Large datasets should not be committed to git.

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
