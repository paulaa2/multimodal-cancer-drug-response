# Development Workflow

Environment setup, git conventions, and the quality gates. The guiding
constraint is that this is a small research repository with one author: the
workflow should keep history readable and results traceable without imposing
process that nobody needs.

## Environment

The package uses a `src/` layout, so it must be installed to be importable as
`mcdrp`. Do this once per environment:

```powershell
conda create -n mcdrp python=3.12
conda activate mcdrp
pip install -e ".[dev,baselines]"
```

Extras:

| Extra | Brings | Needed for |
| --- | --- | --- |
| `dev` | pytest, ruff, mypy | tests and lint |
| `baselines` | rdkit, xgboost | fingerprints, scaffold splits, B1/B2/B6/B7 |
| `gpu` | torch | B3 torch backend, B4, B5 |
| `embeddings` | torch, transformers | generating B7 drug embeddings |

### Module paths

Always `python -m mcdrp.<module>`, never `python -m src.mcdrp.<module>`.

The `src.` prefix appears to work and then fails confusingly. Python resolves
`src` as a namespace package, so `src.mcdrp.splits.make_splits` imports fine —
it has no internal imports. But `src.mcdrp.models.ablation_b6` does
`from mcdrp.features.expression import ...`, which needs `mcdrp` importable
anyway. The result is either an `ImportError` partway through a pipeline or, if
the package *is* installed, two copies of every module loaded under different
names. `mcdrp.experiments.run_experiment` now rejects `src.`-prefixed module
paths in configs so this cannot recur silently.

If `python -m mcdrp...` raises `ModuleNotFoundError`, the package is not
installed in the active environment. Reinstall with `pip install -e .` rather
than working around it with a path prefix.

## Quality gates

The same three checks run locally and in CI:

```powershell
ruff check src tests
mypy
pytest
```

Optionally install the pre-commit hooks so lint runs before each commit:

```powershell
pip install pre-commit
pre-commit install
```

Notes:

- Tests that need PyTorch skip themselves when it is absent, so the suite is
  runnable without the `gpu` extra. CI does not install torch, which means the
  B4/B5 collate tests are skipped there and must be run locally.
- `mypy` sets `no_site_packages`. The stubs shipped inside the RDKit wheel
  contain a syntax error that aborts the whole run, and a syntax error in a stub
  cannot be suppressed per-module. The cost is that third-party signatures are
  untyped; our own annotations are still checked.
- The codebase is not `ruff format`-clean. Reformatting is worth doing, but it
  should be one dedicated commit that touches nothing else, so that it stays
  reviewable.

## Branching

History is linear and there is one author, so use trunk-based development with
short-lived topic branches. `main` should always be green and always
installable.

```text
main ──●──●──────●──────●──  (always green, tagged at milestones)
        \        /
         ●──●──●           feat/pathway-encoder
```

Branch naming, by intent:

| Prefix | For | Example |
| --- | --- | --- |
| `feat/` | new capability | `feat/multi-seed-runner` |
| `fix/` | bug fix | `fix/tuning-selection-empty-frame` |
| `exp/` | an experiment that requires code changes | `exp/b8-multiomics` |
| `docs/` | documentation only | `docs/evaluation-protocol` |
| `chore/` | tooling, CI, refactors | `chore/ruff-format` |

Rules that matter for this project:

- **One concern per branch.** The current `gnn_test` branch accumulated scaffold
  splits, experiment configs, and pathway ablations in a single commit. That is
  three reviewable changes and a name that describes none of them.
- **Delete the branch after merging.** Long-lived feature branches are what
  produced the drift this repository is recovering from.
- **Do not branch per experiment run.** An experiment that only changes
  hyperparameters or splits is a config in `configs/experiments/` plus a run
  manifest, not a branch. Branch only when code changes.
- **Rebase on `main` before merging** to keep history linear.

### Milestones

Tag `main` when a project phase completes, and let the tag carry the results:

```powershell
git tag -a v0.1-baselines -m "B0-B3 baselines across the full split suite"
git push origin v0.1-baselines
```

Tags are the right tool for "the state that produced these numbers", because
`data/reports/` summaries are tracked in git and are therefore recoverable from
any tag. Use a GitHub Release for phases worth writing up.

## Commit messages

Use an imperative summary under ~70 characters, and explain *why* in the body
when the reason is not obvious from the diff. Existing messages such as `audit`,
`baselines`, and `audit modified` are not recoverable six months later.

```text
Add mean-effects baseline and normalized metrics

Global Pearson on random_pair reaches 0.906 with no features at all, so
the previous B0 variants were too weak to be a meaningful reference.
Normalized metrics separate differential response from mean effects.
```

## What belongs in git

| Path | Tracked | Why |
| --- | --- | --- |
| `src/`, `tests/`, `configs/` | yes | the project |
| `data/reports/*.json` | yes | small run summaries, make results traceable |
| `data/mappings/` | yes | identifier mappings are a project artifact |
| `data/raw/`, `data/processed/`, `data/external/` | no | large, regenerable, or licensed |
| `results/`, `figures/` | no | regenerable from configs plus code |

The `check-added-large-files` pre-commit hook guards this boundary at 512 KB.
