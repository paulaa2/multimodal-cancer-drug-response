from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mcdrp.experiments.multiseed import (
    DEFAULT_SEEDS,
    drop_forbidden_columns,
    load_seed_metrics,
    resolve_models,
    seed_layout,
    steps_for_seed,
    summarize_multiseed,
)
from mcdrp.experiments.run_experiment import build_command
from mcdrp.splits.make_splits import SplitSpec, assign_labels


def test_default_seeds_include_the_canonical_seed() -> None:
    assert DEFAULT_SEEDS[0] == 42
    assert len(DEFAULT_SEEDS) >= 3


def test_split_labels_change_when_the_seed_changes() -> None:
    spec_a = SplitSpec(validation_size=0.2, test_size=0.2, seed=42)
    spec_b = SplitSpec(validation_size=0.2, test_size=0.2, seed=7)

    labels_a = assign_labels(20, spec_a, np.random.default_rng(spec_a.seed))
    labels_b = assign_labels(20, spec_b, np.random.default_rng(spec_b.seed))

    assert list(labels_a) != list(labels_b)


def test_steps_for_seed_vary_split_and_model_seeds() -> None:
    layout = seed_layout(7, split_root="data/processed/splits", output_dir="results/multiseed")
    steps = steps_for_seed(
        layout,
        resolve_models(["B0", "B1", "B2"]),
        split_names=("random_pair", "cold_tissue"),
        device="cuda",
    )

    commands = {step.name: build_command(step) for step in steps}
    split_cmd = commands["splits_seed_7"]
    assert "--seed" in split_cmd
    assert split_cmd[split_cmd.index("--seed") + 1] == "7"
    assert "seed_7" in split_cmd[split_cmd.index("--output-dir") + 1]

    b0_cmd = commands["b0_seed_7"]
    assert "--random-state" not in b0_cmd
    assert "seed_7" in b0_cmd[b0_cmd.index("--split-dir") + 1]
    assert "seed_7" in b0_cmd[b0_cmd.index("--output") + 1]

    b2_cmd = commands["b2_seed_7"]
    assert b2_cmd[b2_cmd.index("--random-state") + 1] == "7"
    assert "--device" in b2_cmd
    assert "cold_tissue" in b2_cmd


def test_b6_does_not_reuse_another_seed_tuning_file() -> None:
    layout = seed_layout(13)
    steps = steps_for_seed(
        layout,
        resolve_models(["B6"]),
        split_names=("random_pair",),
        device="cpu",
    )
    command = build_command(steps[-1])
    assert "--no-b2-tuned" in command
    assert "--random-state" in command


def test_summarize_multiseed_reports_mean_and_std_without_pvalues() -> None:
    metrics = pd.DataFrame(
        [
            {
                "stage": "B2",
                "seed": 42,
                "model": "xgboost",
                "split_name": "random_pair",
                "subset": "test",
                "rmse": 1.0,
                "mae": 0.8,
                "pearson": 0.9,
                "spearman": 0.88,
                "r2": 0.7,
            },
            {
                "stage": "B2",
                "seed": 7,
                "model": "xgboost",
                "split_name": "random_pair",
                "subset": "test",
                "rmse": 1.2,
                "mae": 0.9,
                "pearson": 0.85,
                "spearman": 0.84,
                "r2": 0.6,
            },
        ]
    )

    summary = summarize_multiseed(metrics)

    assert len(summary) == 1
    assert summary.iloc[0]["n_seeds"] == 2
    assert summary.iloc[0]["rmse_mean"] == pytest.approx(1.1)
    assert summary.iloc[0]["rmse_std"] == pytest.approx(0.141421356, rel=1e-5)
    assert "pvalue" not in summary.columns
    assert "p_value" not in summary.columns


def test_drop_forbidden_columns_strips_significance_tests() -> None:
    table = pd.DataFrame({"rmse": [1.0], "pvalue": [0.04], "t_stat": [2.1]})
    cleaned = drop_forbidden_columns(table)
    assert list(cleaned.columns) == ["rmse"]


def test_load_seed_metrics_tags_stage_and_seed(tmp_path: Path) -> None:
    layout = seed_layout(7, output_dir=tmp_path)
    layout.result_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "model": "global_mean",
                "split_name": "random_pair",
                "subset": "test",
                "rmse": 2.0,
                "mae": 1.5,
                "pearson": 0.1,
                "spearman": 0.1,
                "r2": 0.0,
            }
        ]
    ).to_csv(layout.result_dir / "b0_metrics.csv", index=False)

    loaded = load_seed_metrics(layout, resolve_models(["B0"]))

    assert loaded.iloc[0]["stage"] == "B0"
    assert loaded.iloc[0]["seed"] == 7


def test_resolve_models_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unknown models"):
        resolve_models(["B9"])
