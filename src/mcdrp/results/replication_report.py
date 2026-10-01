"""Summarize fixed-cohort fitting replications without mixing shuffled variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from mcdrp.experiments.multiseed import summarize_multiseed


def replication_report(
    root: str = "results/transfer_strict",
    summary: str = "data/reports/transfer_prism_replication_summary.json",
) -> dict[str, Any]:
    """Require matching model/subset rows and aggregate each model over all seeds."""

    paths = sorted(Path(root).glob("seed_*/model_metrics.csv"))
    if len(paths) < 2:
        raise ValueError("At least two completed replications are required.")
    tables = []
    expected = None
    keys = ["stage", "model", "variant", "split_name", "subset"]
    for path in paths:
        table = pd.read_csv(path)
        combinations = set(table[keys].itertuples(index=False, name=None))
        if expected is not None and combinations != expected:
            raise ValueError("Replications must contain the same models and subsets.")
        expected = combinations
        table.insert(0, "seed", int(path.parent.name.removeprefix("seed_")))
        tables.append(table)
    metrics = pd.concat(tables, ignore_index=True)
    aggregated = summarize_multiseed(metrics)
    test = metrics.loc[metrics["subset"].eq("test")].copy()
    if test["split_name"].nunique() != 1:
        raise ValueError("Fixed-cohort replication must evaluate exactly one split.")
    baseline = test.loc[
        test["model"].eq("mean_effects")
        & test["stage"].eq("B0")
        & test["variant"].eq("standard")
    ]
    if baseline["seed"].duplicated().any() or set(baseline["seed"]) != set(test["seed"]):
        raise ValueError("Every seed requires exactly one standard B0 mean-effects reference.")
    if test.groupby("seed")["n_rows"].nunique().gt(1).any():
        raise ValueError("All models must evaluate the same cohort size in each seed.")
    if test["n_rows"].nunique() != 1:
        raise ValueError("Fixed-cohort replications must evaluate the same cohort size.")
    references = baseline.set_index("seed")["rmse"]
    test["paired_rmse_delta_vs_mean_effects"] = test["rmse"] - test["seed"].map(references)
    models = test.loc[test["stage"].eq("B6")]
    paired = models.groupby(["model", "variant"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        paired_delta_mean=("paired_rmse_delta_vs_mean_effects", "mean"),
        paired_delta_std=("paired_rmse_delta_vs_mean_effects", "std"),
        seeds_won=("paired_rmse_delta_vs_mean_effects", lambda x: int(x.lt(0).sum())),
    )
    rows = aggregated.loc[aggregated["subset"].eq("test")]
    wanted = [
        *keys,
        "n_seeds",
        "seeds",
        "rmse_mean",
        "rmse_std",
        "r2_normalized_mean",
        "spearman_per_drug_mean",
        "r2_per_drug_mean",
    ]
    report = {
        "root": root,
        "seeds": sorted(metrics["seed"].unique().tolist()),
        "per_seed_test": json.loads(test.to_json(orient="records")),
        "test_summary": json.loads(rows[wanted].to_json(orient="records")),
        "paired_deltas": json.loads(paired.to_json(orient="records")),
        "note": (
            "Fixed cohort and GDSC partition. Seeds change PCA fitting and target "
            "permutation. Dispersion is fitting/permutation sensitivity, not an "
            "independent external-assay confidence interval. No hypothesis tests."
        ),
    }
    Path(summary).parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(Path(root) / "aggregated_metrics.csv", index=False)
    Path(summary).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/transfer_strict")
    parser.add_argument(
        "--summary", default="data/reports/transfer_prism_replication_summary.json"
    )
    return parser


def main() -> None:
    report = replication_report(**vars(build_parser().parse_args()))
    print(json.dumps(report["paired_deltas"], indent=2))


if __name__ == "__main__":
    main()
