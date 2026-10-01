"""Cross-check stored, identity-validated predictions with official DrEvalPy."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mcdrp.results.metrics_report import GROUP_COLUMNS, build_metrics_report
from mcdrp.results.predictions import load_predictions


def crosscheck(
    predictions: pd.DataFrame, cohort: pd.DataFrame, split_dir: str, *, target: str = "ln_ic50"
) -> dict:
    from drevalpy.datasets.dataset import DrugResponseDataset
    from drevalpy.evaluation import evaluate

    own = build_metrics_report(predictions, cohort, split_dir, target=target)
    metrics = {
        "RMSE": "rmse",
        "MAE": "mae",
        "R^2": "r2",
        "Pearson": "pearson",
        "Spearman": "spearman",
    }
    rows = []
    for keys, frame in predictions.groupby(list(GROUP_COLUMNS)):
        aligned = frame.merge(
            cohort[["pair_id", "depmap_id", "drug_id"]], on="pair_id", validate="many_to_one"
        )
        dataset = DrugResponseDataset(
            response=aligned["y_true"].to_numpy(),
            predictions=aligned["y_pred"].to_numpy(),
            cell_line_ids=aligned["depmap_id"].to_numpy(),
            drug_ids=aligned["drug_id"].astype(str).to_numpy(),
        )
        external = evaluate(dataset, list(metrics))
        selected = own
        for column, value in zip(GROUP_COLUMNS, keys, strict=True):
            selected = selected.loc[selected[column].eq(value)]
        for metric, column in metrics.items():
            expected, actual = float(selected.iloc[0][column]), float(external[metric])
            # DrEval returns zero for constant predictions; ours leaves correlation undefined.
            semantic = bool(
                column in {"pearson", "spearman"} and np.isnan(expected) and actual == 0.0
            )
            same = bool(np.isclose(expected, actual, rtol=1e-7, atol=1e-9, equal_nan=True))
            if not same and not semantic:
                raise ValueError(f"DrEval mismatch: {keys}, {metric}: {expected} != {actual}")
            rows.append(
                dict(zip(GROUP_COLUMNS, keys, strict=True))
                | {
                    "metric": metric,
                    "ours": expected if np.isfinite(expected) else None,
                    "dreval": actual if np.isfinite(actual) else None,
                    "agreement": same,
                    "constant_correlation_convention": semantic,
                }
            )
    split_checks = []
    for split_name in predictions["split_name"].unique():
        assignments = pd.read_csv(Path(split_dir) / f"{split_name}.csv")
        data = assignments.merge(cohort[["pair_id", "drug_id", "depmap_id"]], on="pair_id")
        if split_name in {"cold_drug", "cold_cell"}:
            entity = "drug_id" if split_name == "cold_drug" else "depmap_id"
            membership = (
                data.loc[data["split"].isin(("train", "validation", "test"))]
                .groupby(entity)["split"]
                .nunique()
            )
            if membership.max() != 1:
                raise ValueError(f"{split_name} entities leak between partitions.")
            split_checks.append({"split": split_name, "entity": entity, "disjoint": True})
    return {
        "drevalpy_version": importlib.metadata.version("drevalpy"),
        "checks": rows,
        "split_checks": split_checks,
        "scope": "Official DrEval metric/dataset interoperability and entity-disjointness "
        "check; not an independent model retraining benchmark.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--cohort", default="data/processed/cohort_pairs.csv")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--target", default="ln_ic50")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = crosscheck(
        load_predictions(args.predictions),
        pd.read_csv(args.cohort),
        args.split_dir,
        target=args.target,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
