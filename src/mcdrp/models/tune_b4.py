"""Validation-based hyperparameter tuning for B4 GNN.

The grid is deliberately small: four candidates over learning rate and dropout.
Architecture width/depth stay at the B4 defaults so the search answers whether
the published defaults were lucky, without turning into a full NAS run.

Selection is validation RMSE, independently on each split. Transferring a
winner from ``random_pair`` to a cold split is not a valid protocol. On a
fixed split that ranking is identical to r2_normalized, because the
mean-effects reference is constant across candidates. Test is scored only for
the selected candidate.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mcdrp.features.expression import build_cell_features
from mcdrp.features.graphs import build_graph_map
from mcdrp.models.baseline_b1 import EXPRESSION_PATH
from mcdrp.models.baseline_b3 import resolve_torch_device
from mcdrp.results.predictions import write_predictions
from mcdrp.splits.make_splits import DEFAULT_SPLITS

RunSplitFn = Callable[..., tuple[list[dict[str, Any]], list[pd.DataFrame]]]

logger = logging.getLogger(__name__)

DEFAULT_GRID = {
    "learning_rate": (0.0005, 0.001),
    "dropout": (0.1, 0.3),
}


def iter_param_grid(grid: dict[str, tuple[Any, ...]]) -> list[dict[str, Any]]:
    """Expand a small parameter grid."""

    keys = list(grid)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(grid[key] for key in keys))
    ]


def annotate(row: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Attach the candidate hyperparameters to a metric row."""

    annotated = dict(row)
    annotated["params"] = json.dumps(params, sort_keys=True)
    return annotated


def tune_b4_for_split(
    cohort: pd.DataFrame,
    assignments: pd.DataFrame,
    graph_map: dict[str, Any],
    cell_feature_map: dict[str, Any],
    *,
    split_name: str,
    target: str,
    batch_size: int,
    max_epochs: int,
    patience: int,
    weight_decay: float,
    graph_hidden_dim: int,
    graph_layers: int,
    cell_hidden_dim: int,
    fusion_hidden_dim: int,
    gradient_clip_norm: float,
    device: str,
    random_state: int,
    log_every: int,
    run_split: RunSplitFn | None = None,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    """Tune B4 on validation RMSE and evaluate the winner on test."""

    if run_split is None:
        from mcdrp.models.gnn_b4 import run_b4_for_split

        run_split = run_b4_for_split

    candidates: list[tuple[dict[str, Any], list[dict[str, Any]], list[pd.DataFrame]]] = []
    rows: list[dict[str, Any]] = []

    for params in iter_param_grid(DEFAULT_GRID):
        split_rows, split_frames = run_split(
            cohort,
            assignments,
            graph_map,
            cell_feature_map,
            split_name=split_name,
            target=target,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            learning_rate=params["learning_rate"],
            weight_decay=weight_decay,
            graph_hidden_dim=graph_hidden_dim,
            graph_layers=graph_layers,
            cell_hidden_dim=cell_hidden_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            dropout=params["dropout"],
            gradient_clip_norm=gradient_clip_norm,
            device=device,
            random_state=random_state,
            log_every=log_every,
        )
        validation = [
            annotate(row, params) for row in split_rows if row["subset"] == "validation"
        ]
        if not validation:
            raise RuntimeError(f"B4 {split_name} produced no validation rows.")
        rows.extend(validation)
        candidates.append((params, split_rows, split_frames))
        logger.info(
            "B4 %s %s validation RMSE %.3f",
            split_name,
            params,
            validation[0]["rmse"],
        )

    best_params, best_rows, best_frames = min(
        candidates,
        key=lambda item: next(row["rmse"] for row in item[1] if row["subset"] == "validation"),
    )
    test_row = annotate(
        next(row for row in best_rows if row["subset"] == "test"),
        best_params,
    )
    test_row["selected_by_validation"] = True
    rows.append(test_row)

    selected_frames = [
        frame
        for frame in best_frames
        if frame["subset"].isin(["validation", "test"]).all()
    ]
    return rows, selected_frames


def tune_b4(
    cohort_path: str | Path = "data/processed/cohort_pairs.csv",
    split_dir: str | Path = "data/processed/splits",
    expression_path: str | Path = EXPRESSION_PATH,
    output: str | Path = "results/baselines/b4_tuning_metrics.csv",
    summary: str | Path = "data/reports/b4_tuning_summary.json",
    predictions_output: str | Path = "results/predictions/b4_tuned.csv",
    *,
    target: str = "ln_ic50",
    n_components: int = 256,
    split_names: tuple[str, ...] = DEFAULT_SPLITS,
    batch_size: int = 256,
    max_epochs: int = 80,
    patience: int = 10,
    weight_decay: float = 0.0001,
    graph_hidden_dim: int = 128,
    graph_layers: int = 3,
    cell_hidden_dim: int = 128,
    fusion_hidden_dim: int = 256,
    gradient_clip_norm: float = 5.0,
    device: str = "auto",
    random_state: int = 42,
    log_every: int = 1,
) -> pd.DataFrame:
    """Tune B4 hyperparameters for all requested splits."""

    torch_device = resolve_torch_device(device)
    if torch_device is None:
        raise RuntimeError("B4 tuning requires PyTorch.")

    cohort = pd.read_csv(cohort_path)
    split_dir = Path(split_dir)
    graph_map = build_graph_map(cohort)

    all_rows: list[dict[str, Any]] = []
    all_frames: list[pd.DataFrame] = []
    for split_name in split_names:
        logger.info("=== Tuning B4 split: %s ===", split_name)
        assignments = pd.read_csv(split_dir / f"{split_name}.csv")
        merged = cohort.merge(assignments, on="pair_id", how="inner")
        train_ids = set(merged.loc[merged["split"].eq("train"), "depmap_id"].unique())
        _pipeline, cell_feature_map = build_cell_features(
            str(expression_path),
            cohort,
            train_ids,
            n_components=n_components,
        )
        split_rows, split_frames = tune_b4_for_split(
            cohort,
            assignments,
            graph_map,
            cell_feature_map,
            split_name=split_name,
            target=target,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            weight_decay=weight_decay,
            graph_hidden_dim=graph_hidden_dim,
            graph_layers=graph_layers,
            cell_hidden_dim=cell_hidden_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            gradient_clip_norm=gradient_clip_norm,
            device=torch_device,
            random_state=random_state,
            log_every=log_every,
        )
        all_rows.extend(split_rows)
        all_frames.extend(split_frames)

    metrics = pd.DataFrame(all_rows)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)
    write_predictions(all_frames, predictions_output)

    summary_data = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_path": str(cohort_path),
        "expression_path": str(expression_path),
        "output": str(output),
        "predictions_output": str(predictions_output),
        "target": target,
        "grid": {key: list(value) for key, value in DEFAULT_GRID.items()},
        "splits": list(split_names),
        "device_requested": device,
        "device_used": torch_device,
        "best_validation": summarize_best_validation(metrics),
    }
    summary_path = Path(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_data, handle, indent=2)
    return metrics


def summarize_best_validation(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize the selected validation setting per split."""

    rows: list[dict[str, Any]] = []
    validation = metrics.loc[metrics["subset"].eq("validation")]
    for split_name, group in validation.groupby("split_name"):
        best = group.sort_values("rmse").iloc[0]
        rows.append(
            {
                "split_name": split_name,
                "params": best["params"],
                "validation_rmse": float(best["rmse"]),
                "best_epoch": int(best["best_epoch"]),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description="Tune B4 GNN hyperparameters.")
    parser.add_argument("--cohort", default="data/processed/cohort_pairs.csv")
    parser.add_argument("--split-dir", default="data/processed/splits")
    parser.add_argument("--expression", default=EXPRESSION_PATH)
    parser.add_argument("--output", default="results/baselines/b4_tuning_metrics.csv")
    parser.add_argument("--summary", default="data/reports/b4_tuning_summary.json")
    parser.add_argument("--predictions-output", default="results/predictions/b4_tuned.csv")
    parser.add_argument("--target", default="ln_ic50")
    parser.add_argument("--n-components", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--graph-hidden-dim", type=int, default=128)
    parser.add_argument("--graph-layers", type=int, default=3)
    parser.add_argument("--cell-hidden-dim", type=int, default=128)
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=list(DEFAULT_SPLITS),
    )
    return parser


def main() -> None:
    """Command-line entry point."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = build_parser().parse_args()
    metrics = tune_b4(
        cohort_path=args.cohort,
        split_dir=args.split_dir,
        expression_path=args.expression,
        output=args.output,
        summary=args.summary,
        predictions_output=args.predictions_output,
        target=args.target,
        n_components=args.n_components,
        split_names=tuple(args.splits),
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        weight_decay=args.weight_decay,
        graph_hidden_dim=args.graph_hidden_dim,
        graph_layers=args.graph_layers,
        cell_hidden_dim=args.cell_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        gradient_clip_norm=args.gradient_clip_norm,
        device=args.device,
        random_state=args.random_state,
        log_every=args.log_every,
    )
    print(f"Wrote B4 tuning metrics to {args.output}")
    print(metrics.sort_values(["split_name", "subset", "rmse"]).to_string())


if __name__ == "__main__":
    main()
