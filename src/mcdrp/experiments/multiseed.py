"""Multi-seed experiment runner with dispersion reporting.

A single seed cannot support a ranking. This module rebuilds leakage-safe
splits under each seed, trains the selected models with the same seed for
initialization, and then reports mean and spread. It does not compute
p-values: pairs are not independent, so a naive test would be
pseudoreplication.

Canonical seed-42 artifacts under ``data/processed/splits`` and
``results/baselines`` are left untouched. Each replication writes to
``data/processed/splits/seed_<n>/`` and ``results/multiseed/seed_<n>/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mcdrp.experiments.run_experiment import ExperimentStep, build_command, shell_join
from mcdrp.splits.make_splits import DEFAULT_SPLITS

logger = logging.getLogger(__name__)

DEFAULT_SEEDS = (42, 7, 13)
DEFAULT_MODELS = ("B0", "B1", "B2")
DEFAULT_OUTPUT_DIR = "results/multiseed"
DEFAULT_SPLIT_ROOT = "data/processed/splits"

GROUP_COLUMNS = ("stage", "model", "split_name", "subset")
METRIC_COLUMNS = (
    "rmse",
    "mae",
    "pearson",
    "spearman",
    "r2",
    "r2_normalized",
    "pearson_normalized",
    "spearman_normalized",
    "pearson_per_drug",
    "spearman_per_drug",
    "r2_per_drug",
    "n_groups_per_drug",
    "pearson_per_cell",
    "spearman_per_cell",
    "r2_per_cell",
    "n_groups_per_cell",
)
FORBIDDEN_STAT_COLUMNS = {
    "pvalue",
    "p_value",
    "p-value",
    "pval",
    "tstat",
    "t_stat",
    "t-statistic",
    "significant",
    "significance",
}


@dataclass(frozen=True)
class ModelSpec:
    """How to invoke one model for a single seed."""

    name: str
    module: str
    output_name: str
    summary_name: str
    predictions_name: str
    supports_random_state: bool
    needs_device: bool = False
    extra_args: dict[str, Any] | None = None


MODEL_SPECS: dict[str, ModelSpec] = {
    "B0": ModelSpec(
        name="B0",
        module="mcdrp.models.baseline_b0",
        output_name="b0_metrics.csv",
        summary_name="b0_summary.json",
        predictions_name="predictions/b0.csv",
        supports_random_state=False,
    ),
    "B1": ModelSpec(
        name="B1",
        module="mcdrp.models.baseline_b1",
        output_name="b1_metrics.csv",
        summary_name="b1_summary.json",
        predictions_name="predictions/b1.csv",
        supports_random_state=True,
    ),
    "B2": ModelSpec(
        name="B2",
        module="mcdrp.models.baseline_b2",
        output_name="b2_metrics.csv",
        summary_name="b2_summary.json",
        predictions_name="predictions/b2.csv",
        supports_random_state=True,
        needs_device=True,
    ),
    "B3": ModelSpec(
        name="B3",
        module="mcdrp.models.baseline_b3",
        output_name="b3_metrics.csv",
        summary_name="b3_summary.json",
        predictions_name="predictions/b3.csv",
        supports_random_state=True,
        needs_device=True,
    ),
    "B4": ModelSpec(
        name="B4",
        module="mcdrp.models.gnn_b4",
        output_name="b4_gnn_metrics.csv",
        summary_name="b4_gnn_summary.json",
        predictions_name="predictions/b4.csv",
        supports_random_state=True,
        needs_device=True,
    ),
    "B5": ModelSpec(
        name="B5",
        module="mcdrp.models.gnn_b5",
        output_name="b5_hybrid_gnn_metrics.csv",
        summary_name="b5_hybrid_gnn_summary.json",
        predictions_name="predictions/b5.csv",
        supports_random_state=True,
        needs_device=True,
    ),
    "B6": ModelSpec(
        name="B6",
        module="mcdrp.models.ablation_b6",
        output_name="b6_modality_ablation_metrics.csv",
        summary_name="b6_modality_ablation_summary.json",
        predictions_name="predictions/b6.csv",
        supports_random_state=True,
        needs_device=True,
        extra_args={"no-b2-tuned": True},
    ),
    "B7": ModelSpec(
        name="B7",
        module="mcdrp.models.pretrained_b7",
        output_name="b7_pretrained_drug_pathway_metrics.csv",
        summary_name="b7_pretrained_drug_pathway_summary.json",
        predictions_name="predictions/b7.csv",
        supports_random_state=True,
        needs_device=True,
    ),
}


@dataclass(frozen=True)
class SeedLayout:
    """Isolated paths for one replication seed."""

    seed: int
    split_dir: Path
    result_dir: Path
    splits_summary: Path


def seed_layout(
    seed: int,
    *,
    split_root: str | Path = DEFAULT_SPLIT_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> SeedLayout:
    """Build isolated split and result directories for one seed."""

    result_dir = Path(output_dir) / f"seed_{seed}"
    return SeedLayout(
        seed=seed,
        split_dir=Path(split_root) / f"seed_{seed}",
        result_dir=result_dir,
        splits_summary=result_dir / "splits_summary.json",
    )


def resolve_models(names: tuple[str, ...] | list[str]) -> list[ModelSpec]:
    """Look up model specs, preserving the requested order."""

    resolved: list[ModelSpec] = []
    unknown: list[str] = []
    for name in names:
        spec = MODEL_SPECS.get(name)
        if spec is None:
            unknown.append(name)
        else:
            resolved.append(spec)
    if unknown:
        known = ", ".join(MODEL_SPECS)
        raise ValueError(f"Unknown models: {unknown}. Known models: {known}.")
    return resolved


def steps_for_seed(
    layout: SeedLayout,
    models: list[ModelSpec],
    *,
    split_names: tuple[str, ...],
    device: str,
    skip_existing: bool = False,
) -> list[ExperimentStep]:
    """Build split + model steps for one seed.

    Split construction and model initialization share the same seed. B0 is
    deterministic given the split, so it has no ``--random-state``. B6 ignores
    a global B2 tuning file so another seed's hyperparameters cannot leak in.
    """

    steps: list[ExperimentStep] = []
    split_csvs = [layout.split_dir / f"{name}.csv" for name in split_names]
    splits_exist = all(path.exists() for path in split_csvs)
    if not (skip_existing and splits_exist):
        steps.append(
            ExperimentStep(
                name=f"splits_seed_{layout.seed}",
                module="mcdrp.splits.make_splits",
                args={
                    "output-dir": str(layout.split_dir),
                    "summary": str(layout.splits_summary),
                    "seed": layout.seed,
                    "splits": list(split_names),
                },
            )
        )

    for spec in models:
        output_path = layout.result_dir / spec.output_name
        if skip_existing and output_path.exists():
            logger.info("Skipping %s seed %s; %s exists.", spec.name, layout.seed, output_path)
            continue
        args: dict[str, Any] = {
            "split-dir": str(layout.split_dir),
            "output": str(output_path),
            "summary": str(layout.result_dir / spec.summary_name),
            "predictions-output": str(layout.result_dir / spec.predictions_name),
            "splits": list(split_names),
        }
        if spec.supports_random_state:
            args["random-state"] = layout.seed
        if spec.needs_device:
            args["device"] = device
        if spec.extra_args:
            args.update(spec.extra_args)
        steps.append(
            ExperimentStep(
                name=f"{spec.name.lower()}_seed_{layout.seed}",
                module=spec.module,
                args=args,
            )
        )
    return steps


def drop_forbidden_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Remove significance-test columns if a caller added them."""

    dropped = [column for column in table.columns if column.lower() in FORBIDDEN_STAT_COLUMNS]
    if dropped:
        logger.warning("Dropping forbidden significance columns: %s", dropped)
        return table.drop(columns=dropped)
    return table


def load_seed_metrics(
    layout: SeedLayout,
    models: list[ModelSpec],
) -> pd.DataFrame:
    """Load one seed's metric tables and tag them with stage and seed."""

    frames: list[pd.DataFrame] = []
    for spec in models:
        path = layout.result_dir / spec.output_name
        if not path.exists():
            logger.warning("Missing metrics for %s seed %s: %s", spec.name, layout.seed, path)
            continue
        table = pd.read_csv(path)
        table = drop_forbidden_columns(table)
        table.insert(0, "stage", spec.name)
        table.insert(1, "seed", layout.seed)
        frames.append(table)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_multiseed(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and spread across seeds. No hypothesis tests."""

    if metrics.empty:
        return metrics

    present = [column for column in METRIC_COLUMNS if column in metrics.columns]
    rows: list[dict[str, Any]] = []
    grouped = metrics.groupby(list(GROUP_COLUMNS), dropna=False)
    for keys, group in grouped:
        stage, model, split_name, subset = keys
        row: dict[str, Any] = {
            "stage": stage,
            "model": model,
            "split_name": split_name,
            "subset": subset,
            "n_seeds": int(group["seed"].nunique()),
            "seeds": ",".join(str(seed) for seed in sorted(group["seed"].unique())),
        }
        for column in present:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = (
                float(values.mean()) if values.notna().any() else float("nan")
            )
            row[f"{column}_std"] = (
                float(values.std(ddof=1)) if values.notna().sum() >= 2 else float("nan")
            )
            row[f"{column}_min"] = (
                float(values.min()) if values.notna().any() else float("nan")
            )
            row[f"{column}_max"] = (
                float(values.max()) if values.notna().any() else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_multiseed_outputs(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str | Path,
    *,
    seeds: tuple[int, ...],
    models: tuple[str, ...],
    split_names: tuple[str, ...],
) -> dict[str, Any]:
    """Write concatenated metrics, dispersion summary, and a JSON report."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.csv"
    report_path = output_dir / "report.json"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(seeds),
        "models": list(models),
        "splits": list(split_names),
        "n_metric_rows": int(len(metrics)),
        "n_summary_rows": int(len(summary)),
        "metrics_path": str(metrics_path).replace("\\", "/"),
        "summary_path": str(summary_path).replace("\\", "/"),
        "hypothesis_tests": "none",
        "note": (
            "Pairs are not independent. Units of replication are cell lines and "
            "drugs. Report mean and spread only; do not treat std as a p-value."
        ),
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def run_multiseed(
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    models: tuple[str, ...] = DEFAULT_MODELS,
    split_names: tuple[str, ...] = DEFAULT_SPLITS,
    device: str = "auto",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    split_root: str | Path = DEFAULT_SPLIT_ROOT,
    dry_run: bool = False,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """Rebuild splits and models under several seeds, then summarize dispersion."""

    if not seeds:
        raise ValueError("At least one seed is required.")
    specs = resolve_models(models)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(seeds),
        "models": list(models),
        "splits": list(split_names),
        "device": device,
        "dry_run": dry_run,
        "skip_existing": skip_existing,
        "hypothesis_tests": "none",
        "steps": [],
    }

    for seed in seeds:
        layout = seed_layout(seed, split_root=split_root, output_dir=output_dir)
        layout.result_dir.mkdir(parents=True, exist_ok=True)
        layout.split_dir.mkdir(parents=True, exist_ok=True)
        for step in steps_for_seed(
            layout,
            specs,
            split_names=split_names,
            device=device,
            skip_existing=skip_existing,
        ):
            command = build_command(step)
            entry: dict[str, Any] = {
                "name": step.name,
                "module": step.module,
                "command": command,
                "command_text": shell_join(command),
            }
            print(f"[{step.name}] {entry['command_text']}")
            if not dry_run:
                completed = subprocess.run(command, check=True)
                entry["returncode"] = completed.returncode
            manifest["steps"].append(entry)

    if not dry_run:
        metrics_frames = [
            load_seed_metrics(
                seed_layout(seed, split_root=split_root, output_dir=output_dir),
                specs,
            )
            for seed in seeds
        ]
        nonempty = [frame for frame in metrics_frames if not frame.empty]
        metrics = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
        summary = summarize_multiseed(metrics)
        manifest["report"] = write_multiseed_outputs(
            metrics,
            summary,
            output_dir,
            seeds=seeds,
            models=models,
            split_names=split_names,
        )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Wrote multi-seed manifest to {manifest_path}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run models under several split and initialization seeds and report "
            "mean and spread. No significance tests."
        )
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Replication seeds. Each seed rebuilds splits and reinitializes models.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        choices=list(MODEL_SPECS),
        help="Models to run. Defaults to B0 B1 B2; GNN stages are opt-in.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=list(DEFAULT_SPLITS),
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-root", default=DEFAULT_SPLIT_ROOT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write a manifest without executing them.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a seed/model whose metrics CSV already exists.",
    )
    return parser


def main() -> None:
    """Command-line entry point."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = build_parser().parse_args()
    run_multiseed(
        seeds=tuple(args.seeds),
        models=tuple(args.models),
        split_names=tuple(args.splits),
        device=args.device,
        output_dir=args.output_dir,
        split_root=args.split_root,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
