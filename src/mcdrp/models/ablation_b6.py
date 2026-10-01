"""B6 modality ablation study.

This script compares feature blocks under the same model family so the project
can answer a stronger question than "which model wins?": which information
source is responsible for each performance gain?

Default ablations:

- expression_pca
- pathways
- morgan
- morgan_expression_pca
- morgan_pathways
- morgan_expression_pca_pathways
- targets
- targets_expression_pca
- targets_pathways
- targets_expression_pca_pathways

The ``targets_*`` variants swap Morgan fingerprints for a multi-hot encoding
of GDSC ``putative_target``, with the vocabulary fit on training drugs only.
They are not in the default list so a rerun of the original six ablations
stays comparable; pass them explicitly with ``--ablations``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mcdrp.data.target_annotations import apply_target_view
from mcdrp.features.expression import build_cell_features
from mcdrp.features.fingerprints import build_fingerprint_matrix
from mcdrp.features.pathways import build_pathway_features
from mcdrp.features.permutations import permute_cell_map, permute_entity_features
from mcdrp.features.predicted_targets import predict_targets_from_fingerprints
from mcdrp.features.target_expression import build_target_expression_product
from mcdrp.features.targets import build_target_matrix, shuffle_target_matrix
from mcdrp.metrics import regression_metrics
from mcdrp.models.baseline_b2 import (
    fit_xgboost,
    load_selected_xgb_params,
    predict_xgboost,
    xgb_device_params,
    xgb_selection_context,
)
from mcdrp.results.predictions import (
    SHUFFLED_VARIANT,
    STANDARD_VARIANT,
    prediction_frame,
    write_predictions,
)
from mcdrp.splits.make_splits import DEFAULT_SPLITS, KNOWN_SPLIT_CHOICES
from mcdrp.splits.validation import selection_fingerprint, validate_assignments

logger = logging.getLogger(__name__)

EXPRESSION_PATH = "data/raw/OmicsExpressionTPMLogp1HumanProteinCodingGenesStranded.csv"
DEFAULT_ABLATIONS = (
    "expression_pca",
    "pathways",
    "morgan",
    "morgan_expression_pca",
    "morgan_pathways",
    "morgan_expression_pca_pathways",
)
TARGET_ABLATIONS = (
    "targets",
    "targets_expression_pca",
    "targets_pathways",
    "targets_expression_pca_pathways",
    "predicted_targets",
    "predicted_targets_expression_pca",
    "targets_expression_product",
)
ALL_ABLATIONS = DEFAULT_ABLATIONS + TARGET_ABLATIONS


def ablation_uses(ablation: str, block: str) -> bool:
    """Return whether ``ablation`` includes a named feature block.

    Substring matching is unsafe here: ``predicted_targets`` contains
    ``targets``, so the two drug representations have to be distinguished by
    prefix rather than by ``in``.
    """

    if block == "predicted_targets":
        return ablation.startswith("predicted_targets")
    if block == "expression_product":
        return "expression_product" in ablation
    if block == "targets":
        if ablation.startswith("predicted_targets"):
            return False
        return ablation == "targets" or ablation.startswith("targets_")
    if block == "morgan":
        return "morgan" in ablation
    if block == "expression_pca":
        return "expression_pca" in ablation
    if block == "pathways":
        return "pathways" in ablation
    raise ValueError(f"Unknown feature block: {block}")


def feature_matrix_from_cell_map(
    rows: pd.DataFrame,
    cell_feature_map: dict[str, np.ndarray],
) -> np.ndarray:
    """Build a dense cell feature matrix aligned to response rows."""

    n_features = next(iter(cell_feature_map.values())).shape[0]
    features = np.zeros((len(rows), n_features), dtype=np.float32)
    for idx, depmap_id in enumerate(rows["depmap_id"]):
        value = cell_feature_map.get(str(depmap_id))
        if value is None:
            raise ValueError(f"Missing cell features for {depmap_id}.")
        features[idx] = value
    return features


def assemble_ablation_features(
    rows: pd.DataFrame,
    *,
    ablation: str,
    cohort: pd.DataFrame,
    fp_matrix: np.ndarray,
    pca_feature_map: dict[str, np.ndarray],
    pathway_feature_map: dict[str, np.ndarray],
    target_matrix: np.ndarray,
    predicted_target_matrix: np.ndarray,
    product_matrix: np.ndarray,
) -> np.ndarray:
    """Assemble the requested feature blocks."""

    pair_id_to_idx = {pid: idx for idx, pid in enumerate(cohort["pair_id"])}
    row_indices = np.array([pair_id_to_idx[pid] for pid in rows["pair_id"]])

    blocks: list[np.ndarray] = []
    if ablation_uses(ablation, "morgan"):
        blocks.append(fp_matrix[row_indices].astype(np.float32))
    if ablation_uses(ablation, "predicted_targets"):
        blocks.append(predicted_target_matrix[row_indices].astype(np.float32))
    elif ablation_uses(ablation, "expression_product"):
        blocks.append(product_matrix[row_indices].astype(np.float32))
    elif ablation_uses(ablation, "targets"):
        blocks.append(target_matrix[row_indices].astype(np.float32))
    if ablation_uses(ablation, "expression_pca"):
        blocks.append(feature_matrix_from_cell_map(rows, pca_feature_map))
    if ablation_uses(ablation, "pathways"):
        blocks.append(feature_matrix_from_cell_map(rows, pathway_feature_map))

    if not blocks:
        raise ValueError(f"Unknown or empty ablation: {ablation}")
    return np.hstack(blocks).astype(np.float32)


def metric_row(
    model_name: str,
    split_name: str,
    subset: str,
    rows: pd.DataFrame,
    predictions: np.ndarray,
    target: str,
    *,
    n_features: int,
    fit_seconds: float,
    best_iteration: int,
    device: str,
    variant: str = STANDARD_VARIANT,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Create one ablation metrics row and its tidy predictions."""

    metrics = regression_metrics(
        rows[target].to_numpy(),
        predictions,
        cell_keys=rows["depmap_id"].tolist(),
        drug_keys=rows["drug_id"].tolist(),
    )
    row = {
        "model": model_name,
        "split_name": split_name,
        "subset": subset,
        "n_rows": int(len(rows)),
        "n_cell_lines": int(rows["depmap_id"].nunique()),
        "n_drugs": int(rows["drug_id"].nunique()),
        "n_features": n_features,
        "fit_seconds": fit_seconds,
        "best_iteration": int(best_iteration),
        "device": device,
        **metrics,
    }
    frame = prediction_frame(
        rows,
        rows[target].to_numpy(),
        predictions,
        stage="B6",
        model=model_name,
        split_name=split_name,
        subset=subset,
        variant=variant,
    )
    return row, frame


def run_ablation_for_split(
    cohort: pd.DataFrame,
    assignments: pd.DataFrame,
    fp_matrix: np.ndarray,
    pca_feature_map: dict[str, np.ndarray],
    pathway_feature_map: dict[str, np.ndarray],
    target_matrix: np.ndarray,
    predicted_target_matrix: np.ndarray,
    product_matrix: np.ndarray,
    *,
    split_name: str,
    target: str,
    ablations: tuple[str, ...],
    xgb_params: dict[str, Any],
    device: str,
    variant: str = STANDARD_VARIANT,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    """Run every ablation for one split."""

    data = cohort.merge(assignments, on="pair_id", how="inner", validate="one_to_one")
    subsets = {
        subset: data.loc[data["split"].eq(subset)].copy()
        for subset in ("train", "validation", "calibration", "test")
        if data["split"].eq(subset).any()
    }

    results: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    feature_kwargs = {
        "cohort": cohort,
        "fp_matrix": fp_matrix,
        "pca_feature_map": pca_feature_map,
        "pathway_feature_map": pathway_feature_map,
        "target_matrix": target_matrix,
        "predicted_target_matrix": predicted_target_matrix,
        "product_matrix": product_matrix,
    }
    for ablation in ablations:
        x_train = assemble_ablation_features(
            subsets["train"], ablation=ablation, **feature_kwargs
        )
        x_val = assemble_ablation_features(
            subsets["validation"], ablation=ablation, **feature_kwargs
        )
        y_train = subsets["train"][target].to_numpy()
        y_val = subsets["validation"][target].to_numpy()

        logger.info(
            "B6 %s / %s — train=%s, validation=%s",
            split_name,
            ablation,
            x_train.shape,
            x_val.shape,
        )
        t0 = time.time()
        model, actual_device = fit_xgboost(
            xgb_params,
            x_train,
            y_train,
            x_val,
            y_val,
            device=device,
        )
        fit_seconds = time.time() - t0
        best_iteration = getattr(model, "best_iteration", xgb_params["n_estimators"] - 1)

        for subset in (
            name for name in ("validation", "calibration", "test") if name in subsets
        ):
            rows = subsets[subset]
            x_subset = assemble_ablation_features(rows, ablation=ablation, **feature_kwargs)
            predictions = predict_xgboost(model, x_subset)
            row, frame = metric_row(
                ablation,
                split_name,
                subset,
                rows,
                predictions,
                target,
                n_features=x_subset.shape[1],
                fit_seconds=fit_seconds,
                best_iteration=best_iteration,
                device=actual_device,
                variant=variant,
            )
            results.append(row)
            frames.append(frame)
    return results, frames


def run_b6_ablation(
    cohort_path: str | Path = "data/processed/cohort_pairs.csv",
    split_dir: str | Path = "data/processed/splits",
    expression_path: str | Path = EXPRESSION_PATH,
    output: str | Path = "results/ablations/b6_modality_ablation_metrics.csv",
    summary: str | Path = "data/reports/b6_modality_ablation_summary.json",
    predictions_output: str | Path = "results/predictions/b6.csv",
    b2_tuned: str | Path | None = "results/baselines/b2_tuning_metrics.csv",
    *,
    target: str = "ln_ic50",
    n_components: int = 256,
    gene_sets: str = "builtin_cancer_core",
    min_pathway_genes: int = 3,
    split_names: tuple[str, ...] = DEFAULT_SPLITS,
    ablations: tuple[str, ...] = DEFAULT_ABLATIONS,
    n_estimators: int = 700,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    early_stopping_rounds: int = 40,
    device: str = "auto",
    random_state: int = 42,
    shuffle_targets: bool = False,
    shuffle_seed: int = 0,
    permute_cell_features: bool = False,
    permute_drug_features: bool = False,
    permutation_seed: int = 42,
    target_view: str = "raw",
    target_annotations: str = "data/mappings/target_annotations_normalized.csv",
) -> pd.DataFrame:
    """Run the B6 feature ablation study."""

    cohort = pd.read_csv(cohort_path, low_memory=False)
    cohort = apply_target_view(cohort, target_annotations, target_view)
    if shuffle_targets and (permute_cell_features or permute_drug_features):
        raise ValueError("Use one null protocol at a time.")
    if permute_cell_features and any(ablation_uses(a, "expression_product") for a in ablations):
        raise ValueError("Cell permutation is not implemented for expression products.")
    if target_view in {"mechanisms", "typed"} and any(
        ablation_uses(a, "expression_product") for a in ablations
    ):
        raise ValueError("Expression products require unprefixed gene annotations.")
    split_dir = Path(split_dir)
    needs_morgan = any(ablation_uses(name, "morgan") for name in ablations) or any(
        ablation_uses(name, "predicted_targets") for name in ablations
    )
    needs_targets = any(
        ablation_uses(name, block)
        for name in ablations
        for block in ("targets", "predicted_targets", "expression_product")
    )
    needs_predicted = any(ablation_uses(name, "predicted_targets") for name in ablations)
    needs_product = any(ablation_uses(name, "expression_product") for name in ablations)
    needs_pca = any(ablation_uses(name, "expression_pca") for name in ablations)
    needs_pathways = any(ablation_uses(name, "pathways") for name in ablations)
    variant = SHUFFLED_VARIANT if shuffle_targets else STANDARD_VARIANT
    if target_view != "raw":
        variant = f"targets_{target_view}"
    if permute_cell_features or permute_drug_features:
        blocks = (
            "both"
            if permute_cell_features and permute_drug_features
            else ("cell" if permute_cell_features else "drug")
        )
        variant = f"{variant}_{blocks}_permuted_{permutation_seed}"

    if needs_morgan:
        logger.info("Building Morgan fingerprint matrix for %d rows ...", len(cohort))
        fp_matrix = build_fingerprint_matrix(cohort["canonical_smiles"])
    else:
        fp_matrix = np.zeros((len(cohort), 0), dtype=np.float32)

    xgb_params: dict[str, Any] = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "early_stopping_rounds": early_stopping_rounds,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": -1,
    }
    xgb_params.update(xgb_device_params(device))
    expected_fingerprints = None
    if b2_tuned is not None and Path(b2_tuned).exists():
        context = xgb_selection_context(expression_path, n_components, random_state)
        expected_fingerprints = {
            name: selection_fingerprint(
                cohort, pd.read_csv(split_dir / f"{name}.csv"), target=target, context=context
            )
            for name in split_names
        }
    selected_by_split = load_selected_xgb_params(
        b2_tuned, fallback=xgb_params, expected_fingerprints=expected_fingerprints
    )

    all_results: list[dict[str, Any]] = []
    all_frames: list[pd.DataFrame] = []
    used_params: dict[str, dict[str, Any]] = {}
    pathway_names: list[str] = []
    for split_name in split_names:
        logger.info("=== Processing B6 ablations split: %s ===", split_name)
        assignments = pd.read_csv(split_dir / f"{split_name}.csv")
        validate_assignments(cohort, assignments)
        merged = cohort.merge(assignments, on="pair_id", how="inner")
        train_ids = set(merged.loc[merged["split"].eq("train"), "depmap_id"].unique())
        train_drugs = set(
            merged.loc[merged["split"].eq("train"), "drug_id"].astype(str).unique()
        )

        pca_feature_map: dict[str, np.ndarray] = {}
        pathway_feature_map: dict[str, np.ndarray] = {}
        pca_variance = float("nan")
        if needs_pca:
            pca_pipeline, pca_feature_map = build_cell_features(
                str(expression_path),
                cohort,
                train_ids,
                n_components=n_components,
                random_state=random_state,
            )
            pca_variance = pca_pipeline.explained_variance_ratio_sum * 100
        if needs_pathways:
            pathway_pipeline, pathway_feature_map = build_pathway_features(
                str(expression_path),
                cohort,
                train_ids,
                gene_sets=gene_sets,
                min_genes=min_pathway_genes,
            )
            pathway_names = pathway_pipeline.pathway_names

        if needs_targets:
            target_pipeline, target_matrix = build_target_matrix(cohort, train_drugs)
            n_target_tokens = len(target_pipeline.target_names)
            if shuffle_targets:
                target_matrix = shuffle_target_matrix(
                    target_matrix,
                    cohort["drug_id"].astype(str).tolist(),
                    seed=shuffle_seed,
                )
                logger.info(
                    "Shuffled target-drug pairing for %s with seed %d",
                    split_name,
                    shuffle_seed,
                )
        else:
            target_pipeline = None
            target_matrix = np.zeros((len(cohort), 0), dtype=np.float32)
            n_target_tokens = 0

        labels = (
            cohort[["pair_id"]]
            .merge(assignments, on="pair_id", validate="one_to_one")["split"]
            .tolist()
        )
        split_fp_matrix = fp_matrix
        provenance_dir = Path(predictions_output).parent / f"{split_name}_{variant}_donors"
        if permute_drug_features:
            provenance_dir.mkdir(parents=True, exist_ok=True)
            if needs_morgan:
                split_fp_matrix, donors = permute_entity_features(
                    fp_matrix,
                    cohort["drug_id"].astype(str).tolist(),
                    labels,
                    seed=permutation_seed,
                )
                donors.to_csv(provenance_dir / "morgan.csv", index=False)
            if needs_targets:
                target_matrix, donors = permute_entity_features(
                    target_matrix,
                    cohort["drug_id"].astype(str).tolist(),
                    labels,
                    seed=permutation_seed,
                )
                donors.to_csv(provenance_dir / "targets.csv", index=False)
        if permute_cell_features:
            provenance_dir.mkdir(parents=True, exist_ok=True)
            if needs_pca:
                pca_feature_map, donors = permute_cell_map(
                    pca_feature_map, cohort, assignments, seed=permutation_seed
                )
                donors.to_csv(provenance_dir / "expression_pca.csv", index=False)
            if needs_pathways:
                pathway_feature_map, donors = permute_cell_map(
                    pathway_feature_map, cohort, assignments, seed=permutation_seed
                )
                donors.to_csv(provenance_dir / "pathways.csv", index=False)

        if needs_predicted:
            predicted_target_matrix = predict_targets_from_fingerprints(
                cohort, train_drugs, split_fp_matrix, target_matrix
            )
        else:
            predicted_target_matrix = np.zeros((len(cohort), 0), dtype=np.float32)

        if needs_product:
            if target_pipeline is None:
                raise ValueError("expression_product requires a target vocabulary.")
            _fusion, product_matrix = build_target_expression_product(
                str(expression_path),
                cohort,
                target_pipeline,
                target_matrix,
                train_ids,
            )
        else:
            product_matrix = np.zeros((len(cohort), 0), dtype=np.float32)

        logger.info(
            "B6 %s — PCA variance %s, pathways=%d, target tokens=%d",
            split_name,
            "n/a" if not needs_pca else f"{pca_variance:.1f}%",
            len(pathway_names),
            n_target_tokens,
        )

        split_params = selected_by_split.get(split_name, xgb_params)
        used_params[split_name] = {
            key: value for key, value in split_params.items() if key != "n_jobs"
        }
        logger.info("B6 %s XGBoost params: %s", split_name, used_params[split_name])

        split_results, split_frames = run_ablation_for_split(
            cohort,
            assignments,
            split_fp_matrix,
            pca_feature_map,
            pathway_feature_map,
            target_matrix,
            predicted_target_matrix,
            product_matrix,
            split_name=split_name,
            target=target,
            ablations=ablations,
            xgb_params=split_params,
            device=device,
            variant=variant,
        )
        all_results.extend(split_results)
        all_frames.extend(split_frames)

    metrics = pd.DataFrame(all_results)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)
    write_predictions(all_frames, predictions_output, variant=variant)

    summary_data = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_path": str(cohort_path),
        "expression_path": str(expression_path),
        "split_dir": str(split_dir),
        "output": str(output),
        "predictions_output": str(predictions_output),
        "b2_tuned": None if b2_tuned is None else str(b2_tuned),
        "target": target,
        "n_pca_components": n_components,
        "gene_sets": str(gene_sets),
        "min_pathway_genes": min_pathway_genes,
        "pathways": pathway_names,
        "ablations": list(ablations),
        "shuffle_targets": shuffle_targets,
        "shuffle_seed": shuffle_seed,
        "target_view": target_view,
        "target_annotations": target_annotations,
        "permute_cell_features": permute_cell_features,
        "permute_drug_features": permute_drug_features,
        "permutation_seed": permutation_seed,
        "xgb_params_fallback": {k: v for k, v in xgb_params.items() if k != "n_jobs"},
        "xgb_params_by_split": used_params,
        "device_requested": device,
        "splits": list(split_names),
        "best_by_split_subset": summarize_best(metrics),
    }
    summary_path = Path(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_data, handle, indent=2)
    return metrics


def summarize_best(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize the best ablation for each split/subset."""

    rows: list[dict[str, Any]] = []
    for (split_name, subset), group in metrics.groupby(["split_name", "subset"]):
        best = group.sort_values("rmse").iloc[0]
        rows.append(
            {
                "split_name": split_name,
                "subset": subset,
                "best_model": best["model"],
                "rmse": float(best["rmse"]),
                "mae": float(best["mae"]),
                "pearson": float(best["pearson"]) if pd.notna(best["pearson"]) else None,
                "spearman": float(best["spearman"]) if pd.notna(best["spearman"]) else None,
                "r2": float(best["r2"]) if pd.notna(best["r2"]) else None,
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description="Run B6 modality ablations.")
    parser.add_argument("--cohort", default="data/processed/cohort_pairs.csv")
    parser.add_argument("--split-dir", default="data/processed/splits")
    parser.add_argument("--expression", default=EXPRESSION_PATH)
    parser.add_argument(
        "--output",
        default="results/ablations/b6_modality_ablation_metrics.csv",
    )
    parser.add_argument(
        "--summary",
        default="data/reports/b6_modality_ablation_summary.json",
    )
    parser.add_argument("--predictions-output", default="results/predictions/b6.csv")
    parser.add_argument(
        "--b2-tuned",
        default="results/baselines/b2_tuning_metrics.csv",
        help="Validation-selected B2 tuning CSV. Used so ablations share B2's hyperparameters.",
    )
    parser.add_argument(
        "--no-b2-tuned",
        action="store_true",
        help="Ignore the B2 tuning file and use the CLI XGBoost hyperparameters instead.",
    )
    parser.add_argument("--target", default="ln_ic50")
    parser.add_argument("--n-components", type=int, default=256)
    parser.add_argument("--gene-sets", default="builtin_cancer_core")
    parser.add_argument("--min-pathway-genes", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for XGBoost and PCA.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=list(KNOWN_SPLIT_CHOICES),
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=list(DEFAULT_ABLATIONS),
        choices=list(ALL_ABLATIONS),
    )
    parser.add_argument(
        "--shuffle-targets",
        action="store_true",
        help=(
            "Permute whole-drug target vectors before fitting. The null for "
            "whether target identity, rather than vector geometry, transfers."
        ),
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help="Seed for --shuffle-targets.",
    )
    parser.add_argument("--permute-cell-features", action="store_true")
    parser.add_argument("--permute-drug-features", action="store_true")
    parser.add_argument("--permutation-seed", type=int, default=42)
    parser.add_argument(
        "--target-view", choices=["raw", "genes", "mechanisms", "typed"], default="raw"
    )
    parser.add_argument(
        "--target-annotations", default="data/mappings/target_annotations_normalized.csv"
    )
    return parser


def main() -> None:
    """Command-line entry point."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = build_parser().parse_args()
    metrics = run_b6_ablation(
        cohort_path=args.cohort,
        split_dir=args.split_dir,
        expression_path=args.expression,
        output=args.output,
        summary=args.summary,
        predictions_output=args.predictions_output,
        b2_tuned=None if args.no_b2_tuned else args.b2_tuned,
        target=args.target,
        n_components=args.n_components,
        gene_sets=args.gene_sets,
        min_pathway_genes=args.min_pathway_genes,
        split_names=tuple(args.splits),
        ablations=tuple(args.ablations),
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        early_stopping_rounds=args.early_stopping_rounds,
        device=args.device,
        random_state=args.random_state,
        shuffle_targets=args.shuffle_targets,
        shuffle_seed=args.shuffle_seed,
        permute_cell_features=args.permute_cell_features,
        permute_drug_features=args.permute_drug_features,
        permutation_seed=args.permutation_seed,
        target_view=args.target_view,
        target_annotations=args.target_annotations,
    )
    print(f"\nWrote B6 ablation metrics to {args.output}")
    print("=" * 72)
    for _, row in metrics.sort_values(["split_name", "subset", "rmse"]).iterrows():
        print(
            f"  {row['split_name']:>14s} / {row['subset']:>10s} / "
            f"{row['model']:>30s}: RMSE={row['rmse']:.3f}  "
            f"MAE={row['mae']:.3f}  Pearson={row['pearson']:.3f}  "
            f"R²={row['r2']:.3f}"
        )


if __name__ == "__main__":
    main()
