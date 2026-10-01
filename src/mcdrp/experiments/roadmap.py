"""Execute the frozen roadmap 2--6 comparisons with per-task resumable artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mcdrp.data.target_annotations import apply_target_view, file_sha256
from mcdrp.features.expression import build_cell_features
from mcdrp.features.fingerprints import build_fingerprint_matrix
from mcdrp.features.permutations import permute_cell_map, permute_entity_features
from mcdrp.features.targets import build_target_matrix
from mcdrp.models.ablation_b6 import run_ablation_for_split
from mcdrp.models.baseline_b0 import run_b0
from mcdrp.models.random_forest import run_random_forest
from mcdrp.results.metrics_report import build_metrics_report
from mcdrp.results.predictions import load_predictions, write_predictions
from mcdrp.results.uncertainty import uncertainty_report
from mcdrp.splits.calibration import reserve_calibration

logger = logging.getLogger(__name__)
EXPRESSION = "data/raw/OmicsExpressionTPMLogp1HumanProteinCodingGenesStranded.csv"
PARAMS = {
    "n_estimators": 700,
    "learning_rate": 0.05,
    "max_depth": 6,
    "early_stopping_rounds": 40,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": 8,
    "device": "cuda",
}
MODELS = ("targets_expression_pca", "morgan_expression_pca")


def save_case(
    cohort,
    assignments,
    fp,
    pca,
    targets,
    output: Path,
    *,
    split_name: str,
    target: str,
    models=MODELS,
    variant: str = "standard",
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    split_dir = output / "splits"
    split_dir.mkdir(exist_ok=True)
    assignments.to_csv(split_dir / f"{split_name}.csv", index=False)
    zeros = np.zeros((len(cohort), 0), dtype=np.float32)
    _, frames = run_ablation_for_split(
        cohort,
        assignments,
        fp,
        pca,
        {},
        targets,
        zeros,
        zeros,
        split_name=split_name,
        target=target,
        ablations=models,
        xgb_params=PARAMS,
        device="cuda",
        variant=variant,
    )
    predictions = write_predictions(frames, output / "predictions.csv")
    metrics = build_metrics_report(predictions, cohort, split_dir, target=target)
    metrics.to_csv(output / "metrics.csv", index=False)
    (output / "complete.json").write_text(
        json.dumps(
            {
                "params": PARAMS,
                "target": target,
                "models": models,
                "variant": variant,
                "metrics": json.loads(metrics.to_json(orient="records")),
            },
            indent=2,
        )
        + "\n"
    )


def ablations(folds: list[int], root: Path) -> None:
    cohort = pd.read_csv("data/processed/cohort_pairs.csv", low_memory=False)
    fp = build_fingerprint_matrix(cohort["canonical_smiles"])
    annotation_path = "data/mappings/target_annotations_normalized.csv"
    for fold in folds:
        out = root / "ablation" / f"fold_{fold}"
        assignments = pd.read_csv(
            f"data/processed/splits/cv/cold_drug/fold_{fold}/cold_drug.csv"
        )
        train = cohort.merge(assignments, on="pair_id").query("split == 'train'")
        _, pca = build_cell_features(
            EXPRESSION, cohort, set(train["depmap_id"]), n_components=256, random_state=42
        )
        drugs = set(train["drug_id"].astype(str))
        _, targets = build_target_matrix(cohort, drugs)
        if not (out / "standard" / "complete.json").exists():
            save_case(
                cohort,
                assignments,
                fp,
                pca,
                targets,
                out / "standard",
                split_name="cold_drug",
                target="ln_ic50",
            )
        for view in ("genes", "mechanisms", "typed"):
            case = out / f"targets_{view}"
            if (case / "complete.json").exists():
                continue
            viewed = apply_target_view(cohort, annotation_path, view)
            _, matrix = build_target_matrix(viewed, drugs)
            save_case(
                cohort,
                assignments,
                fp,
                pca,
                matrix,
                case,
                split_name="cold_drug",
                target="ln_ic50",
                models=("targets_expression_pca",),
                variant=f"targets_{view}",
            )
        labels = cohort[["pair_id"]].merge(assignments, on="pair_id")["split"].tolist()
        for seed in (7, 13, 42):
            for block in ("cell", "drug"):
                case = out / f"{block}_permuted_{seed}"
                if (case / "complete.json").exists():
                    continue
                case.mkdir(parents=True, exist_ok=True)
                if block == "cell":
                    features, donors = permute_cell_map(pca, cohort, assignments, seed=seed)
                    donor_fp, donor_targets = fp, targets
                else:
                    features = pca
                    donor_fp, donors = permute_entity_features(
                        fp, cohort["drug_id"].astype(str).tolist(), labels, seed=seed
                    )
                    donor_targets, target_donors = permute_entity_features(
                        targets, cohort["drug_id"].astype(str).tolist(), labels, seed=seed
                    )
                    if not donors.equals(target_donors):
                        raise ValueError("Drug modalities must use the same permutation.")
                donors.to_csv(case / "donors.csv", index=False)
                save_case(
                    cohort,
                    assignments,
                    donor_fp,
                    features,
                    donor_targets,
                    case,
                    split_name="cold_drug",
                    target="ln_ic50",
                    variant=f"{block}_permuted_{seed}",
                )
        case = out / "calibration"
        if not (case / "complete.json").exists():
            reserved = reserve_calibration(
                cohort, assignments, group="drug_id", fraction=0.15, seed=42
            )
            cal_train = cohort.merge(reserved, on="pair_id").query("split == 'train'")
            _, cal_pca = build_cell_features(
                EXPRESSION,
                cohort,
                set(cal_train["depmap_id"]),
                n_components=256,
                random_state=42,
            )
            _, cal_targets = build_target_matrix(cohort, set(cal_train["drug_id"].astype(str)))
            save_case(
                cohort,
                reserved,
                fp,
                cal_pca,
                cal_targets,
                case,
                split_name="cold_drug",
                target="ln_ic50",
                variant="independent_calibration",
            )
        report = uncertainty_report(
            load_predictions(case / "predictions.csv"),
            cohort,
            str(case / "splits"),
            target="ln_ic50",
            group="drug_id",
        )
        (case / "uncertainty.json").write_text(
            json.dumps(json.loads(report.to_json(orient="records")), indent=2) + "\n"
        )
        logger.info("Finished all ablation/control/calibration cases for fold %s", fold)


def harmonized(root: Path) -> None:
    cohort_path = "data/processed/harmonized_transfer_pairs.csv"
    cohort = pd.read_csv(cohort_path, low_memory=False)
    assignments = pd.read_csv("data/processed/harmonized_transfer/transfer_prism.csv")
    train = cohort.merge(assignments, on="pair_id").query("split == 'train'")
    fp = build_fingerprint_matrix(cohort["canonical_smiles"])
    _, pca = build_cell_features(
        EXPRESSION, cohort, set(train["depmap_id"]), n_components=256, random_state=42
    )
    _, targets = build_target_matrix(cohort, set(train["drug_id"].astype(str)))
    case = root / "harmonized"
    if not (case / "standard" / "complete.json").exists():
        save_case(
            cohort,
            assignments,
            fp,
            pca,
            targets,
            case / "standard",
            split_name="transfer_prism",
            target="auc",
        )
    run_b0(
        cohort_path=cohort_path,
        split_dir="data/processed/harmonized_transfer",
        target="auc",
        split_names=("transfer_prism",),
        output=case / "b0_metrics.csv",
        predictions_output=case / "b0.csv",
        summary=case / "b0_summary.json",
    )
    reserved = reserve_calibration(
        cohort, assignments, group="depmap_id", fraction=0.15, seed=42
    )
    cal_case = case / "calibration"
    if not (cal_case / "complete.json").exists():
        cal_train = cohort.merge(reserved, on="pair_id").query("split == 'train'")
        _, cal_pca = build_cell_features(
            EXPRESSION, cohort, set(cal_train["depmap_id"]), n_components=256, random_state=42
        )
        _, cal_targets = build_target_matrix(cohort, set(cal_train["drug_id"].astype(str)))
        save_case(
            cohort,
            reserved,
            fp,
            cal_pca,
            cal_targets,
            cal_case,
            split_name="transfer_prism",
            target="auc",
            variant="independent_calibration",
        )
    report = uncertainty_report(
        load_predictions(cal_case / "predictions.csv"),
        cohort,
        str(cal_case / "splits"),
        target="auc",
        group="depmap_id",
        bounds=(0.0, 1.0),
    )
    (cal_case / "uncertainty.json").write_text(
        json.dumps(json.loads(report.to_json(orient="records")), indent=2) + "\n"
    )


def forests(folds: list[int], root: Path) -> None:
    for fold in folds:
        case = root / "random_forest" / f"fold_{fold}"
        if (case / "summary.json").exists():
            continue
        run_random_forest(
            split_dir=f"data/processed/splits/cv/cold_drug/fold_{fold}",
            output=str(case / "metrics.csv"),
            predictions_output=str(case / "predictions.csv"),
            summary=str(case / "summary.json"),
            split_names=("cold_drug",),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["forests", "ablations", "harmonized"], required=True
    )
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--root", default="results/roadmap_2_6")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Require a new output root; never delete prior results.",
    )
    args = parser.parse_args()
    root = Path(args.root)
    manifest_path = root / f"{args.phase}_inputs.json"
    inputs = [
        "data/processed/cohort_pairs.csv",
        EXPRESSION,
        "data/mappings/target_annotations_normalized.csv",
    ]
    inputs.extend(
        f"data/processed/splits/cv/cold_drug/fold_{fold}/cold_drug.csv" for fold in args.folds
    )
    if args.phase == "harmonized":
        inputs.extend(
            [
                "data/processed/harmonized_transfer_pairs.csv",
                "data/processed/harmonized_transfer/transfer_prism.csv",
            ]
        )
    manifest = {
        "inputs": {path: file_sha256(Path(path)) for path in inputs},
        "params": PARAMS,
        "source": {
            str(path): file_sha256(path) for path in sorted(Path("src/mcdrp").rglob("*.py"))
        },
    }
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(
            "Inputs or code changed: use a new --root instead of mixing resumed experiments."
        )
    if args.fresh and root.exists():
        raise ValueError("--fresh requires a nonexistent output root.")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logging.basicConfig(level=logging.INFO)
    if args.phase == "forests":
        forests(args.folds, root)
    elif args.phase == "ablations":
        ablations(args.folds, root)
    else:
        harmonized(root)


if __name__ == "__main__":
    main()
