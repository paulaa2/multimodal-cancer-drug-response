"""Refresh target views after a curation change without refitting unrelated controls."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mcdrp.data.target_annotations import apply_target_view, file_sha256
from mcdrp.experiments.roadmap import EXPRESSION, save_case
from mcdrp.features.expression import build_cell_features
from mcdrp.features.targets import build_target_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations", default="data/mappings/target_annotations_normalized.csv"
    )
    parser.add_argument("--root", default="results/roadmap_2_6/curation_corrected")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    cohort_path = "data/processed/cohort_pairs.csv"
    cohort = pd.read_csv(cohort_path, low_memory=False)
    inputs = [cohort_path, EXPRESSION, args.annotations]
    inputs.extend(
        f"data/processed/splits/cv/cold_drug/fold_{fold}/cold_drug.csv" for fold in range(5)
    )
    manifest = {
        "inputs": {path: file_sha256(Path(path)) for path in inputs},
        "source": {
            str(path): file_sha256(path) for path in sorted(Path("src/mcdrp").rglob("*.py"))
        },
    }
    manifest_path = root / "inputs.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Curation refresh inputs/code changed; use a new output root.")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    pca = None
    fitted_cells = None
    zeros = np.zeros((len(cohort), 0), dtype=np.float32)
    for fold in range(5):
        assignments = pd.read_csv(
            f"data/processed/splits/cv/cold_drug/fold_{fold}/cold_drug.csv"
        )
        train = cohort.merge(assignments, on="pair_id").query("split == 'train'")
        cells = set(train["depmap_id"])
        if pca is None or cells != fitted_cells:
            _, pca = build_cell_features(
                EXPRESSION, cohort, cells, n_components=256, random_state=42
            )
            fitted_cells = cells
        drugs = set(train["drug_id"].astype(str))
        for view in ("genes", "mechanisms", "typed"):
            case = root / f"fold_{fold}" / f"targets_{view}"
            if (case / "complete.json").exists():
                continue
            viewed = apply_target_view(cohort, args.annotations, view)
            _, matrix = build_target_matrix(viewed, drugs)
            save_case(
                cohort,
                assignments,
                zeros,
                pca,
                matrix,
                case,
                split_name="cold_drug",
                target="ln_ic50",
                models=("targets_expression_pca",),
                variant=f"targets_{view}",
            )
            complete_path = case / "complete.json"
            complete = json.loads(complete_path.read_text())
            complete["annotation_sha256"] = file_sha256(Path(args.annotations))
            complete_path.write_text(json.dumps(complete, indent=2) + "\n")


if __name__ == "__main__":
    main()
