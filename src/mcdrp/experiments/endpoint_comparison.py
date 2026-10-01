"""Published versus harmonized endpoints with exactly the same retained observations."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from mcdrp.experiments.roadmap import EXPRESSION, save_case
from mcdrp.features.expression import build_cell_features
from mcdrp.features.fingerprints import build_fingerprint_matrix
from mcdrp.features.targets import build_target_matrix
from mcdrp.models.baseline_b0 import run_b0


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    source = "data/processed/harmonized_transfer_pairs.csv"
    output = Path("results/roadmap_2_6/harmonized/published_matched")
    output.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(source, low_memory=False)
    cohort["auc"] = cohort["published_auc"]
    cohort.to_csv(output / "cohort.csv", index=False)
    assignments = pd.read_csv("data/processed/harmonized_transfer/transfer_prism.csv")
    train = cohort.merge(assignments, on="pair_id").query("split == 'train'")
    _, pca = build_cell_features(
        EXPRESSION, cohort, set(train["depmap_id"]), n_components=256, random_state=42
    )
    _, targets = build_target_matrix(cohort, set(train["drug_id"].astype(str)))
    fp = build_fingerprint_matrix(cohort["canonical_smiles"])
    save_case(
        cohort,
        assignments,
        fp,
        pca,
        targets,
        output,
        split_name="transfer_prism",
        target="auc",
        variant="published_matched",
    )
    run_b0(
        cohort_path=output / "cohort.csv",
        split_dir=output / "splits",
        target="auc",
        split_names=("transfer_prism",),
        output=output / "b0_metrics.csv",
        predictions_output=output / "b0.csv",
        summary=output / "b0_summary.json",
    )
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "scope": "Same pairs and fitting settings as shared-window experiment; "
                "published AUC remains a separate endpoint."
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
