"""Require an exact checkpoint, chemistry identity and hashes for frozen embeddings."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mcdrp.data.target_annotations import file_sha256


def audit_embeddings(cohort_path: str, embeddings_path: str, manifest_path: str) -> dict:
    cohort = pd.read_csv(cohort_path, dtype={"drug_id": str})
    table = pd.read_csv(embeddings_path, dtype={"drug_id": str})
    manifest = json.loads(Path(manifest_path).read_text())
    revision = manifest.get("resolved_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError(
            "Embedding provenance needs a resolved 40-character checkpoint commit."
        )
    if file_sha256(Path(embeddings_path)) != manifest.get("output_sha256") or file_sha256(
        Path(cohort_path)
    ) != manifest.get("cohort_sha256"):
        raise ValueError("Embedding/cohort content does not match the manifest hashes.")
    if table["drug_id"].isna().any() or table["drug_id"].duplicated().any():
        raise ValueError("Embedding IDs must be non-null and unique.")
    if set(table["drug_id"]) != set(cohort["drug_id"]):
        raise ValueError("Frozen embeddings must cover exactly the cohort drugs.")
    expected = cohort[["drug_id", "canonical_smiles"]].drop_duplicates()
    if expected["drug_id"].duplicated().any():
        raise ValueError("Cohort has contradictory drug chemistry.")
    joined = expected.merge(
        table, on="drug_id", suffixes=("_cohort", "_embedding"), validate="one_to_one"
    )
    if not joined["canonical_smiles_cohort"].eq(joined["canonical_smiles_embedding"]).all():
        raise ValueError("Embedding SMILES do not match cohort identities.")
    for column, expected_value in (
        ("encoder_revision", revision),
        ("encoder_name", manifest["model_name"]),
        ("pooling", manifest["pooling"]),
    ):
        if not table[column].eq(expected_value).all():
            raise ValueError(f"Mixed or inconsistent {column} provenance.")
    columns = [key for key in table if key.startswith("emb_")]
    if (
        len(columns) != manifest["n_embedding_dimensions"]
        or not np.isfinite(table[columns].to_numpy()).all()
    ):
        raise ValueError("Embedding dimensions or finite-value checks failed.")
    return {
        "status": "verified",
        "resolved_revision": revision,
        "n_drugs": len(table),
        "n_dimensions": len(columns),
        "output_sha256": manifest["output_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default="data/processed/cohort_pairs.csv")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_embeddings(args.cohort, args.embeddings, args.manifest)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
