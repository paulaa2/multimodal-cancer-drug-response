"""Normalize HGNC genes while keeping families, mechanisms and ambiguity explicit."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from mcdrp.data.audit import file_sha256
from mcdrp.features.targets import parse_putative_targets

FAMILIES = frozenset(
    {
        "PI3K",
        "MEK",
        "ERK",
        "RAF",
        "AKT",
        "JNK",
        "CDK",
        "HDAC",
        "BRD",
        "FGFR",
        "VEGFR",
        "PDGFR",
        "HSP90",
        "PIM",
        "SRC",
        "BCL2 FAMILY",
    }
)
MECHANISM = re.compile(
    r"inhibit|metabolite|stabilis|stabiliz|destabilis|destabiliz|dna |dna$|"
    r"proteasome|microtubule|oxidative|synthesis|replication|crosslink|kinase family",
    re.IGNORECASE,
)


def gene_lookup(hgnc: pd.DataFrame) -> tuple[dict[str, str], dict[str, set[str]]]:
    approved = hgnc.loc[hgnc["status"].eq("Approved")]
    symbols = {str(symbol).casefold(): str(symbol) for symbol in approved["symbol"]}
    aliases: dict[str, set[str]] = defaultdict(set)
    for row in approved.to_dict("records"):
        for column in ("alias_symbol", "prev_symbol"):
            value = row.get(column)
            if isinstance(value, str):
                for alias in value.split("|"):
                    aliases[alias.casefold()].add(row["symbol"])
    return symbols, dict(aliases)


def classify_token(
    token: str, symbols: dict[str, str], aliases: dict[str, set[str]]
) -> dict[str, Any]:
    key = token.casefold()
    if key in symbols:
        return {"kind": "gene", "normalized": symbols[key], "resolution": "approved_symbol"}
    # Pharmacological family labels must not become a single gene just because
    # HGNC lists the generic term as an alias for one family member.
    if token.upper() in FAMILIES:
        return {
            "kind": "family",
            "normalized": token.upper(),
            "resolution": "curated_family_label",
        }
    candidates = sorted(aliases.get(key, set()))
    if len(candidates) == 1:
        return {"kind": "gene", "normalized": candidates[0], "resolution": "unique_hgnc_alias"}
    if candidates:
        return {
            "kind": "unresolved",
            "normalized": token,
            "resolution": "ambiguous_hgnc_alias",
            "candidates": "|".join(candidates),
        }
    if MECHANISM.search(token):
        return {"kind": "mechanism", "normalized": key, "resolution": "mechanism_text"}
    return {"kind": "unresolved", "normalized": token, "resolution": "not_resolved"}


def audit_targets(
    cohort_path: str = "data/processed/cohort_pairs.csv",
    hgnc_path: str = "data/external/references/hgnc_complete_set_2026-09-30.tsv",
    output: str = "data/mappings/target_annotations_normalized.csv",
    summary: str = "data/reports/target_annotations_summary.json",
) -> dict[str, Any]:
    cohort = pd.read_csv(cohort_path, low_memory=False)
    cohort["drug_id"] = cohort["drug_id"].astype(str)
    annotations = cohort[["drug_id", "putative_target"]].drop_duplicates()
    # Contradictory annotations require curation instead of first-row selection.
    if annotations["drug_id"].duplicated().any():
        raise ValueError("A drug has contradictory target annotations.")
    symbols, aliases = gene_lookup(pd.read_csv(hgnc_path, sep="\t", low_memory=False))
    rows = []
    for drug_id, value in annotations.itertuples(index=False, name=None):
        tokens = parse_putative_targets(value)
        if not tokens:
            rows.append(
                {
                    "drug_id": drug_id,
                    "raw_token": "",
                    "kind": "missing",
                    "normalized": "",
                    "resolution": "missing_annotation",
                }
            )
        for token in tokens:
            rows.append(
                {
                    "drug_id": drug_id,
                    "raw_token": token,
                    **classify_token(token, symbols, aliases),
                }
            )
    table = pd.DataFrame(rows)
    table["annotation_source"] = "GDSC2 cohort putative_target"
    table["curation_date"] = "not_provided_by_input"
    table["hgnc_snapshot"] = Path(hgnc_path).name
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    report = {
        "cohort": cohort_path,
        "cohort_sha256": file_sha256(Path(cohort_path)),
        "hgnc": hgnc_path,
        "hgnc_sha256": file_sha256(Path(hgnc_path)),
        "output": output,
        "output_sha256": file_sha256(Path(output)),
        "n_drugs": annotations["drug_id"].nunique(),
        "token_counts": table["kind"].value_counts().to_dict(),
        "resolution_counts": table["resolution"].value_counts().to_dict(),
        "n_drugs_with_genes": int(table.loc[table["kind"].eq("gene"), "drug_id"].nunique()),
        "notes": [
            "Gene aliases resolve only when unique; families are not expanded into "
            "guessed genes.",
            "Annotation date is unknown; cohort release/hash is provenance, "
            "not a curation date.",
            "Views are features for ablations, not causal targets validated "
            "by these experiments.",
        ],
    }
    Path(summary).parent.mkdir(parents=True, exist_ok=True)
    Path(summary).write_text(json.dumps(report, indent=2) + "\n")
    return report


def apply_target_view(cohort: pd.DataFrame, table_path: str, view: str) -> pd.DataFrame:
    if view == "raw":
        return cohort
    if view not in {"genes", "mechanisms", "typed"}:
        raise ValueError("Target view must be raw, genes, mechanisms or typed.")
    table = pd.read_csv(table_path, dtype={"drug_id": str}).fillna("")
    if view == "genes":
        table = table.loc[table["kind"].eq("gene")].copy()
    elif view == "mechanisms":
        table = table.loc[table["kind"].isin(("family", "mechanism"))].copy()
    else:
        table = table.loc[~table["kind"].eq("missing")].copy()
        table["normalized"] = table["kind"] + ":" + table["normalized"]
    encoded = table.groupby("drug_id")["normalized"].agg(
        lambda values: ", ".join(sorted(set(values)))
    )
    result = cohort.copy()
    result["putative_target"] = result["drug_id"].astype(str).map(encoded).fillna("")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort", dest="cohort_path", default="data/processed/cohort_pairs.csv"
    )
    parser.add_argument(
        "--hgnc",
        dest="hgnc_path",
        default="data/external/references/hgnc_complete_set_2026-09-30.tsv",
    )
    parser.add_argument("--output", default="data/mappings/target_annotations_normalized.csv")
    parser.add_argument("--summary", default="data/reports/target_annotations_summary.json")
    return parser


def main() -> None:
    print(
        json.dumps(
            audit_targets(**vars(build_parser().parse_args()))["resolution_counts"], indent=2
        )
    )


if __name__ == "__main__":
    main()
