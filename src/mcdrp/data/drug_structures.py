"""Build a drug structure table from GDSC drug names.

This command queries PubChem by drug name and writes a local
`data/raw/drug_structures.csv` file with canonical SMILES. The output is an
auditable starting point, not a final truth table: unmatched or ambiguous drugs
should be reviewed manually before modelling.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd


PUBCHEM_PROPERTY_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
    "{name}/property/ConnectivitySMILES,IsomericSMILES,InChIKey/JSON"
)


@dataclass(frozen=True)
class PubChemResult:
    query_name: str
    status: str
    canonical_smiles: str
    isomeric_smiles: str
    inchi_key: str
    cid: str
    source: str
    error: str


def load_cache(path: Path) -> dict[str, PubChemResult]: 

    if not path.exists():
        return {}

    cache: dict[str, PubChemResult] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            cache[item["query_name"].casefold()] = PubChemResult(**item)

    return cache


def append_cache(path: Path, result: PubChemResult) -> None:
    """Append one PubChem lookup result to the JSONL cache."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def query_pubchem_name(name: str, timeout: float) -> PubChemResult:
    """Query PubChem for a compound name."""

    encoded_name = quote(name, safe="")
    url = PUBCHEM_PROPERTY_URL.format(name=encoded_name)
    request = Request(
        url,
        headers={
            "User-Agent": "mcdrp-data-builder/0.1 (educational research project)"
        },
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 404:
            return PubChemResult(
                query_name=name,
                status="not_found",
                canonical_smiles="",
                isomeric_smiles="",
                inchi_key="",
                cid="",
                source="pubchem",
                error="HTTP 404",
            )
        return PubChemResult(
            query_name=name,
            status="error",
            canonical_smiles="",
            isomeric_smiles="",
            inchi_key="",
            cid="",
            source="pubchem",
            error=f"HTTP {exc.code}: {exc.reason}",
        )
    except (TimeoutError, URLError) as exc:
        return PubChemResult(
            query_name=name,
            status="error",
            canonical_smiles="",
            isomeric_smiles="",
            inchi_key="",
            cid="",
            source="pubchem",
            error=str(exc),
        )

    properties = payload.get("PropertyTable", {}).get("Properties", [])
    if not properties:
        return PubChemResult(
            query_name=name,
            status="not_found",
            canonical_smiles="",
            isomeric_smiles="",
            inchi_key="",
            cid="",
            source="pubchem",
            error="No PubChem properties returned",
        )

    first = properties[0]
    canonical_smiles = str(
        first.get("CanonicalSMILES", "") or first.get("ConnectivitySMILES", "") or ""
    )
    isomeric_smiles = str(
        first.get("IsomericSMILES", "") or first.get("SMILES", "") or ""
    )
    status = "matched" if canonical_smiles else "missing_smiles"
    return PubChemResult(
        query_name=name,
        status=status,
        canonical_smiles=canonical_smiles,
        isomeric_smiles=isomeric_smiles,
        inchi_key=str(first.get("InChIKey", "") or ""),
        cid=str(first.get("CID", "") or ""),
        source="pubchem",
        error="",
    )


def load_unique_drugs(input_path: Path) -> pd.DataFrame:
    """Load unique GDSC drugs from the mapping template."""

    table = pd.read_csv(input_path)
    required = {"original_drug_id", "original_drug_name"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {sorted(missing)}")

    return (
        table.loc[
            table["original_drug_name"].notna(),
            ["original_drug_id", "original_drug_name"],
        ]
        .drop_duplicates()
        .sort_values(["original_drug_id", "original_drug_name"])
        .reset_index(drop=True)
    )


def build_drug_structures(
    input_path: str | Path,
    output_path: str | Path,
    cache_path: str | Path,
    *,
    delay_seconds: float = 0.2,
    timeout_seconds: float = 20.0,
    limit: int | None = None,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """Build a drug-to-SMILES table from PubChem lookups."""

    input_file = Path(input_path)
    output_file = Path(output_path)
    cache_file = Path(cache_path)

    drugs = load_unique_drugs(input_file)
    if limit is not None:
        drugs = drugs.head(limit)

    cache = {} if refresh_cache else load_cache(cache_file)
    rows: list[dict[str, str]] = []

    total = len(drugs)
    for position, drug in enumerate(drugs.to_dict("records"), start=1):
        drug_id = str(drug["original_drug_id"])
        drug_name = str(drug["original_drug_name"])
        cache_key = drug_name.casefold()

        if cache_key in cache:
            result = cache[cache_key]
        else:
            result = query_pubchem_name(drug_name, timeout_seconds)
            append_cache(cache_file, result)
            cache[cache_key] = result
            time.sleep(delay_seconds)

        rows.append(
            {
                "drug_id": drug_id,
                "drug_name": drug_name,
                "canonical_smiles": result.canonical_smiles,
                "isomeric_smiles": result.isomeric_smiles,
                "inchi_key": result.inchi_key,
                "pubchem_cid": result.cid,
                "mapping_status": result.status,
                "source": result.source,
                "query_name": result.query_name,
                "notes": result.error,
            }
        )

        if position % 25 == 0 or position == total:
            print(f"Processed {position}/{total} drugs", flush=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(rows)
    output.to_csv(output_file, index=False)
    return output


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Create data/raw/drug_structures.csv from PubChem lookups."
    )
    parser.add_argument(
        "--input",
        default="data/mappings/drug_mapping_template.csv",
        help="Drug mapping template produced by the data audit.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/drug_structures.csv",
        help="Output CSV with drug names and canonical SMILES.",
    )
    parser.add_argument(
        "--cache",
        default="data/interim/pubchem_smiles_cache.jsonl",
        help="JSONL cache for PubChem lookup results.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay between uncached PubChem requests.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout in seconds for each PubChem request.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for test runs.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore existing cache and query PubChem again.",
    )
    return parser


def main() -> None:
    """Command-line entry point."""

    args = build_parser().parse_args()
    output = build_drug_structures(
        args.input,
        args.output,
        args.cache,
        delay_seconds=args.delay,
        timeout_seconds=args.timeout,
        limit=args.limit,
        refresh_cache=args.refresh_cache,
    )
    counts = output["mapping_status"].value_counts().to_dict()
    print(f"Wrote {len(output)} rows to {args.output}")
    print(f"Mapping status counts: {counts}")


if __name__ == "__main__":
    main()
