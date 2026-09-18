"""Create frozen drug embeddings from canonical SMILES.

This script turns the unique drugs in ``cohort_pairs.csv`` into a CSV of
pretrained molecular embeddings that B7 can consume. It deliberately writes a
plain CSV instead of training-time hidden state caches so the representation is
frozen, inspectable, and easy to compare across experiments.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "DeepChem/ChemBERTa-77M-MLM"


@dataclass(frozen=True)
class DrugRecord:
    """One unique drug and the SMILES used for embedding."""

    drug_id: str
    canonical_smiles: str
    n_response_rows: int


def extract_unique_drugs(
    cohort: pd.DataFrame,
    *,
    drug_id_column: str = "drug_id",
    smiles_column: str = "canonical_smiles",
) -> pd.DataFrame:
    """Return one deterministic SMILES row per drug."""

    required = {drug_id_column, smiles_column}
    missing = required - set(cohort.columns)
    if missing:
        raise ValueError(f"Cohort is missing required columns: {sorted(missing)}")

    data = cohort[[drug_id_column, smiles_column]].dropna().copy()
    data[drug_id_column] = data[drug_id_column].astype(str)
    data[smiles_column] = data[smiles_column].astype(str).str.strip()
    data = data.loc[data[smiles_column].ne("")]
    if data.empty:
        raise ValueError("No valid drug/SMILES rows found in cohort.")

    rows: list[dict[str, Any]] = []
    for drug_id, group in data.groupby(drug_id_column, sort=True):
        counts = group[smiles_column].value_counts()
        smiles = str(counts.index[0])
        if len(counts) > 1:
            logger.warning(
                "Drug %s has %d canonical SMILES values; using most frequent.",
                drug_id,
                len(counts),
            )
        rows.append(
            {
                "drug_id": str(drug_id),
                "canonical_smiles": smiles,
                "n_response_rows": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def resolve_device(device: str) -> str:
    """Resolve auto/cuda/cpu against the installed PyTorch build."""

    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot see a CUDA GPU.")
    if device not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported device: {device}")
    return device


def mean_pool_hidden_state(hidden_state: Any, attention_mask: Any) -> Any:
    """Mean-pool token hidden states while ignoring padding tokens."""

    mask = attention_mask.unsqueeze(-1).type_as(hidden_state)
    summed = (hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def pool_model_output(
    model_output: Any,
    attention_mask: Any,
    *,
    pooling: str,
) -> Any:
    """Pool a transformer output into one vector per molecule."""

    if pooling == "cls":
        return model_output.last_hidden_state[:, 0, :]
    if pooling == "mean":
        return mean_pool_hidden_state(model_output.last_hidden_state, attention_mask)
    raise ValueError(f"Unsupported pooling strategy: {pooling}")


def encode_smiles(
    smiles: list[str],
    *,
    model_name: str = DEFAULT_MODEL,
    revision: str | None = None,
    batch_size: int = 32,
    max_length: int = 256,
    pooling: str = "mean",
    device: str = "auto",
    trust_remote_code: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode SMILES strings with a Hugging Face transformer model."""

    import torch
    from transformers import AutoModel, AutoTokenizer

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if not smiles:
        raise ValueError("No SMILES strings were provided for encoding.")

    actual_device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    model.to(actual_device)
    model.eval()

    embeddings: list[np.ndarray] = []
    n_batches = int(np.ceil(len(smiles) / batch_size))
    logger.info(
        "Encoding %d SMILES with %s on %s (%d batches).",
        len(smiles),
        model_name,
        actual_device,
        n_batches,
    )

    with torch.inference_mode():
        for start in range(0, len(smiles), batch_size):
            batch = smiles[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(actual_device) for key, value in encoded.items()}
            output = model(**encoded)
            pooled = pool_model_output(
                output,
                encoded["attention_mask"],
                pooling=pooling,
            )
            embeddings.append(pooled.detach().cpu().numpy().astype(np.float32))
            logger.info(
                "Encoded batch %d/%d.",
                min(start // batch_size + 1, n_batches),
                n_batches,
            )

    matrix = np.vstack(embeddings).astype(np.float32)
    metadata = {
        "model_name": model_name,
        "revision": revision,
        "device": actual_device,
        "pooling": pooling,
        "max_length": max_length,
        "embedding_dim": int(matrix.shape[1]),
    }
    return matrix, metadata


def write_embedding_table(
    drugs: pd.DataFrame,
    embeddings: np.ndarray,
    output: str | Path,
    *,
    model_name: str,
    revision: str | None,
    pooling: str,
) -> Path:
    """Write B7-compatible embedding CSV."""

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embedding_columns = {
        f"emb_{idx}": embeddings[:, idx] for idx in range(embeddings.shape[1])
    }
    table = pd.concat(
        [
            drugs[["drug_id", "canonical_smiles", "n_response_rows"]].reset_index(
                drop=True
            ),
            pd.DataFrame(
                {
                    "encoder_name": model_name,
                    "encoder_revision": revision or "",
                    "pooling": pooling,
                },
                index=range(len(drugs)),
            ),
            pd.DataFrame(embedding_columns),
        ],
        axis=1,
    )
    table.to_csv(output_path, index=False)
    return output_path


def create_drug_embeddings(
    cohort_path: str | Path = "data/processed/cohort_pairs.csv",
    output: str | Path = "data/external/drug_embeddings.csv",
    summary: str | Path = "data/reports/drug_embeddings_summary.json",
    *,
    model_name: str = DEFAULT_MODEL,
    revision: str | None = None,
    batch_size: int = 32,
    max_length: int = 256,
    pooling: str = "mean",
    device: str = "auto",
    limit: int | None = None,
    trust_remote_code: bool = False,
) -> pd.DataFrame:
    """Create a B7-compatible drug embedding CSV from the cohort."""

    cohort = pd.read_csv(cohort_path)
    drugs = extract_unique_drugs(cohort)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >= 1 when provided.")
        drugs = drugs.head(limit).copy()
        logger.warning("Limiting embedding generation to first %d drugs.", limit)

    embeddings, metadata = encode_smiles(
        drugs["canonical_smiles"].tolist(),
        model_name=model_name,
        revision=revision,
        batch_size=batch_size,
        max_length=max_length,
        pooling=pooling,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    output_path = write_embedding_table(
        drugs,
        embeddings,
        output,
        model_name=model_name,
        revision=revision,
        pooling=pooling,
    )

    summary_data = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_path": str(cohort_path),
        "output": str(output_path),
        "n_drugs": int(len(drugs)),
        "n_embedding_dimensions": int(embeddings.shape[1]),
        "embedding_prefix": "emb_",
        "model_name": model_name,
        "revision": revision,
        "batch_size": batch_size,
        "max_length": max_length,
        "pooling": pooling,
        "device_requested": device,
        "device_used": metadata["device"],
        "trust_remote_code": trust_remote_code,
        "limit": limit,
    }
    summary_path = Path(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_data, handle, indent=2)

    logger.info("Wrote drug embeddings to %s.", output_path)
    logger.info("Wrote embedding summary to %s.", summary_path)
    return pd.read_csv(output_path)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Create B7-compatible pretrained drug embeddings."
    )
    parser.add_argument("--cohort", default="data/processed/cohort_pairs.csv")
    parser.add_argument("--output", default="data/external/drug_embeddings.csv")
    parser.add_argument(
        "--summary",
        default="data/reports/drug_embeddings_summary.json",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--revision",
        default=None,
        help=(
            "Optional Hugging Face model revision, tag, branch, or commit hash. "
            "Use a commit hash for fully frozen embeddings."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional smoke-test limit on number of unique drugs.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom code from the Hugging Face model repository.",
    )
    return parser


def main() -> None:
    """Command-line entry point."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = build_parser().parse_args()
    table = create_drug_embeddings(
        cohort_path=args.cohort,
        output=args.output,
        summary=args.summary,
        model_name=args.model,
        revision=args.revision,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pooling=args.pooling,
        device=args.device,
        limit=args.limit,
        trust_remote_code=args.trust_remote_code,
    )
    embedding_columns = [column for column in table.columns if column.startswith("emb_")]
    print(f"Wrote drug embeddings to {args.output}")
    print(
        f"Encoded {len(table)} drugs with {len(embedding_columns)} dimensions "
        f"using {args.model}."
    )


if __name__ == "__main__":
    main()
