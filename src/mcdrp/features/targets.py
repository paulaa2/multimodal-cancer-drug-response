"""Multi-hot drug target profiles from GDSC ``putative_target``.

The scientific question is whether annotated protein targets transfer to
unseen drugs better than Morgan fingerprints. On this cohort the fingerprints
made every cold-drug ablation worse than the same ablation without them, so
this representation is the direct next test.

The vocabulary is the set of target tokens that appear on *training* drugs.
A held-out drug can only use targets the model has already seen attached to
another molecule; a target unique to the test set is a zero column it never
trained on, and including it would leak the test vocabulary into the feature
space. Drugs with a missing annotation become the zero vector.

GDSC mixes gene symbols (``EGFR``, ``MEK1, MEK2``) with mechanism phrases
(``Antimetabolite``, ``Microtubule destabiliser``). Both are kept: a shared
mechanism string is still a transferable biological label. Comma-separated
class suffixes such as ``IIa`` after ``HDAC inhibitor Class I`` are dropped
because they are not targets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COLUMN = "putative_target"

# Fragments produced when a class list is comma-split, e.g.
# "HDAC inhibitor Class I, IIa, IIb, IV".
CLASS_FRAGMENTS = frozenset(
    {"i", "ii", "iii", "iv", "iia", "iib", "iic", "others", "other"}
)


def parse_putative_targets(value: object) -> list[str]:
    """Split one GDSC putative-target field into tokens."""

    if not isinstance(value, str):
        return []

    tokens: list[str] = []
    seen: set[str] = set()
    for part in value.replace(";", ",").split(","):
        token = " ".join(part.split())
        if not token or token.casefold() in CLASS_FRAGMENTS:
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


@dataclass
class TargetPipeline:
    """Training-drug vocabulary for multi-hot target profiles."""

    target_names: list[str]
    n_drugs_without_targets: int = 0


def fit_target_pipeline(drug_annotations: pd.Series) -> TargetPipeline:
    """Build the target vocabulary from one row per training drug.

    ``drug_annotations`` is indexed by ``drug_id`` and holds the raw
    ``putative_target`` strings.
    """

    vocab: set[str] = set()
    n_missing = 0
    for value in drug_annotations:
        tokens = parse_putative_targets(value)
        if not tokens:
            n_missing += 1
        vocab.update(tokens)

    target_names = sorted(vocab)
    logger.info(
        "Target vocabulary: %d tokens from %d training drugs (%d without a target)",
        len(target_names),
        len(drug_annotations),
        n_missing,
    )
    if not target_names:
        raise ValueError("Training drugs produced an empty target vocabulary.")
    return TargetPipeline(
        target_names=target_names,
        n_drugs_without_targets=n_missing,
    )


def build_target_matrix(
    cohort: pd.DataFrame,
    train_drug_ids: set[str],
    *,
    column: str = TARGET_COLUMN,
) -> tuple[TargetPipeline, np.ndarray]:
    """Encode every cohort row with a vocabulary fit on training drugs only."""

    if column not in cohort.columns:
        raise ValueError(
            f"Target profiles require a {column} column. Rebuild the cohort "
            "with mcdrp.data.build_cohort so GDSC putative targets are present."
        )

    drug_table = (
        cohort.loc[cohort["drug_id"].astype(str).isin(train_drug_ids), ["drug_id", column]]
        .drop_duplicates("drug_id")
        .set_index("drug_id")[column]
    )
    if drug_table.empty:
        raise ValueError("No training drugs were available to fit target profiles.")

    pipeline = fit_target_pipeline(drug_table)
    index = {name: i for i, name in enumerate(pipeline.target_names)}

    annotations = (
        cohort[["drug_id", column]].drop_duplicates("drug_id").set_index("drug_id")[column]
    )
    drug_vectors: dict[str, np.ndarray] = {}
    for drug_id, value in annotations.items():
        vector = np.zeros(len(pipeline.target_names), dtype=np.float32)
        for token in parse_putative_targets(value):
            position = index.get(token)
            if position is not None:
                vector[position] = 1.0
        drug_vectors[str(drug_id)] = vector

    matrix = np.zeros((len(cohort), len(pipeline.target_names)), dtype=np.float32)
    for i, drug_id in enumerate(cohort["drug_id"].astype(str)):
        vector = drug_vectors.get(drug_id)
        if vector is not None:
            matrix[i] = vector
    return pipeline, matrix
