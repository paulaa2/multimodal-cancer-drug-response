"""Entity-level feature nulls that preserve partition membership and repeated rows."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def permute_entity_features(
    matrix: np.ndarray,
    ids: Sequence[str],
    labels: Sequence[str],
    *,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Permute within identical entity partition-membership signatures.

    All pairs for an entity receive the same donor vector. A training-only
    entity can never borrow a held-out-only entity's features. Shared entities
    remain shared; singleton strata are explicitly fixed points.
    """

    ids = np.asarray([str(value) for value in ids], dtype=object)
    labels = np.asarray(labels, dtype=object)
    if matrix.ndim != 2 or len(matrix) != len(ids) or len(labels) != len(ids):
        raise ValueError("Entity features, IDs and partition labels must align.")
    if not np.isfinite(matrix).all():
        raise ValueError("Permutation features must be finite.")
    unique, first, inverse = np.unique(ids, return_index=True, return_inverse=True)
    vectors = matrix[first]
    if not np.array_equal(matrix, vectors[inverse]):
        raise ValueError("One entity has inconsistent feature vectors across pairs.")
    signatures = (
        pd.DataFrame({"entity": ids, "label": labels})
        .groupby("entity")["label"]
        .agg(lambda values: ",".join(sorted(set(values))))
    )
    groups = np.asarray([signatures[value] for value in unique])
    donors = np.arange(len(unique))
    rng = np.random.default_rng(seed)
    for signature in sorted(set(groups)):
        indices = np.flatnonzero(groups == signature)
        donors[indices] = rng.permutation(indices)
    mapping = pd.DataFrame(
        {
            "entity_id": unique,
            "donor_id": unique[donors],
            "partition_signature": groups,
            "fixed_point": unique == unique[donors],
        }
    )
    return vectors[donors][inverse], mapping


def permute_cell_map(
    feature_map: dict[str, np.ndarray],
    cohort: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    seed: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    aligned = cohort[["pair_id", "depmap_id"]].merge(
        assignments, on="pair_id", validate="one_to_one"
    )
    matrix = np.stack([feature_map[str(key)] for key in aligned["depmap_id"]])
    permuted, mapping = permute_entity_features(
        matrix, aligned["depmap_id"].tolist(), aligned["split"].tolist(), seed=seed
    )
    first = ~aligned["depmap_id"].duplicated()
    return {
        str(key): vector
        for key, vector in zip(
            aligned.loc[first, "depmap_id"], permuted[first.to_numpy()], strict=True
        )
    }, mapping
