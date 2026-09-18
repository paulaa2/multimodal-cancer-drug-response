"""

Loads the DepMap expression matrix, selects gene columns, and produces a
reduced-dimension PCA representation.  Scaling and PCA are fit on training
cell lines only to avoid data leakage.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Metadata columns in the DepMap expression CSV that are not gene features.
_METADATA_COLUMNS = frozenset(
    {
        "Unnamed: 0",
        "SequencingID",
        "ModelConditionID",
        "ModelID",
        "IsDefaultEntryForMC",
        "IsDefaultEntryForModel",
    }
)

_GENE_COLUMN_RE = re.compile(r".+\(\d+\)$")


@dataclass
class ExpressionPipeline:
    """Fitted StandardScaler + PCA pipeline for gene expression features.
    """

    scaler: StandardScaler
    pca: PCA
    gene_columns: list[str]
    n_components: int

    @property
    def explained_variance_ratio_sum(self) -> float:
        return float(self.pca.explained_variance_ratio_.sum())


def load_expression_matrix(
    path: str,
    model_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Load DepMap expression matrix, keeping only gene columns and ModelID.

    Parameters
    ----------
    path:
        Path to the DepMap expression CSV.
    model_ids:
        If provided, keep only rows whose ``ModelID`` is in this set.
        This reduces memory usage substantially for large matrices.

    Returns
    -------
    DataFrame indexed by ``ModelID`` with gene expression columns only.
    """
    logger.info("Loading expression matrix from %s ...", path)
    expression = pd.read_csv(path)

    # First pass: exclude known metadata columns and keep DepMap gene columns.
    # Gene columns are named like "TSPAN6 (7105)"; metadata columns can contain
    # strings such as "Yes" and should never enter the numeric matrix.
    candidate_columns = [
        c
        for c in expression.columns
        if c not in _METADATA_COLUMNS and _GENE_COLUMN_RE.fullmatch(c)
    ]

    # Second pass: keep only columns that are fully numeric.
    # pd.to_numeric with errors='coerce' turns non-parseable values to NaN;
    # a column with any NaN after coercion (that had no NaN before) is non-numeric.
    gene_columns: list[str] = []
    dropped: list[str] = []
    for col in candidate_columns:
        original_na = expression[col].isna().sum()
        coerced_na = pd.to_numeric(expression[col], errors="coerce").isna().sum()
        if coerced_na > original_na:
            dropped.append(col)
        else:
            gene_columns.append(col)

    if dropped:
        logger.info("Dropped %d non-numeric candidate columns: %s", len(dropped), dropped)

    logger.info(
        "Expression matrix: %d rows × %d gene columns.",
        len(expression),
        len(gene_columns),
    )

    # Filter to relevant cell lines.
    expression["ModelID"] = expression["ModelID"].astype(str)
    if model_ids is not None:
        before = len(expression)
        expression = expression.loc[expression["ModelID"].isin(model_ids)].copy()
        logger.info(
            "Filtered expression to %d / %d rows matching cohort model IDs.",
            len(expression),
            before,
        )

    expression = expression.set_index("ModelID")[gene_columns]
    expression = expression.apply(pd.to_numeric, errors="coerce")

    # Fill any rare NaNs with 0 (unexpressed gene).
    n_nan = int(expression.isna().sum().sum())
    if n_nan > 0:
        logger.warning("Filling %d NaN values in expression matrix with 0.", n_nan)
        expression = expression.fillna(0.0)

    return expression



def fit_expression_pipeline(
    expression: pd.DataFrame,
    *,
    n_components: int = 256,
) -> ExpressionPipeline:
    """Fit StandardScaler + PCA on training gene expression rows.

    Parameters
    ----------
    expression:
        DataFrame indexed by ``ModelID`` with gene columns only.
        Must contain **only training** cell lines to avoid leakage.
    n_components:
        Number of PCA components to retain.

    Returns
    -------
    A fitted :class:`ExpressionPipeline` that can transform new rows.
    """
    gene_columns = list(expression.columns)
    values = expression.values.astype(np.float32)

    n_components = min(n_components, values.shape[0], values.shape[1])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(scaled)

    explained = pca.explained_variance_ratio_.sum()
    logger.info(
        "PCA fitted: %d components explain %.1f%% of variance "
        "(from %d training cell lines × %d genes).",
        n_components,
        explained * 100,
        values.shape[0],
        values.shape[1],
    )

    return ExpressionPipeline(
        scaler=scaler,
        pca=pca,
        gene_columns=gene_columns,
        n_components=n_components,
    )


def transform_expression(
    pipeline: ExpressionPipeline,
    expression: pd.DataFrame,
) -> np.ndarray:
    """Transform expression rows using an already-fitted pipeline.

    Parameters
    ----------
    pipeline:
        A pipeline previously fit on training rows.
    expression:
        DataFrame indexed by ``ModelID`` with the same gene columns.

    Returns
    -------
    np.ndarray of shape ``(n_rows, n_components)`` with dtype ``float32``.
    """
    # Ensure column alignment (handles column reordering or subsetting).
    values = expression[pipeline.gene_columns].values.astype(np.float32)
    scaled = pipeline.scaler.transform(values)
    return pipeline.pca.transform(scaled).astype(np.float32)


def build_cell_features(
    expression_path: str,
    cohort: pd.DataFrame,
    train_depmap_ids: set[str],
    *,
    n_components: int = 256,
) -> tuple[ExpressionPipeline, dict[str, np.ndarray]]:
    """End-to-end helper: load expression, fit PCA on train, transform all.

    Parameters
    ----------
    expression_path:
        Path to the DepMap expression CSV.
    cohort:
        Full cohort DataFrame with ``depmap_id`` column.
    train_depmap_ids:
        Set of ``depmap_id`` values belonging to training rows.
    n_components:
        PCA components.

    Returns
    -------
    Tuple of ``(pipeline, cell_feature_map)`` where ``cell_feature_map``
    is a dict mapping ``depmap_id`` → PCA feature vector.
    """
    all_ids = set(cohort["depmap_id"].dropna().unique())
    expression = load_expression_matrix(expression_path, model_ids=all_ids)

    # Fit on training cell lines only.
    train_expr = expression.loc[expression.index.isin(train_depmap_ids)]
    pipeline = fit_expression_pipeline(train_expr, n_components=n_components)

    # Transform all cell lines that appear in the expression matrix.
    all_transformed = transform_expression(pipeline, expression)

    cell_feature_map: dict[str, np.ndarray] = {}
    for idx, model_id in enumerate(expression.index):
        cell_feature_map[model_id] = all_transformed[idx]

    return pipeline, cell_feature_map
