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
from threadpoolctl import threadpool_limits

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

# Genes that are effectively constant across the training cell lines make
# StandardScaler divide by almost zero, so a held-out cell line that happens to
# express one of them produces an astronomically large z-score. Observed in
# practice: SMIM10L3 (122526779) has std 6.3e-08 across 501 training lines, and
# cell line ACH-000470 came out at z = 3.05e7, which PCA carried through to
# |component| = 6.2e5 and made B1 ridge predict 28269 against a target range of
# roughly [-9, 12]. Tree models hide this because they never extrapolate; linear
# models and MLPs do not.
MIN_GENE_STD = 1e-3

# Second line of defence for genes that clear the std floor but are still
# near-constant. With ~500 training lines a single legitimate outlier reaches
# |z| of about sqrt(500) = 22, so a cap above that bounds extrapolation without
# touching real training signal. Applied identically to train and test rows.
MAX_ABS_ZSCORE = 30.0

# Freeze BLAS reduction order as well as precision and RNG for new pipelines.
EXPRESSION_BLAS_THREADS = 8


@dataclass
class ExpressionPipeline:
    """Fitted StandardScaler + PCA pipeline for gene expression features."""

    scaler: StandardScaler
    pca: PCA
    gene_columns: list[str]
    n_components: int
    max_abs_zscore: float = MAX_ABS_ZSCORE
    n_genes_dropped_low_variance: int = 0
    # Keep old serialized float32 pipelines readable. New fits explicitly use float64.
    computation_dtype: str = "float32"
    blas_threads: int | None = None

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

    # This release includes alternative sequencing/model-condition entries.
    # Select the declared default biological profile, rather than weighting
    # duplicated cells twice in PCA and overwriting their features by file order.
    if "IsDefaultEntryForModel" in expression.columns:
        defaults = (
            expression["IsDefaultEntryForModel"]
            .astype(str)
            .str.lower()
            .isin(("yes", "true", "1"))
        )
        selected = expression.loc[defaults].copy()
        missing_ids = set(expression["ModelID"]) - set(selected["ModelID"])
        if missing_ids:
            raise ValueError(f"No default expression profile for {len(missing_ids)} models.")
        expression = selected
    if expression["ModelID"].duplicated().any():
        raise ValueError("Expression must contain one default profile per ModelID.")
    if model_ids is not None and model_ids - set(expression["ModelID"]):
        raise ValueError("Some requested cohort models have no expression profile.")
    logger.info("Using %d unique default expression profiles.", len(expression))

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
    random_state: int = 42,
    min_gene_std: float = MIN_GENE_STD,
    max_abs_zscore: float = MAX_ABS_ZSCORE,
) -> ExpressionPipeline:
    """Fit StandardScaler + PCA on training gene expression rows.

    Parameters
    ----------
    expression:
        DataFrame indexed by ``ModelID`` with gene columns only.
        Must contain **only training** cell lines to avoid leakage.
    n_components:
        Number of PCA components to retain.
    random_state:
        Seed for randomized SVD, so multi-seed runs also vary PCA.
    min_gene_std:
        Genes whose standard deviation across the training rows falls below
        this are dropped. See :data:`MIN_GENE_STD`.
    max_abs_zscore:
        Scaled values are clipped to ``[-max_abs_zscore, max_abs_zscore]``.
        See :data:`MAX_ABS_ZSCORE`.

    Returns
    -------
    A fitted :class:`ExpressionPipeline` that can transform new rows.
    """
    # Slice the numpy array rather than the DataFrame: the expression matrix is
    # wide enough that a second full materialization is worth avoiding.
    # Float32 randomized SVD is sensitive to BLAS reduction order even with a
    # fixed RNG seed. Fit the scaler and PCA in double precision; cast only the
    # final model features to float32.
    values = expression.values.astype(np.float64)
    keep = values.std(axis=0) >= min_gene_std
    n_dropped = int((~keep).sum())
    if n_dropped:
        logger.info(
            "Dropping %d of %d genes with training std below %.1e "
            "(they would make StandardScaler divide by near-zero).",
            n_dropped,
            len(keep),
            min_gene_std,
        )
    if not keep.any():
        raise ValueError(
            f"No gene has training std above {min_gene_std:.1e}; "
            "the expression matrix is constant across training cell lines."
        )

    gene_columns = expression.columns[keep].tolist()
    values = values[:, keep]

    n_components = min(n_components, values.shape[0], values.shape[1])

    with threadpool_limits(limits=EXPRESSION_BLAS_THREADS, user_api="blas"):
        scaler = StandardScaler()
        scaled = np.clip(scaler.fit_transform(values), -max_abs_zscore, max_abs_zscore)
        pca = PCA(n_components=n_components, random_state=random_state)
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
        max_abs_zscore=max_abs_zscore,
        n_genes_dropped_low_variance=n_dropped,
        computation_dtype="float64",
        blas_threads=EXPRESSION_BLAS_THREADS,
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
    # Ensure column alignment (handles column reordering, and drops the
    # low-variance genes the pipeline excluded at fit time).
    values = expression[pipeline.gene_columns].values.astype(
        getattr(pipeline, "computation_dtype", "float32")
    )
    with threadpool_limits(limits=getattr(pipeline, "blas_threads", None), user_api="blas"):
        scaled = np.clip(
            pipeline.scaler.transform(values),
            -pipeline.max_abs_zscore,
            pipeline.max_abs_zscore,
        )
        return pipeline.pca.transform(scaled).astype(np.float32)


def build_cell_features(
    expression_path: str,
    cohort: pd.DataFrame,
    train_depmap_ids: set[str],
    *,
    n_components: int = 256,
    random_state: int = 42,
    min_gene_std: float = MIN_GENE_STD,
    max_abs_zscore: float = MAX_ABS_ZSCORE,
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
    random_state:
        Seed forwarded to PCA.
    min_gene_std:
        Forwarded to :func:`fit_expression_pipeline`.
    max_abs_zscore:
        Forwarded to :func:`fit_expression_pipeline`.

    Returns
    -------
    Tuple of ``(pipeline, cell_feature_map)`` where ``cell_feature_map``
    is a dict mapping ``depmap_id`` → PCA feature vector.
    """
    all_ids = set(cohort["depmap_id"].dropna().unique())
    expression = load_expression_matrix(expression_path, model_ids=all_ids)

    # Fit on training cell lines only.
    train_expr = expression.loc[expression.index.isin(train_depmap_ids)]
    pipeline = fit_expression_pipeline(
        train_expr,
        n_components=n_components,
        random_state=random_state,
        min_gene_std=min_gene_std,
        max_abs_zscore=max_abs_zscore,
    )

    # Transform all cell lines that appear in the expression matrix.
    all_transformed = transform_expression(pipeline, expression)

    cell_feature_map: dict[str, np.ndarray] = {}
    for idx, model_id in enumerate(expression.index):
        cell_feature_map[model_id] = all_transformed[idx]

    return pipeline, cell_feature_map
