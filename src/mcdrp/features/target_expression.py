"""Hadamard fusion of a drug's target profile with the cell's expression of those genes.

This exploratory fusion multiplies target annotations by scaled mRNA abundance.
It does not model protein activity, binding affinity, inhibition, or a causal
relationship between expression and sensitivity. Scaling uses training cells only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from mcdrp.features.expression import MAX_ABS_ZSCORE, load_expression_matrix
from mcdrp.features.pathways import expression_column_to_symbol
from mcdrp.features.targets import TargetPipeline

logger = logging.getLogger(__name__)

# GDSC putative-target aliases that are not the DepMap gene symbol.
TARGET_SYMBOL_ALIASES = {
    "MEK1": "MAP2K1",
    "MEK2": "MAP2K2",
    "ERK1": "MAPK3",
    "ERK2": "MAPK1",
    "PARP1": "PARP1",
    "MTORC1": "MTOR",
    "MTORC2": "MTOR",
    "MTOR": "MTOR",
    "PI3KALPHA": "PIK3CA",
    "PI3KBETA": "PIK3CB",
    "PI3KDELTA": "PIK3CD",
    "PI3KGAMMA": "PIK3CG",
    "ABL": "ABL1",
    "HER2": "ERBB2",
    "VEGFR": "KDR",
    "VEGFR2": "KDR",
    "PDGFR": "PDGFRA",
}


@dataclass
class TargetExpressionProduct:
    """Matched target tokens and the gene columns they aligned to."""

    target_names: list[str]
    gene_columns: list[str]
    scaler: StandardScaler


def _token_to_symbol(token: str) -> str | None:
    """Return a gene symbol when the token is a single gene-like name."""

    if " " in token or "(" in token:
        return None
    cleaned = token.replace("-", "").upper()
    return TARGET_SYMBOL_ALIASES.get(cleaned, token.upper())


def fit_target_expression_product(
    expression: pd.DataFrame,
    pipeline: TargetPipeline,
    train_depmap_ids: set[str],
) -> TargetExpressionProduct:
    """Align target tokens to expression columns and scale on training cells."""

    symbol_to_column = {
        expression_column_to_symbol(column): column for column in expression.columns
    }
    target_names: list[str] = []
    gene_columns: list[str] = []
    for token in pipeline.target_names:
        symbol = _token_to_symbol(token)
        if symbol is None:
            continue
        column = symbol_to_column.get(symbol)
        if column is None:
            continue
        target_names.append(token)
        gene_columns.append(column)

    if not gene_columns:
        raise ValueError(
            "No putative-target tokens aligned to expression columns. "
            "The product fusion needs at least one gene-like target."
        )

    train_ids = [cell_id for cell_id in train_depmap_ids if cell_id in expression.index]
    if not train_ids:
        raise ValueError("No training cell lines found in the expression matrix.")

    scaler = StandardScaler()
    scaler.fit(expression.loc[train_ids, gene_columns].to_numpy(dtype=np.float32))
    logger.info(
        "Target-expression product: %d/%d tokens aligned to genes",
        len(gene_columns),
        len(pipeline.target_names),
    )
    return TargetExpressionProduct(
        target_names=target_names,
        gene_columns=gene_columns,
        scaler=scaler,
    )


def transform_target_expression_product(
    fusion: TargetExpressionProduct,
    pipeline: TargetPipeline,
    target_matrix: np.ndarray,
    cohort: pd.DataFrame,
    expression: pd.DataFrame,
) -> np.ndarray:
    """Return one product vector per cohort row, ``t_drug ⊙ e_cell`` on matched genes."""

    token_index = {name: i for i, name in enumerate(pipeline.target_names)}
    columns = [token_index[name] for name in fusion.target_names]
    drug_block = target_matrix[:, columns]

    scaled = np.clip(
        fusion.scaler.transform(
            expression.loc[:, fusion.gene_columns].to_numpy(dtype=np.float32)
        ),
        -MAX_ABS_ZSCORE,
        MAX_ABS_ZSCORE,
    ).astype(np.float32)
    cell_lookup = {str(cell_id): scaled[i] for i, cell_id in enumerate(expression.index)}
    product = np.zeros((len(cohort), len(fusion.gene_columns)), dtype=np.float32)
    for i, cell_id in enumerate(cohort["depmap_id"].astype(str)):
        cell_vec = cell_lookup.get(cell_id)
        if cell_vec is None:
            continue
        product[i] = drug_block[i] * cell_vec
    return product


def build_target_expression_product(
    expression_path: str,
    cohort: pd.DataFrame,
    pipeline: TargetPipeline,
    target_matrix: np.ndarray,
    train_depmap_ids: set[str],
) -> tuple[TargetExpressionProduct, np.ndarray]:
    """Fit the aligned product on training cells and transform the whole cohort."""

    model_ids = set(cohort["depmap_id"].astype(str)) | set(train_depmap_ids)
    expression = load_expression_matrix(expression_path, model_ids=model_ids)
    fusion = fit_target_expression_product(expression, pipeline, train_depmap_ids)
    matrix = transform_target_expression_product(
        fusion, pipeline, target_matrix, cohort, expression
    )
    return fusion, matrix
