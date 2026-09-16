"""Pathway-level expression features.

This module converts gene expression into pathway activity scores in a
leakage-safe way: gene scaling is fit on training cell lines only, then pathway
scores are computed for every cell line by averaging scaled member-gene values.

It supports external GMT files such as MSigDB Hallmark, and also includes a
small built-in cancer pathway panel so the pipeline can be exercised without
redistributing licensed pathway resources.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from mcdrp.features.expression import load_expression_matrix

logger = logging.getLogger(__name__)

_GENE_SYMBOL_RE = re.compile(r"^(.+?)\s+\(\d+\)$")


BUILTIN_CANCER_CORE: dict[str, tuple[str, ...]] = {
    "MAPK_SIGNALING": (
        "EGFR",
        "ERBB2",
        "GRB2",
        "KRAS",
        "NRAS",
        "BRAF",
        "RAF1",
        "MAP2K1",
        "MAP2K2",
        "MAPK1",
        "MAPK3",
        "DUSP6",
        "SPRY2",
    ),
    "PI3K_AKT_MTOR": (
        "PIK3CA",
        "PIK3CB",
        "PIK3R1",
        "PTEN",
        "AKT1",
        "AKT2",
        "MTOR",
        "RICTOR",
        "RPTOR",
        "TSC1",
        "TSC2",
        "FOXO3",
    ),
    "P53_APOPTOSIS": (
        "TP53",
        "MDM2",
        "CDKN1A",
        "BAX",
        "BBC3",
        "PMAIP1",
        "CASP3",
        "CASP8",
        "CASP9",
        "FAS",
    ),
    "DNA_DAMAGE_REPAIR": (
        "BRCA1",
        "BRCA2",
        "ATM",
        "ATR",
        "CHEK1",
        "CHEK2",
        "RAD51",
        "PARP1",
        "XRCC1",
        "MSH2",
        "MLH1",
    ),
    "CELL_CYCLE": (
        "CCND1",
        "CCNE1",
        "CDK2",
        "CDK4",
        "CDK6",
        "RB1",
        "E2F1",
        "AURKA",
        "AURKB",
        "MKI67",
    ),
    "EMT_INVASION": (
        "VIM",
        "CDH1",
        "CDH2",
        "SNAI1",
        "SNAI2",
        "ZEB1",
        "ZEB2",
        "TWIST1",
        "FN1",
        "MMP2",
        "MMP9",
    ),
    "HYPOXIA_ANGIOGENESIS": (
        "HIF1A",
        "EPAS1",
        "VEGFA",
        "KDR",
        "FLT1",
        "ANGPT2",
        "SLC2A1",
        "CA9",
        "LDHA",
    ),
    "INFLAMMATION_NFKB": (
        "NFKB1",
        "RELA",
        "IKBKB",
        "TNF",
        "IL6",
        "IL1B",
        "CXCL8",
        "CCL2",
        "JUN",
        "FOS",
    ),
}


@dataclass(frozen=True)
class PathwayDefinition:
    """One pathway and its gene members."""

    name: str
    genes: tuple[str, ...]


@dataclass
class PathwayPipeline:
    """Fitted pathway scoring pipeline."""

    scaler: StandardScaler
    gene_columns: list[str]
    pathway_to_indices: dict[str, list[int]]
    min_genes: int

    @property
    def pathway_names(self) -> list[str]:
        return list(self.pathway_to_indices)


def expression_column_to_symbol(column: str) -> str:
    """Extract a gene symbol from a DepMap expression column."""

    match = _GENE_SYMBOL_RE.match(column)
    if match:
        return match.group(1).upper()
    return column.upper()


def load_builtin_pathways() -> list[PathwayDefinition]:
    """Return the built-in compact cancer pathway panel."""

    return [
        PathwayDefinition(name=name, genes=tuple(gene.upper() for gene in genes))
        for name, genes in BUILTIN_CANCER_CORE.items()
    ]


def load_gmt(path: str | Path) -> list[PathwayDefinition]:
    """Load pathway definitions from a GMT file."""

    definitions: list[PathwayDefinition] = []
    gmt_path = Path(path)
    with gmt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = tuple(gene.upper() for gene in parts[2:] if gene)
            definitions.append(PathwayDefinition(name=name, genes=genes))
    if not definitions:
        raise ValueError(f"No pathway definitions found in {gmt_path}.")
    return definitions


def load_pathway_definitions(source: str | Path) -> list[PathwayDefinition]:
    """Load pathway definitions from ``builtin_cancer_core`` or a GMT path."""

    if str(source) == "builtin_cancer_core":
        return load_builtin_pathways()
    return load_gmt(source)


def fit_pathway_pipeline(
    expression: pd.DataFrame,
    pathway_definitions: list[PathwayDefinition],
    *,
    min_genes: int = 3,
) -> PathwayPipeline:
    """Fit gene scaling and map pathway genes to expression columns."""

    gene_columns = list(expression.columns)
    symbols = [expression_column_to_symbol(column) for column in gene_columns]
    symbol_to_indices: dict[str, list[int]] = {}
    for idx, symbol in enumerate(symbols):
        symbol_to_indices.setdefault(symbol, []).append(idx)

    pathway_to_indices: dict[str, list[int]] = {}
    for definition in pathway_definitions:
        indices: list[int] = []
        for gene in definition.genes:
            indices.extend(symbol_to_indices.get(gene.upper(), []))
        unique_indices = sorted(set(indices))
        if len(unique_indices) >= min_genes:
            pathway_to_indices[definition.name] = unique_indices

    if not pathway_to_indices:
        raise ValueError(
            "No pathways met the min_genes threshold after matching expression columns."
        )

    scaler = StandardScaler()
    scaler.fit(expression[gene_columns].to_numpy(dtype=np.float32))
    logger.info(
        "Pathway pipeline fitted: %d pathways retained from %d definitions.",
        len(pathway_to_indices),
        len(pathway_definitions),
    )
    return PathwayPipeline(
        scaler=scaler,
        gene_columns=gene_columns,
        pathway_to_indices=pathway_to_indices,
        min_genes=min_genes,
    )


def transform_pathways(
    pipeline: PathwayPipeline,
    expression: pd.DataFrame,
) -> np.ndarray:
    """Transform expression rows into pathway activity scores."""

    values = expression[pipeline.gene_columns].to_numpy(dtype=np.float32)
    scaled = pipeline.scaler.transform(values)
    scores = np.zeros((len(expression), len(pipeline.pathway_to_indices)), dtype=np.float32)
    for pathway_idx, indices in enumerate(pipeline.pathway_to_indices.values()):
        scores[:, pathway_idx] = scaled[:, indices].mean(axis=1)
    return scores


def build_pathway_features(
    expression_path: str,
    cohort: pd.DataFrame,
    train_depmap_ids: set[str],
    *,
    gene_sets: str | Path = "builtin_cancer_core",
    min_genes: int = 3,
) -> tuple[PathwayPipeline, dict[str, np.ndarray]]:
    """Load expression and build leakage-safe pathway features."""

    all_ids = set(cohort["depmap_id"].dropna().unique())
    expression = load_expression_matrix(expression_path, model_ids=all_ids)
    train_expr = expression.loc[expression.index.isin(train_depmap_ids)]
    definitions = load_pathway_definitions(gene_sets)
    pipeline = fit_pathway_pipeline(
        train_expr,
        definitions,
        min_genes=min_genes,
    )
    transformed = transform_pathways(pipeline, expression)
    feature_map: dict[str, np.ndarray] = {}
    for idx, model_id in enumerate(expression.index):
        feature_map[model_id] = transformed[idx]
    return pipeline, feature_map
