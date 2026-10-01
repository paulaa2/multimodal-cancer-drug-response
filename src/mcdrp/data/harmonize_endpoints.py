"""Recompute viable-fraction AUC over one prespecified shared log10-uM window.

This harmonizes measurement definitions, not exposure duration or assay biology.
Raw drug names only resolve identities through frozen within-study catalogs;
PRISM-to-GDSC chemistry is inherited from the strict InChIKey mapping.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit

from mcdrp.data.audit import file_sha256
from mcdrp.splits.validation import validate_assignments

logger = logging.getLogger(__name__)


def unique_name_lookup(frame: pd.DataFrame, name: str, identifier: str) -> dict[str, str]:
    """Resolve only unique case-insensitive names; punctuation/stereo are retained."""

    pairs = frame[[name, identifier]].dropna().copy()
    pairs[name] = pairs[name].astype(str).str.strip().str.casefold()
    pairs[identifier] = pairs[identifier].astype(str)
    pairs = pairs.drop_duplicates()
    pairs = pairs.loc[~pairs[name].duplicated(keep=False)]
    return dict(pairs.itertuples(index=False, name=None))


def dose_to_log10_um(values: np.ndarray, scale: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Doses must be finite.")
    if scale == "linear_um":
        if (values <= 0).any():
            raise ValueError("Linear doses must be positive.")
        return np.log10(values)
    if scale == "log10_um":
        return values
    raise ValueError("Dose scale must be linear_um or log10_um.")


def fit_shared_auc(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lower: float,
    upper: float,
    min_doses: int = 4,
) -> dict[str, Any]:
    """Fit a decreasing 3PL (top=1, bottom in [0,1]); never extrapolate.

    Responses are not clipped before fitting. Noise outside [0,1] is reported.
    A prespecified constant fallback handles flat profiles without inventing
    identifiable EC50/slope estimates. Fits use all measured doses, whereas
    integration uses only the shared window.
    """

    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or x.shape != y.shape or not np.isfinite([x, y]).all():
        raise ValueError("Curve doses/responses must be aligned finite vectors.")
    if lower >= upper:
        raise ValueError("Shared dose window must have positive width.")
    order = np.argsort(x)
    x, y = x[order], y[order]
    base = {
        "n_doses": len(x),
        "dose_min": float(x.min()) if len(x) else None,
        "dose_max": float(x.max()) if len(x) else None,
        "response_outside_unit_fraction": float(np.mean((y < 0) | (y > 1))) if len(y) else None,
    }
    if len(np.unique(x)) < min_doses:
        return {**base, "status": "too_few_doses", "auc_harmonized": None}
    if x.min() > lower or x.max() < upper:
        return {**base, "status": "insufficient_dose_coverage", "auc_harmonized": None}
    constant = float(np.clip(y.mean(), 0, 1))
    constant_rmse = float(np.sqrt(np.mean((y - constant) ** 2)))
    if np.ptp(y) < 0.05:
        return {**base, "status": "flat", "auc_harmonized": constant, "fit_rmse": constant_rmse}

    def curve(values, params):
        bottom, log_ec50, slope = params
        return bottom + (1 - bottom) * expit(-slope * (values - log_ec50))

    result = least_squares(
        lambda params: curve(x, params) - y,
        x0=[np.clip(y.min(), 0.01, 0.99), float(np.median(x)), 1.0],
        bounds=([0, x.min() - 2, 0.05], [1, x.max() + 2, 10]),
        max_nfev=150,
    )
    if not result.success:
        return {**base, "status": "fit_failed", "auc_harmonized": None}
    rmse = float(np.sqrt(np.mean((curve(x, result.x) - y) ** 2)))
    if rmse >= constant_rmse - 1e-6:
        return {
            **base,
            "status": "constant_fallback",
            "auc_harmonized": constant,
            "fit_rmse": constant_rmse,
        }
    grid = np.linspace(lower, upper, 257)
    # Explicit trapezoidal sum works on supported NumPy 1.x and 2.x versions.
    values = curve(grid, result.x)
    auc = float(np.sum((values[1:] + values[:-1]) * np.diff(grid) / 2) / (upper - lower))
    return {
        **base,
        "status": "sigmoid_3pl",
        "auc_harmonized": auc,
        "fit_rmse": rmse,
        "bottom": float(result.x[0]),
        "log_ec50": float(result.x[1]),
        "slope": float(result.x[2]),
    }


def mapped_doses(
    source: dict[str, Any],
    *,
    cells: dict[str, str],
    drugs: dict[str, str],
    eligible: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    expected_hash = source.get("sha256")
    if expected_hash and file_sha256(Path(source["path"])) != expected_hash:
        raise ValueError(f"Raw input hash changed for {source['study']}.")
    chunks = []
    counts = {
        "raw_rows": 0,
        "selected_screen_rows": 0,
        "mapped_rows": 0,
        "invalid_numeric_rows": 0,
    }
    eligibility = pd.MultiIndex.from_frame(eligible[["depmap_id", "drug_id"]].drop_duplicates())
    for raw in pd.read_csv(source["path"], chunksize=200000, low_memory=False):
        counts["raw_rows"] += len(raw)
        if source.get("replicate_prefix"):
            raw = raw.loc[
                raw["replicate"].astype(str).str.startswith(source["replicate_prefix"])
            ]
        counts["selected_screen_rows"] += len(raw)
        raw = raw.copy()
        raw["depmap_id"] = raw["cell_line"].astype(str).str.strip().str.casefold().map(cells)
        raw["drug_id"] = raw["drug"].astype(str).str.strip().str.casefold().map(drugs)
        raw = raw.loc[raw["depmap_id"].notna() & raw["drug_id"].notna()].copy()
        index = pd.MultiIndex.from_frame(raw[["depmap_id", "drug_id"]])
        raw = raw.loc[index.isin(eligibility)].copy()
        for key in ("dose", "response"):
            raw[key] = pd.to_numeric(raw[key], errors="coerce")
        finite = np.isfinite(raw[["dose", "response"]].to_numpy(float)).all(axis=1)
        if source["dose_scale"] == "linear_um":
            finite &= raw["dose"].gt(0).to_numpy()
        counts["invalid_numeric_rows"] += int((~finite).sum())
        raw = raw.loc[finite].copy()
        raw["log10_um"] = dose_to_log10_um(raw["dose"].to_numpy(), source["dose_scale"])
        raw["study"] = source["study"]
        counts["mapped_rows"] += len(raw)
        chunks.append(
            raw[["study", "depmap_id", "drug_id", "replicate", "log10_um", "response"]]
        )
    if not chunks or not counts["mapped_rows"]:
        raise ValueError(f"No eligible raw dose data for {source['study']}.")
    doses = pd.concat(chunks, ignore_index=True)
    # Technical repeats at a dose are summarized within a replicate first.
    replicate = doses.groupby(
        ["study", "depmap_id", "drug_id", "replicate", "log10_um"], as_index=False
    )["response"].median()
    pooled = replicate.groupby(["study", "depmap_id", "drug_id", "log10_um"], as_index=False)[
        "response"
    ].median()
    return pooled, {**source, **counts, "sha256": file_sha256(Path(source["path"]))}


def harmonize_endpoints(config_path: str = "configs/harmonization.json") -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text())
    cohort = pd.read_csv(config["cohort"], low_memory=False)
    cohort["drug_id"] = cohort["drug_id"].astype(str)
    assignments = pd.read_csv(config["assignments"])
    validate_assignments(cohort, assignments)
    gdsc = cohort.loc[cohort["study"].eq("gdsc2")]
    prism = cohort.loc[cohort["study"].eq("prism")]
    shared = set(prism["drug_id"])
    alias_frames = []
    for name in (
        "gdsc_cell_line_name",
        "depmap_cell_line_name",
        "depmap_stripped_cell_line_name",
    ):
        if name in gdsc:
            alias_frames.append(gdsc[[name, "depmap_id"]].rename(columns={name: "name"}))
    cells = unique_name_lookup(pd.concat(alias_frames), "name", "depmap_id")
    gdsc_drugs = unique_name_lookup(
        gdsc.loc[gdsc["drug_id"].isin(shared)], "gdsc_drug_name", "drug_id"
    )
    strict = pd.read_csv(config["drug_map"])
    strict = strict.loc[strict["drug_id"].notna()].copy()
    strict["drug_id"] = strict["drug_id"].astype(int).astype(str)
    raw_catalog = pd.read_csv(config["prism_metadata"], usecols=["name", "smiles"])
    catalog = raw_catalog.merge(
        strict[["prism_smiles", "drug_id"]], left_on="smiles", right_on="prism_smiles"
    )
    prism_drugs = unique_name_lookup(catalog, "name", "drug_id")
    pools, inputs = [], []
    for source in config["sources"]:
        rows = cohort.loc[cohort["study"].eq(source["study"]) & cohort["drug_id"].isin(shared)]
        pool, report = mapped_doses(
            source,
            cells=cells,
            drugs=gdsc_drugs if source["study"] == "gdsc2" else prism_drugs,
            eligible=rows,
        )
        pools.append(pool)
        inputs.append(report)
        logger.info("%s: %s", source["study"], report)
    doses = pd.concat(pools, ignore_index=True)
    groups = list(doses.groupby(["study", "depmap_id", "drug_id"], sort=True))

    def fit_group(item):
        keys, group = item
        return dict(zip(("study", "depmap_id", "drug_id"), keys, strict=True)) | fit_shared_auc(
            group["log10_um"].to_numpy(),
            group["response"].to_numpy(),
            lower=config["log10_um_lower"],
            upper=config["log10_um_upper"],
        )

    with ThreadPoolExecutor(max_workers=config.get("workers", 4)) as pool:
        fits = pd.DataFrame(pool.map(fit_group, groups))
    fits_path = Path(config["curve_fits"])
    fits_path.parent.mkdir(parents=True, exist_ok=True)
    fits.to_csv(fits_path, index=False)
    eligible_fits = fits.loc[fits["auc_harmonized"].notna()]
    result = cohort.merge(
        eligible_fits, on=["study", "depmap_id", "drug_id"], validate="one_to_one"
    )
    result = result.rename(columns={"auc": "published_auc"})
    result["auc"] = result["auc_harmonized"]
    result["endpoint_protocol"] = "shared_window_3pl_top1_v1"
    kept = assignments.loc[assignments["pair_id"].isin(result["pair_id"])].copy()
    validate_assignments(result, kept)
    if not {"train", "validation", "test"}.issubset(set(kept["split"])):
        raise ValueError("Harmonized cohort must retain all evaluation subsets.")
    Path(config["output_cohort"]).parent.mkdir(parents=True, exist_ok=True)
    Path(config["output_assignments"]).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(config["output_cohort"], index=False)
    kept.to_csv(config["output_assignments"], index=False)
    counts = (
        result.merge(kept, on="pair_id")
        .groupby(["study", "split"])
        .agg(
            n_pairs=("pair_id", "size"),
            n_cells=("depmap_id", "nunique"),
            n_drugs=("drug_id", "nunique"),
        )
    )
    report = {
        "config": config,
        "inputs": inputs,
        "fit_status": fits.groupby(["study", "status"])
        .size()
        .rename("n")
        .reset_index()
        .to_dict("records"),
        "counts": counts.reset_index().to_dict("records"),
        "notes": [
            "Common endpoint definition; exposure duration remains study-specific/unknown "
            "in these inputs.",
            "No extrapolation outside measured dose support. Fits use all measured doses.",
            "Eligibility requires strict chemical mapping, existing cohort pairs "
            "and dose coverage.",
            "Exploratory reuse of an already examined external screen, "
            "not new confirmatory validation.",
        ],
    }
    Path(config["summary"]).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/harmonization.json")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    print(
        json.dumps(harmonize_endpoints(build_parser().parse_args().config)["counts"], indent=2)
    )


if __name__ == "__main__":
    main()
