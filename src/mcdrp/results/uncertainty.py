"""Independent grouped conformal bands and paired entity-bootstrap comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mcdrp.results.metrics_report import GROUP_COLUMNS, build_metrics_report
from mcdrp.results.predictions import load_predictions


def conformal_radius(residuals: np.ndarray, groups: np.ndarray, alpha: float) -> float:
    """Finite-sample order statistic of maximum absolute residual per entity."""
    if not 0 < alpha < 1 or len(residuals) != len(groups) or not len(residuals):
        raise ValueError("Nonempty aligned residuals and alpha in (0,1) are required.")
    if not np.isfinite(residuals).all() or pd.isna(groups).any():
        raise ValueError("Calibration residuals/groups must be finite and non-null.")
    scores = (
        pd.DataFrame({"group": groups, "score": np.abs(residuals)})
        .groupby("group")["score"]
        .max()
    )
    rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return float("inf") if rank > len(scores) else float(np.sort(scores)[rank - 1])


def affine_parameters(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    variance = float(np.var(pred))
    slope = (
        max(0.0, float(np.mean((y - y.mean()) * (pred - pred.mean()))) / variance)
        if variance > 1e-12
        else 0.0
    )
    return float(y.mean() - slope * pred.mean()), slope


def interval_diagnostics(
    y: np.ndarray,
    pred: np.ndarray,
    groups: np.ndarray,
    radius: float,
    bounds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    lower, upper = pred - radius, pred + radius
    if bounds is not None:
        # Clip endpoints, not just the centre, including out-of-domain predictions.
        lower, upper = np.clip(lower, *bounds), np.clip(upper, *bounds)
    covered = (y >= lower) & (y <= upper)
    group_coverage = (
        pd.DataFrame({"group": groups, "covered": covered}).groupby("group")["covered"].all()
    )
    intercept, slope = affine_parameters(y, pred)
    width = float(np.mean(upper - lower))
    return {
        "n_rows": len(y),
        "n_groups": len(group_coverage),
        "row_coverage": float(covered.mean()),
        "whole_group_coverage": float(group_coverage.mean()),
        "mean_width": width if np.isfinite(width) else None,
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "bias_pred_minus_true": float(np.mean(pred - y)),
        "diagnostic_intercept": intercept,
        "diagnostic_slope": slope,
    }


def paired_bootstrap(
    y: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int = 42,
    repeats: int = 2000,
) -> dict[str, float]:
    """Resample entities jointly for both models; row-weighted RMSE difference."""
    data = pd.DataFrame(
        {"group": groups, "candidate": (y - candidate) ** 2, "reference": (y - reference) ** 2}
    )
    sums = data.groupby("group").agg(
        candidate=("candidate", "sum"),
        reference=("reference", "sum"),
        count=("candidate", "size"),
    )
    if (
        len(sums) < 2
        or repeats < 1
        or not np.isfinite(data[["candidate", "reference"]]).all().all()
    ):
        raise ValueError(
            "Bootstrap needs finite predictions, at least two groups and positive repeats."
        )
    values = sums.to_numpy()
    indices = np.random.default_rng(seed).integers(0, len(values), (repeats, len(values)))
    totals = values[indices].sum(axis=1)
    delta = np.sqrt(totals[:, 0] / totals[:, 2]) - np.sqrt(totals[:, 1] / totals[:, 2])
    return {
        "rmse_delta": float(
            np.sqrt(data["candidate"].mean()) - np.sqrt(data["reference"].mean())
        ),
        "ci95_low": float(np.quantile(delta, 0.025)),
        "ci95_high": float(np.quantile(delta, 0.975)),
    }


def uncertainty_report(
    predictions: pd.DataFrame,
    cohort: pd.DataFrame,
    split_dir: str,
    *,
    target: str,
    group: str,
    alpha: float = 0.1,
    bounds: tuple[float, float] | None = None,
) -> pd.DataFrame:
    build_metrics_report(predictions, cohort, split_dir, target=target)
    results = []
    identity = cohort[["pair_id", group]]
    for keys, frame in predictions.groupby(list(GROUP_COLUMNS[:-1])):
        data = frame.merge(identity, on="pair_id", validate="many_to_one")
        subsets = {label: rows for label, rows in data.groupby("subset")}
        if not {"validation", "calibration", "test"}.issubset(subsets):
            raise ValueError(
                "Calibration predictions must come from a reserved, independent subset."
            )
        labels = pd.read_csv(Path(split_dir) / f"{keys[-1]}.csv").merge(identity, on="pair_id")
        train_groups = set(labels.loc[labels["split"].eq("train"), group])
        calibration_groups = set(subsets["calibration"][group])
        if train_groups & calibration_groups:
            raise ValueError("Training and calibration entities overlap.")
        if set(subsets["validation"][group]) & calibration_groups:
            raise ValueError("Validation and calibration entities overlap.")
        val, cal, test = (subsets[label] for label in ("validation", "calibration", "test"))
        for method in ("raw", "affine_validation"):
            intercept, slope = (
                affine_parameters(val["y_true"].to_numpy(), val["y_pred"].to_numpy())
                if method != "raw"
                else (0.0, 1.0)
            )
            radius = conformal_radius(
                cal["y_true"].to_numpy() - (intercept + slope * cal["y_pred"].to_numpy()),
                cal[group].to_numpy(),
                alpha,
            )
            results.append(
                dict(zip(GROUP_COLUMNS[:-1], keys, strict=True))
                | {
                    "method": method,
                    "alpha": alpha,
                    "group_column": group,
                    "n_calibration_groups": len(calibration_groups),
                    "radius": radius if np.isfinite(radius) else None,
                    "status": "finite"
                    if np.isfinite(radius)
                    else "insufficient_calibration_groups",
                    "fitted_intercept": intercept,
                    "fitted_slope": slope,
                    **interval_diagnostics(
                        test["y_true"].to_numpy(),
                        intercept + slope * test["y_pred"].to_numpy(),
                        test[group].to_numpy(),
                        radius,
                        bounds,
                    ),
                }
            )
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--target", default="ln_ic50")
    parser.add_argument("--group", default="drug_id")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--bounds", nargs=2, type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = uncertainty_report(
        load_predictions(args.predictions),
        pd.read_csv(args.cohort),
        args.split_dir,
        target=args.target,
        group=args.group,
        alpha=args.alpha,
        bounds=tuple(args.bounds) if args.bounds else None,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(json.loads(report.to_json(orient="records")), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
