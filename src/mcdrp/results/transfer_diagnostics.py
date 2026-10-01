"""Paired assay concordance and train-vocabulary annotation coverage diagnostics.

These describe existing data/predictions; they do not tune on external labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mcdrp.features.targets import build_target_matrix, parse_putative_targets
from mcdrp.metrics import safe_corr
from mcdrp.results.metrics_report import build_metrics_report, fit_split_reference
from mcdrp.results.predictions import load_predictions
from mcdrp.splits.validation import validate_assignments


def assay_concordance(cohort: pd.DataFrame, *, min_cells: int = 5) -> pd.DataFrame:
    """Correlate published AUC within each drug over the same cell lines.

    AUC is not harmonized to a shared dose range. Correlations are descriptive,
    and low concordance does not establish an upper bound on predictive accuracy.
    """

    if min_cells < 2:
        raise ValueError("Concordance needs at least two cells per drug.")
    keys = ["depmap_id", "drug_id"]
    frames = []
    for study in ("gdsc2", "prism"):
        frame = cohort.loc[cohort["study"].eq(study), [*keys, "auc"]].copy()
        if frame.duplicated(keys).any():
            raise ValueError(f"Duplicate cell-drug pairs in {study}; resolve replicates first.")
        if not np.isfinite(frame["auc"].to_numpy(float)).all():
            raise ValueError("Concordance requires finite AUC values.")
        frames.append(frame.rename(columns={"auc": f"auc_{study}"}))
    paired = frames[0].merge(frames[1], on=keys, validate="one_to_one")
    if paired.empty:
        raise ValueError("No shared cell-drug pairs for assay concordance.")
    rows = []
    for drug_id, group in paired.groupby("drug_id"):
        enough = len(group) >= min_cells
        gdsc = group["auc_gdsc2"].to_numpy()
        prism = group["auc_prism"].to_numpy()
        rows.append(
            {
                "drug_id": str(drug_id),
                "n_shared_cells": len(group),
                "eligible": enough,
                "pearson": safe_corr(gdsc, prism, method="pearson") if enough else np.nan,
                "spearman": safe_corr(gdsc, prism, method="spearman") if enough else np.nan,
                "gdsc_auc_mean": float(gdsc.mean()),
                "prism_auc_mean": float(prism.mean()),
                "mean_shift_prism_minus_gdsc": float((prism - gdsc).mean()),
                "rmse_between_published_auc": float(np.sqrt(np.mean((prism - gdsc) ** 2))),
            }
        )
    return pd.DataFrame(rows)


def annotation_coverage(
    cohort: pd.DataFrame, cv_dir: Path, predictions_dir: Path, *, n_folds: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired drug errors stratified by coverage of each fold's training vocabulary."""

    per_drug = []
    for fold in range(n_folds):
        split_dir = cv_dir / f"fold_{fold}"
        assignments = pd.read_csv(split_dir / "cold_drug.csv")
        validate_assignments(cohort, assignments)
        labelled = cohort.merge(assignments, on="pair_id", validate="one_to_one")
        train_drugs = set(labelled.loc[labelled["split"].eq("train"), "drug_id"].astype(str))
        pipeline, _matrix = build_target_matrix(cohort, train_drugs)
        vocabulary = set(pipeline.target_names)
        predictions = load_predictions(predictions_dir / f"fold_{fold}" / "b6_targets.csv")
        predictions = predictions.loc[
            predictions["model"].eq("targets_expression_pca")
            & predictions["subset"].eq("test")
            & predictions["variant"].eq("standard")
        ]
        # Validate coverage, labels and target identity with the common scorer.
        build_metrics_report(predictions, cohort, split_dir)
        rows = predictions.merge(cohort, on="pair_id", validate="one_to_one")
        reference = fit_split_reference(cohort, split_dir / "cold_drug.csv")
        rows["reference_pred"] = reference.predict(
            rows["depmap_id"].tolist(), rows["drug_id"].tolist()
        )
        for drug_id, group in rows.groupby("drug_id"):
            tokens = set(parse_putative_targets(group["putative_target"].iloc[0]))
            covered = tokens & vocabulary
            stratum = "covered" if covered else ("unseen_tokens" if tokens else "missing")
            model_mse = float(np.mean((group["y_true"] - group["y_pred"]) ** 2))
            reference_mse = float(np.mean((group["y_true"] - group["reference_pred"]) ** 2))
            per_drug.append(
                {
                    "fold": fold,
                    "drug_id": str(drug_id),
                    "coverage": stratum,
                    "n_rows": len(group),
                    "n_annotated_tokens": len(tokens),
                    "n_training_vocabulary_tokens": len(covered),
                    "model_mse": model_mse,
                    "reference_mse": reference_mse,
                    "model_rmse": np.sqrt(model_mse),
                    "reference_rmse": np.sqrt(reference_mse),
                    "paired_rmse_delta": np.sqrt(model_mse) - np.sqrt(reference_mse),
                }
            )
    drugs = pd.DataFrame(per_drug)
    if drugs["drug_id"].duplicated().any() or set(drugs["drug_id"]) != set(
        cohort["drug_id"].astype(str)
    ):
        raise ValueError("Coverage diagnostic must test every drug exactly once.")
    summaries = []
    for coverage, group in drugs.groupby("coverage"):
        model_rmse = np.sqrt(np.average(group["model_mse"], weights=group["n_rows"]))
        reference_rmse = np.sqrt(np.average(group["reference_mse"], weights=group["n_rows"]))
        summaries.append(
            {
                "coverage": coverage,
                "n_drugs": len(group),
                "n_rows": int(group["n_rows"].sum()),
                "row_weighted_model_rmse": model_rmse,
                "row_weighted_reference_rmse": reference_rmse,
                "row_weighted_rmse_delta": model_rmse - reference_rmse,
                "drug_mean_paired_rmse_delta": float(group["paired_rmse_delta"].mean()),
                "drugs_won": int(group["paired_rmse_delta"].lt(0).sum()),
            }
        )
    return drugs, pd.DataFrame(summaries)


def run_diagnostics(
    transfer_cohort: str = "data/processed/transfer_checked_pairs.csv",
    cohort_path: str = "data/processed/cohort_pairs.csv",
    cv_dir: str = "data/processed/splits/cv/cold_drug",
    predictions_dir: str = "results/predictions/cv_checked/cold_drug",
    output_dir: str = "results/reports/diagnostics",
    summary: str = "data/reports/transfer_diagnostics_summary.json",
) -> dict[str, Any]:
    """Execute roadmap items 7b and 8c without retraining."""

    concordance = assay_concordance(pd.read_csv(transfer_cohort))
    drugs, coverage = annotation_coverage(
        pd.read_csv(cohort_path), Path(cv_dir), Path(predictions_dir)
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    concordance.to_csv(output / "assay_concordance_per_drug.csv", index=False)
    drugs.to_csv(output / "annotation_coverage_per_drug.csv", index=False)
    coverage.to_csv(output / "annotation_coverage_summary.csv", index=False)
    eligible = concordance.loc[concordance["eligible"]]
    report = {
        "inputs": {
            "transfer_cohort": transfer_cohort,
            "cohort": cohort_path,
            "cv_dir": cv_dir,
            "predictions_dir": predictions_dir,
        },
        "assay_concordance": {
            "n_drugs": len(concordance),
            "n_shared_pairs": int(concordance["n_shared_cells"].sum()),
            "n_eligible_drugs": len(eligible),
            "n_finite_spearman": int(eligible["spearman"].notna().sum()),
            "median_spearman": float(eligible["spearman"].median()),
            "mean_spearman": float(eligible["spearman"].mean()),
            "median_pearson": float(eligible["pearson"].median()),
            "drugs_spearman_above_0_3": int(eligible["spearman"].gt(0.3).sum()),
        },
        "annotation_coverage": coverage.to_dict("records"),
        "notes": [
            "Published AUC values are not harmonized to a common dose range.",
            "Coverage uses the vocabulary of training drugs separately in each fold.",
            "Pooled squared errors use each prediction's own fold reference.",
            "Descriptive diagnostics, no hypothesis tests or causal conclusions.",
        ],
    }
    Path(summary).parent.mkdir(parents=True, exist_ok=True)
    Path(summary).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transfer-cohort", default="data/processed/transfer_checked_pairs.csv"
    )
    parser.add_argument("--cohort-path", default="data/processed/cohort_pairs.csv")
    parser.add_argument("--cv-dir", default="data/processed/splits/cv/cold_drug")
    parser.add_argument("--predictions-dir", default="results/predictions/cv_checked/cold_drug")
    parser.add_argument("--output-dir", default="results/reports/diagnostics")
    parser.add_argument("--summary", default="data/reports/transfer_diagnostics_summary.json")
    report = run_diagnostics(**vars(parser.parse_args()))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
