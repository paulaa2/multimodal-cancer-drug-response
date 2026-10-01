"""Collect completed roadmap experiments, matched-endpoint diagnostics and sensitivities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mcdrp.data.target_annotations import file_sha256
from mcdrp.features.fingerprints import build_fingerprint_matrix
from mcdrp.features.targets import build_target_matrix
from mcdrp.metrics import regression_metrics
from mcdrp.results.metrics_report import (
    GROUP_COLUMNS,
    build_metrics_report,
    fit_split_reference,
)
from mcdrp.results.predictions import load_predictions
from mcdrp.results.transfer_diagnostics import assay_concordance
from mcdrp.results.uncertainty import interval_diagnostics, paired_bootstrap


def matching_sensitivity() -> dict:
    cohort = pd.read_csv("data/processed/transfer_checked_pairs.csv", low_memory=False)
    results = []
    for seed in (7, 13, 42):
        root = Path(f"results/transfer_strict/seed_{seed}")
        predictions = load_predictions(sorted((root / "predictions").glob("*.csv")))
        split_dir = "data/processed/transfer_checked"
        build_metrics_report(predictions, cohort, split_dir, target="auc")
        reference = fit_split_reference(
            cohort, Path(split_dir) / "transfer_prism.csv", target="auc"
        )
        data = predictions.loc[predictions["subset"].eq("test")].merge(
            cohort, on="pair_id", validate="many_to_one"
        )
        for keys, frame in data.groupby(list(GROUP_COLUMNS)):
            for matching in ("all_strict", "full_inchikey_only"):
                selected = (
                    frame
                    if matching == "all_strict"
                    else frame.loc[frame["match_type"].eq("full_inchikey")]
                )
                if selected.empty:
                    raise ValueError(
                        "No full-InChIKey pairs; check matching label definitions."
                    )
                pred_ref = reference.predict(
                    selected["depmap_id"].tolist(), selected["drug_id"].tolist()
                )
                results.append(
                    dict(zip(GROUP_COLUMNS, keys, strict=True))
                    | {
                        "seed": seed,
                        "matching": matching,
                        "n_rows": len(selected),
                        "n_drugs": selected["drug_id"].nunique(),
                        **regression_metrics(
                            selected["y_true"].to_numpy(),
                            selected["y_pred"].to_numpy(),
                            reference_pred=pred_ref,
                            cell_keys=selected["depmap_id"].tolist(),
                            drug_keys=selected["drug_id"].tolist(),
                        ),
                    }
                )
    table = pd.DataFrame(results)
    aggregate = table.groupby(["stage", "model", "variant", "matching"], as_index=False).agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        r2_normalized_mean=("r2_normalized", "mean"),
        n_drugs=("n_drugs", "first"),
        n_rows=("n_rows", "first"),
    )
    return {
        "per_seed": json.loads(table.to_json(orient="records")),
        "summary": json.loads(aggregate.to_json(orient="records")),
        "scope": "Held-out subset sensitivity of frozen predictions; "
        "no retraining or test-based selection.",
    }


def endpoint_concordance() -> dict:
    cohort = pd.read_csv("data/processed/harmonized_transfer_pairs.csv", low_memory=False)
    matched = []
    for endpoint in ("published", "harmonized"):
        data = cohort.copy()
        if endpoint == "published":
            data["auc"] = data["published_auc"]
        table = assay_concordance(data)
        table["endpoint"] = endpoint
        matched.append(table)
    table = pd.concat(matched, ignore_index=True)
    summary = (
        table.loc[table["eligible"]]
        .groupby("endpoint", as_index=False)
        .agg(
            n_drugs=("drug_id", "nunique"),
            n_matched_pairs=("n_shared_cells", "sum"),
            pearson_mean=("pearson", "mean"),
            spearman_mean=("spearman", "mean"),
            mean_shift=("mean_shift_prism_minus_gdsc", "mean"),
            mean_drug_auc_discrepancy=("rmse_between_published_auc", "mean"),
        )
    )
    return {
        "paired_drugs": json.loads(table.to_json(orient="records")),
        "summary": json.loads(summary.to_json(orient="records")),
        "scope": "Both endpoints evaluated on identical retained cell-drug intersections; "
        "drug means unweighted.",
    }


def transfer_endpoints(root: Path) -> list[dict]:
    cohort = pd.read_csv("data/processed/harmonized_transfer_pairs.csv", low_memory=False)
    assignments = pd.read_csv("data/processed/harmonized_transfer/transfer_prism.csv")
    labelled = cohort.merge(assignments, on="pair_id", validate="one_to_one")
    seen = set(labelled.loc[labelled["split"].eq("train"), "drug_id"])
    results = []
    for endpoint in ("published_matched", "harmonized"):
        case = (
            root
            / "harmonized"
            / ("published_matched" if endpoint == "published_matched" else "standard")
        )
        source = cohort.copy()
        if endpoint == "published_matched":
            source["auc"] = source["published_auc"]
        b0_path = (
            case / "b0.csv"
            if endpoint == "published_matched"
            else root / "harmonized" / "b0.csv"
        )
        predictions = load_predictions([case / "predictions.csv", b0_path])
        build_metrics_report(predictions, source, case / "splits", target="auc")
        reference = fit_split_reference(
            source, case / "splits" / "transfer_prism.csv", target="auc"
        )
        test = predictions.loc[predictions["subset"].eq("test")].merge(
            source[["pair_id", "drug_id", "depmap_id"]], on="pair_id", validate="many_to_one"
        )
        for keys, frame in test.groupby(list(GROUP_COLUMNS)):
            for stratum in ("all", "seen_drugs", "unseen_drugs"):
                selected = frame
                if stratum != "all":
                    mask = frame["drug_id"].isin(seen)
                    selected = frame.loc[mask if stratum == "seen_drugs" else ~mask]
                if selected.empty:
                    continue
                pred_ref = reference.predict(
                    selected["depmap_id"].tolist(), selected["drug_id"].tolist()
                )
                results.append(
                    dict(zip(GROUP_COLUMNS, keys, strict=True))
                    | {
                        "endpoint": endpoint,
                        "stratum": stratum,
                        "n_rows": len(selected),
                        "n_drugs": int(selected["drug_id"].nunique()),
                        **regression_metrics(
                            selected["y_true"].to_numpy(),
                            selected["y_pred"].to_numpy(),
                            reference_pred=pred_ref,
                            cell_keys=selected["depmap_id"].tolist(),
                            drug_keys=selected["drug_id"].tolist(),
                        ),
                    }
                )
    # Pandas converts all optional NaN correlation values to strict JSON null.
    return json.loads(pd.DataFrame(results).to_json(orient="records"))


def write_interval_strata(
    predictions: pd.DataFrame,
    cohort: pd.DataFrame,
    report: list[dict],
    output: Path,
    *,
    bounds=None,
) -> None:
    columns = [
        column for column in ("drug_id", "depmap_id", "oncotree_lineage") if column in cohort
    ]
    data = predictions.loc[predictions["subset"].eq("test")].merge(
        cohort[["pair_id", *columns]], on="pair_id", validate="many_to_one"
    )
    rows = []
    for record in report:
        frame = data.loc[
            data["model"].eq(record["model"]) & data["variant"].eq(record["variant"])
        ]
        for column in columns:
            for key, group in frame.groupby(column, dropna=False):
                radius = record["radius"] if record["radius"] is not None else float("inf")
                pred = (
                    record["fitted_intercept"]
                    + record["fitted_slope"] * group["y_pred"].to_numpy()
                )
                diagnostics = interval_diagnostics(
                    group["y_true"].to_numpy(),
                    pred,
                    group[column].fillna("unknown").to_numpy(),
                    radius,
                    bounds,
                )
                rows.append(
                    {
                        "model": record["model"],
                        "method": record["method"],
                        "group_column": column,
                        "entity": key,
                        **diagnostics,
                    }
                )
    pd.DataFrame(rows).to_csv(output, index=False)


def donor_diagnostics(root: Path, cohort: pd.DataFrame, folds: int) -> list[dict]:
    catalog = cohort.drop_duplicates("drug_id").copy()
    catalog["drug_id"] = catalog["drug_id"].astype(str)
    fp = build_fingerprint_matrix(catalog["canonical_smiles"])
    indices = {key: idx for idx, key in enumerate(catalog["drug_id"])}
    results = []
    for fold in range(folds):
        assignments = pd.read_csv(
            f"data/processed/splits/cv/cold_drug/fold_{fold}/cold_drug.csv"
        )
        train = cohort.merge(assignments, on="pair_id").query("split == 'train'")
        _, targets = build_target_matrix(catalog, set(train["drug_id"].astype(str)))
        for block in ("cell", "drug"):
            for seed in (7, 13, 42):
                path = (
                    root
                    / "ablation"
                    / f"fold_{fold}"
                    / f"{block}_permuted_{seed}"
                    / "donors.csv"
                )
                donors = pd.read_csv(path, dtype={"entity_id": str, "donor_id": str})
                signatures = donors.set_index("entity_id")["partition_signature"]
                if (
                    set(donors["entity_id"]) != set(donors["donor_id"])
                    or donors["donor_id"].duplicated().any()
                ):
                    raise ValueError("Donor mapping is not an entity permutation.")
                if (
                    not donors["donor_id"]
                    .map(signatures)
                    .eq(donors["partition_signature"])
                    .all()
                ):
                    raise ValueError("Donor partition signatures differ.")
                row = {
                    "fold": fold,
                    "block": block,
                    "seed": seed,
                    "n_entities": len(donors),
                    "n_identity_fixed_points": int(
                        donors["entity_id"].eq(donors["donor_id"]).sum()
                    ),
                }
                if block == "drug":
                    own = [indices[key] for key in donors["entity_id"]]
                    other = [indices[key] for key in donors["donor_id"]]
                    row["n_unchanged_target_vectors"] = int(
                        (targets[own] == targets[other]).all(axis=1).sum()
                    )
                    row["n_unchanged_morgan_vectors"] = int(
                        (fp[own] == fp[other]).all(axis=1).sum()
                    )
                results.append(row)
    return results


def collect(root: Path, *, folds: int = 5) -> dict:
    cohort = pd.read_csv("data/processed/cohort_pairs.csv", low_memory=False)
    tables: list[pd.DataFrame] = []
    comparisons: list[dict] = []
    uncertainty: list[dict] = []
    missing = []
    annotation_path = "data/mappings/target_annotations_normalized.csv"
    annotation_hash = file_sha256(Path(annotation_path))
    refreshed_views = root / "curation_corrected"
    for fold in range(folds):
        split_dir = Path(f"data/processed/splits/cv/cold_drug/fold_{fold}")
        reference = fit_split_reference(cohort, split_dir / "cold_drug.csv")
        standard_path = root / "ablation" / f"fold_{fold}" / "standard" / "predictions.csv"
        standard = load_predictions(standard_path) if standard_path.exists() else None
        cases = [
            root / "random_forest" / f"fold_{fold}",
            root / "graph_only" / f"fold_{fold}",
            root / "b7_pinned" / f"fold_{fold}",
        ]
        expected = ["standard", "calibration"]
        expected.extend(
            f"{block}_permuted_{seed}" for block in ("cell", "drug") for seed in (7, 13, 42)
        )
        cases.extend(root / "ablation" / f"fold_{fold}" / case for case in expected)
        for view in ("genes", "mechanisms", "typed"):
            refreshed = refreshed_views / f"fold_{fold}" / f"targets_{view}"
            selected = (
                refreshed
                if refreshed_views.exists()
                else root / "ablation" / f"fold_{fold}" / f"targets_{view}"
            )
            if (selected / "complete.json").exists():
                complete = json.loads((selected / "complete.json").read_text())
                declared = complete.get("annotation_sha256")
                if declared is None:
                    declared = json.loads((root / "ablations_inputs.json").read_text())[
                        "inputs"
                    ][annotation_path]
                if declared != annotation_hash:
                    raise ValueError(
                        "Target views need a refresh after the annotation file changed."
                    )
            cases.append(selected)
        for case in cases:
            path = case / "predictions.csv"
            if not path.exists():
                missing.append(str(path))
                continue
            predictions = load_predictions(path)
            local_splits = case / "splits" if (case / "splits").exists() else split_dir
            metrics = build_metrics_report(predictions, cohort, local_splits)
            metrics["fold"] = fold
            metrics["case"] = case.name
            metrics["prediction_artifact"] = str(path)
            tables.append(metrics)
            test = predictions.loc[predictions["subset"].eq("test")].merge(
                cohort[["pair_id", "depmap_id", "drug_id"]],
                on="pair_id",
                validate="many_to_one",
            )
            if case.name != "calibration":
                for keys, frame in test.groupby(list(GROUP_COLUMNS)):
                    pred_ref = reference.predict(
                        frame["depmap_id"].tolist(), frame["drug_id"].tolist()
                    )
                    comparison = dict(zip(GROUP_COLUMNS, keys, strict=True)) | {
                        "fold": fold,
                        "case": case.name,
                        "prediction_artifact": str(path),
                        **paired_bootstrap(
                            frame["y_true"].to_numpy(),
                            frame["y_pred"].to_numpy(),
                            pred_ref,
                            frame["drug_id"].to_numpy(),
                            repeats=2000,
                        ),
                    }
                    if standard is not None and str(keys[0]) == "B6":
                        standard_model = standard.loc[
                            standard["model"].eq(keys[1]) & standard["subset"].eq("test")
                        ].set_index("pair_id")
                        aligned_pred = frame["pair_id"].map(standard_model["y_pred"]).to_numpy()
                        comparison["vs_standard"] = paired_bootstrap(
                            frame["y_true"].to_numpy(),
                            frame["y_pred"].to_numpy(),
                            aligned_pred,
                            frame["drug_id"].to_numpy(),
                            repeats=2000,
                        )
                    comparisons.append(comparison)
            if (case / "uncertainty.json").exists():
                calibration_report = json.loads((case / "uncertainty.json").read_text())
                write_interval_strata(
                    predictions, cohort, calibration_report, case / "interval_strata.csv"
                )
                uncertainty.extend({"fold": fold, **row} for row in calibration_report)
    if missing:
        raise ValueError("Experiments incomplete: " + ", ".join(missing))
    table = pd.concat(tables, ignore_index=True)
    output_dir = root / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "fold_metrics.csv", index=False)
    test = table.loc[table["subset"].eq("test")]
    aggregate = test.groupby(["stage", "model", "variant"], as_index=False).agg(
        n_folds=("fold", "nunique"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        r2_normalized_mean=("r2_normalized", "mean"),
        spearman_per_drug_mean=("spearman_per_drug", "mean"),
    )
    return {
        "n_folds": folds,
        "annotation_sha256": annotation_hash,
        "annotation_view_source": str(refreshed_views)
        if refreshed_views.exists()
        else str(root / "ablation"),
        "aggregate_test": json.loads(aggregate.to_json(orient="records")),
        "donor_diagnostics": donor_diagnostics(root, cohort, folds),
        "paired_drug_bootstrap": comparisons,
        "independent_calibration": uncertainty,
        "endpoint_concordance": endpoint_concordance(),
        "matched_endpoint_transfer": transfer_endpoints(root),
        "transfer_independent_calibration": json.loads(
            (root / "harmonized" / "calibration" / "uncertainty.json").read_text()
        ),
        "chemical_matching_sensitivity": matching_sensitivity(),
        "interpretation": "Exploratory diagnostics. Overlapping folds are not independent "
        "studies; bootstrap conditions on fitted models. Assay shift does not meet "
        "ordinary exchangeability guarantees.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/roadmap_2_6")
    parser.add_argument("--output", default="data/reports/roadmap_2_6_summary.json")
    parser.add_argument("--sensitivities-only", action="store_true")
    args = parser.parse_args()
    report = (
        {
            "endpoint_concordance": endpoint_concordance(),
            "chemical_matching_sensitivity": matching_sensitivity(),
        }
        if args.sensitivities_only
        else collect(Path(args.root))
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
