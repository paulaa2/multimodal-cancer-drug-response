"""Random Forest baseline selected on each partition's declared validation rows."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from mcdrp.features.expression import build_cell_features
from mcdrp.features.fingerprints import build_fingerprint_matrix
from mcdrp.features.targets import build_target_matrix
from mcdrp.metrics import regression_metrics
from mcdrp.models.ablation_b6 import assemble_ablation_features
from mcdrp.results.predictions import TUNED_VARIANT, prediction_frame, write_predictions
from mcdrp.splits.make_splits import DEFAULT_SPLITS, KNOWN_SPLIT_CHOICES
from mcdrp.splits.validation import validate_assignments

logger = logging.getLogger(__name__)


def fit_forest_candidates(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    n_estimators: int = 200,
    n_jobs: int = 8,
    random_state: int = 42,
) -> tuple[RandomForestRegressor, list[dict[str, Any]]]:
    """A fixed four-point grid; test responses are absent from the selection API."""

    rows = []
    best = None
    best_rmse = float("inf")
    for features in ("sqrt", 0.1):
        for leaf in (5, 20):
            params = {
                "n_estimators": n_estimators,
                "max_features": features,
                "min_samples_leaf": leaf,
                "random_state": random_state,
                "n_jobs": n_jobs,
                "bootstrap": True,
            }
            model = RandomForestRegressor(**params)
            model.fit(x_train, y_train)
            rmse = float(np.sqrt(mean_squared_error(y_validation, model.predict(x_validation))))
            rows.append({"params": json.dumps(params, sort_keys=True), "validation_rmse": rmse})
            logger.info("Forest candidate %s: validation RMSE %.5f", params, rmse)
            if rmse < best_rmse:
                best, best_rmse = model, rmse
    assert best is not None
    return best, rows


def run_random_forest(
    cohort_path: str = "data/processed/cohort_pairs.csv",
    split_dir: str = "data/processed/splits",
    expression_path: str = (
        "data/raw/OmicsExpressionTPMLogp1HumanProteinCodingGenesStranded.csv"
    ),
    output: str = "results/random_forest/metrics.csv",
    predictions_output: str = "results/random_forest/predictions.csv",
    summary: str = "data/reports/random_forest_summary.json",
    *,
    split_names: tuple[str, ...] = DEFAULT_SPLITS,
    models: tuple[str, ...] = ("morgan_expression_pca", "targets_expression_pca"),
    target: str = "ln_ic50",
    n_components: int = 256,
    n_estimators: int = 200,
    n_jobs: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    cohort = pd.read_csv(cohort_path, low_memory=False)
    fingerprints = build_fingerprint_matrix(cohort["canonical_smiles"])
    zeros = np.zeros((len(cohort), 0), dtype=np.float32)
    frames: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for split_name in split_names:
        assignments = pd.read_csv(Path(split_dir) / f"{split_name}.csv")
        validate_assignments(cohort, assignments)
        data = cohort.merge(assignments, on="pair_id", validate="one_to_one")
        subsets = {
            name: data.loc[data["split"].eq(name)].copy()
            for name in ("train", "validation", "calibration", "test")
            if data["split"].eq(name).any()
        }
        train = subsets["train"]
        _, pca_map = build_cell_features(
            expression_path,
            cohort,
            set(train["depmap_id"]),
            n_components=n_components,
            random_state=random_state,
        )
        _, targets = build_target_matrix(cohort, set(train["drug_id"].astype(str)))
        for name in models:
            logger.info("Forest %s / %s", split_name, name)
            x = {
                subset: assemble_ablation_features(
                    rows,
                    ablation=name,
                    cohort=cohort,
                    fp_matrix=fingerprints,
                    pca_feature_map=pca_map,
                    pathway_feature_map={},
                    target_matrix=targets,
                    predicted_target_matrix=zeros,
                    product_matrix=zeros,
                )
                for subset, rows in subsets.items()
            }
            # Remove only features constant on training data, never on held-out rows.
            keep = np.ptp(x["train"], axis=0) > 0
            x = {key: matrix[:, keep] for key, matrix in x.items()}
            model, grid = fit_forest_candidates(
                x["train"],
                train[target].to_numpy(),
                x["validation"],
                subsets["validation"][target].to_numpy(),
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                random_state=random_state,
            )
            candidates.extend({"split_name": split_name, "model": name, **row} for row in grid)
            for subset in (
                name for name in ("validation", "calibration", "test") if name in subsets
            ):
                rows = subsets[subset]
                predictions = model.predict(x[subset])
                frames.append(
                    prediction_frame(
                        rows,
                        rows[target].to_numpy(),
                        predictions,
                        stage="RF",
                        model=name,
                        split_name=split_name,
                        subset=subset,
                        variant=TUNED_VARIANT,
                    )
                )
                metrics.append(
                    {
                        "stage": "RF",
                        "model": name,
                        "variant": TUNED_VARIANT,
                        "split_name": split_name,
                        "subset": subset,
                        "n_rows": len(rows),
                        "n_features": int(keep.sum()),
                        "params": json.dumps(model.get_params(), sort_keys=True),
                        **regression_metrics(
                            rows[target].to_numpy(),
                            predictions,
                            cell_keys=rows["depmap_id"].tolist(),
                            drug_keys=rows["drug_id"].tolist(),
                        ),
                    }
                )
    table = pd.DataFrame(metrics)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    pd.DataFrame(candidates).to_csv(
        Path(output).with_name("tuning_candidates.csv"), index=False
    )
    write_predictions(frames, predictions_output)
    report = {
        "cohort": cohort_path,
        "split_dir": split_dir,
        "target": target,
        "selection": "minimum declared-validation RMSE; test evaluated only for winner",
        "n_components": n_components,
        "random_state": random_state,
        "n_estimators": n_estimators,
        "candidates": candidates,
        "metrics": json.loads(table.to_json(orient="records")),
    }
    Path(summary).parent.mkdir(parents=True, exist_ok=True)
    Path(summary).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort", dest="cohort_path", default="data/processed/cohort_pairs.csv"
    )
    parser.add_argument("--split-dir", default="data/processed/splits")
    parser.add_argument(
        "--expression",
        dest="expression_path",
        default="data/raw/OmicsExpressionTPMLogp1HumanProteinCodingGenesStranded.csv",
    )
    parser.add_argument("--output", default="results/random_forest/metrics.csv")
    parser.add_argument("--predictions-output", default="results/random_forest/predictions.csv")
    parser.add_argument("--summary", default="data/reports/random_forest_summary.json")
    parser.add_argument(
        "--splits",
        nargs="+",
        dest="split_names",
        default=list(DEFAULT_SPLITS),
        choices=list(KNOWN_SPLIT_CHOICES),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["morgan_expression_pca", "targets_expression_pca"],
        choices=["morgan_expression_pca", "targets_expression_pca"],
    )
    parser.add_argument("--target", default="ln_ic50")
    parser.add_argument("--n-components", type=int, default=256)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = vars(build_parser().parse_args())
    args["split_names"] = tuple(args["split_names"])
    args["models"] = tuple(args["models"])
    print(run_random_forest(**args)[["model", "subset", "rmse"]].to_string(index=False))


if __name__ == "__main__":
    main()
