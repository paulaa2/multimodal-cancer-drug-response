"""Frozen B0/RF/B6/B7 fits, persisted models and verified prediction replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from mcdrp.metrics import MeanEffectsReference
from mcdrp.models.baseline_b2 import fit_xgboost, predict_xgboost
from mcdrp.models.random_forest import fit_forest_candidates
from mcdrp.results.metrics_report import build_metrics_report
from mcdrp.results.predictions import prediction_frame, write_predictions
from mcdrp.splits.validation import validate_assignments

SPECIFICATIONS = (
    ("B0", "mean_effects", None),
    ("RF", "targets_expression_pca", "targets"),
    ("RF", "morgan_expression_pca", "morgan"),
    ("B6", "targets_expression_pca", "targets"),
    ("B6", "morgan_expression_pca", "morgan"),
    ("B7", "pretrained_drug_expression_pca", "embeddings"),
)


def assemble(rows: pd.DataFrame, features: dict, block: str) -> np.ndarray:
    """Apply saved, training-fitted features without fitting on evaluation rows."""
    drug = np.vstack([features[block][str(k)] for k in rows["drug_id"]])
    cell = np.vstack([features["cells"][str(k)] for k in rows["depmap_id"]])
    return np.hstack([drug, cell]).astype(np.float32)


def replay(artifact: dict, rows: pd.DataFrame, features: dict) -> np.ndarray:
    model = artifact["model"]
    if artifact["stage"] == "B0":
        return model.predict(rows["depmap_id"].tolist(), rows["drug_id"].tolist())
    matrix = assemble(rows, features, artifact["block"])
    if artifact["keep"] is not None:
        matrix = matrix[:, artifact["keep"]]
    if artifact["stage"] == "RF":
        return model.predict(matrix)
    return predict_xgboost(model, matrix)


def train_case(config_path: str, families: tuple[str, ...]) -> None:
    config = json.loads(Path(config_path).read_text())
    cohort = pd.read_csv(config["cohort"], low_memory=False)
    assignments = pd.read_csv(config["assignments"])
    validate_assignments(cohort, assignments)
    rows = cohort.merge(assignments, on="pair_id", validate="one_to_one")
    if config["split_name"] == "cold_drug":
        columns = ["drug_id"]
        if config.get("chemical_group_column"):
            columns.append(config["chemical_group_column"])
        for column in columns:
            if rows[column].isna().any():
                raise ValueError(f"Cold-drug grouping values cannot be null: {column}")
            groups = {
                label: set(frame[column].astype(str))
                for label, frame in rows.groupby("split")
                if label != "unused"
            }
            labels = list(groups)
            for i, left in enumerate(labels):
                for right in labels[i + 1 :]:
                    if groups[left] & groups[right]:
                        raise ValueError(f"Cold-drug evaluation entities overlap: {column}")
    elif config["split_name"] == "transfer_prism":
        if (
            rows.loc[rows["split"].isin(("train", "validation", "calibration")), "study"]
            .ne("gdsc2")
            .any()
        ):
            raise ValueError("PRISM leaked into training, validation or calibration.")
        if rows.loc[rows["split"].eq("test"), "study"].ne("prism").any():
            raise ValueError("GDSC leaked into PRISM test.")
    subsets = {
        label: rows.loc[rows["split"].eq(label)]
        for label in ("train", "validation", "calibration", "test")
        if rows["split"].eq(label).any()
    }
    if not {"train", "validation", "test"}.issubset(subsets):
        raise ValueError("Stable fitting requires nonempty train, validation and test.")
    features = joblib.load(config["features"])
    params = config["xgb_params"]
    target = config["target"]
    for stage, name, block in SPECIFICATIONS:
        if stage not in families:
            continue
        output = Path(config["output"]) / f"{stage}_{name}"
        if output.exists():
            raise ValueError(f"Training never overwrites an existing model: {output}")
        output.mkdir(parents=True)
        print(f"Fitting {config['name']} / {stage} / {name}", flush=True)
        candidates: list[dict[str, Any]] = []
        keep = None
        model: Any
        if stage == "B0":
            model = MeanEffectsReference.fit(
                subsets["train"]["depmap_id"].tolist(),
                subsets["train"]["drug_id"].tolist(),
                subsets["train"][target].to_numpy(),
            )
        else:
            x_train = assemble(subsets["train"], features, str(block))
            x_val = assemble(subsets["validation"], features, str(block))
            if stage == "RF":
                keep = np.ptp(x_train, axis=0) > 0
                model, candidates = fit_forest_candidates(
                    x_train[:, keep],
                    subsets["train"][target].to_numpy(),
                    x_val[:, keep],
                    subsets["validation"][target].to_numpy(),
                    n_estimators=config["rf_trees"],
                    n_jobs=params["n_jobs"],
                    random_state=params["random_state"],
                )
            else:
                model, actual_device = fit_xgboost(
                    params,
                    x_train,
                    subsets["train"][target].to_numpy(),
                    x_val,
                    subsets["validation"][target].to_numpy(),
                    device=params["device"],
                )
                if actual_device != params["device"]:
                    raise ValueError("A stable run cannot silently change execution backend.")
                backend = json.loads(model.get_booster().save_config())["learner"][
                    "generic_param"
                ]["device"]
                if not backend.startswith(params["device"]):
                    raise ValueError(f"Requested backend was not used: {backend}")
            del x_train, x_val
        artifact = {"stage": stage, "model": model, "block": block, "keep": keep}
        variant = "tuned" if stage == "RF" else "standard"
        frames = []
        original = {}
        for subset, data in subsets.items():
            if subset == "train":
                continue
            prediction = replay(artifact, data, features)
            original[subset] = prediction
            frames.append(
                prediction_frame(
                    data,
                    data[target].to_numpy(),
                    prediction,
                    stage=stage,
                    model=name,
                    split_name=config["split_name"],
                    subset=subset,
                    variant=variant,
                )
            )
        model_path = output / "model.joblib"
        joblib.dump(artifact, model_path, compress=3)
        restored = joblib.load(model_path)
        max_delta = 0.0
        for subset, expected in original.items():
            actual = replay(restored, subsets[subset], features)
            np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)
            max_delta = max(max_delta, float(np.max(np.abs(actual - expected))))
        predictions = write_predictions(frames, output / "predictions.csv")
        metrics = build_metrics_report(
            predictions, cohort, Path(config["assignments"]).parent, target=target
        )
        metrics.to_csv(output / "metrics.csv", index=False)
        (output / "fit.json").write_text(
            json.dumps(
                {
                    "stage": stage,
                    "model": name,
                    "variant": variant,
                    "selection": "RF validation grid; XGB validation early stopping",
                    "train_pairs": len(subsets["train"]),
                    "target": target,
                    "best_iteration": getattr(model, "best_iteration", None),
                    "candidates": candidates,
                    "replay_max_abs_delta": max_delta,
                    "params": json.loads(pd.Series(model.get_params()).to_json())
                    if stage != "B0"
                    else None,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        print(f"Saved and replayed {output}; max delta {max_delta:.3g}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--families", nargs="+", choices=["B0", "RF", "B6", "B7"], required=True
    )
    args = parser.parse_args()
    train_case(args.case, tuple(args.families))


if __name__ == "__main__":
    main()
