"""Freeze compact plotting inputs from validated final experiment artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sources: dict[str, str] = {}

    def record_hash(relative: str) -> None:
        path = root / relative
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        sources[relative] = digest.hexdigest()

    def read_csv(relative: str) -> pd.DataFrame:
        record_hash(relative)
        path = root / relative
        return pd.read_csv(path)

    for name in (
        "roadmap_2_6_summary.json", "roadmap_2_6_validation.json",
        "harmonization_summary.json", "target_annotations_summary.json",
        "drug_embeddings_pinned_audit.json", "dreval_crosscheck_summary.json",
    ):
        record_hash(f"data/reports/{name}")

    folds = read_csv("results/roadmap_2_6/reports/fold_metrics.csv")
    baseline = read_csv("results/reports/cv_checked/cold_drug_fold_metrics.csv")
    baseline = baseline.loc[baseline["stage"].eq("B0")].copy()
    columns = [
        "stage", "model", "variant", "subset", "fold", "n_rows", "rmse", "mae",
        "r2_normalized", "spearman_per_drug", "pearson", "spearman",
    ]
    combined = pd.concat([folds[columns], baseline[columns]], ignore_index=True)
    assert not combined.duplicated(["stage", "model", "variant", "subset", "fold"]).any()
    for _, group in combined.loc[combined["subset"].eq("test")].groupby(
        ["stage", "model", "variant"]
    ):
        assert set(group["fold"]) == set(range(5)), "Incomplete final CV comparison"

    tuning = []
    for fold in range(5):
        frame = read_csv(f"results/roadmap_2_6/random_forest/fold_{fold}/tuning_candidates.csv")
        for row in frame.to_dict("records"):
            params = json.loads(row.pop("params"))
            row.update(
                fold=fold,
                candidate=(
                    f"features={params['max_features']}; leaf={params['min_samples_leaf']}"
                ),
            )
            tuning.append(row)
    assert len(tuning) == 40

    # All outer-test pairs, exactly once per model. Binning uses identical edges
    # and includes every observation: no aesthetic trimming or random sampling.
    predictions = {}
    specifications = {
        "B0": ("results/predictions/cv_checked/cold_drug/fold_{fold}/b0.csv", "mean_effects"),
        "RF_targets": (
            "results/roadmap_2_6/random_forest/fold_{fold}/predictions.csv",
            "targets_expression_pca",
        ),
        "XGB_targets": (
            "results/roadmap_2_6/ablation/fold_{fold}/standard/predictions.csv",
            "targets_expression_pca",
        ),
    }
    for name, (pattern, model) in specifications.items():
        frames = []
        for fold in range(5):
            frame = read_csv(pattern.format(fold=fold))
            selected = frame.loc[frame["model"].eq(model) & frame["subset"].eq("test")]
            assert len(selected) > 0
            frames.append(selected[["pair_id", "y_true", "y_pred"]])
        frame = pd.concat(frames).sort_values("pair_id").reset_index(drop=True)
        assert len(frame) == 149364 and frame["pair_id"].is_unique
        assert np.isfinite(frame[["y_true", "y_pred"]].to_numpy()).all()
        predictions[name] = frame
    reference = predictions["B0"]
    for frame in predictions.values():
        assert frame["pair_id"].equals(reference["pair_id"])
        np.testing.assert_allclose(frame["y_true"], reference["y_true"], rtol=0, atol=1e-10)
    lower = min(frame[["y_true", "y_pred"]].to_numpy().min() for frame in predictions.values())
    upper = max(frame[["y_true", "y_pred"]].to_numpy().max() for frame in predictions.values())
    edges = np.linspace(np.floor(lower), np.ceil(upper), 71)
    density = {}
    for name, frame in predictions.items():
        counts, _, _ = np.histogram2d(frame["y_true"], frame["y_pred"], bins=(edges, edges))
        assert counts.sum() == len(frame)
        residual = frame["y_pred"].to_numpy() - frame["y_true"].to_numpy()
        density[name] = {
            "counts": counts.astype(int).tolist(), "n_pairs": len(frame),
            "pooled_rmse": float(np.sqrt(np.mean(residual ** 2))),
            "pooled_mae": float(np.mean(np.abs(residual))),
            "bias_pred_minus_true": float(np.mean(residual)),
        }

    report = {
        "schema_version": 1,
        "purpose": "Compact, auditable plotting snapshot; no model fitting or metric selection",
        "fold_metrics": json.loads(combined.to_json(orient="records", double_precision=15)),
        "rf_validation_tuning": tuning,
        "prediction_density": {"edges": edges.tolist(), "models": density},
        "source_sha256": sources,
        "density_scope": "All 149364 outer-test pairs once per model; 70x70 common bins. "
        "Pooled RMSE differs from the unweighted mean of five fold RMSEs.",
    }
    output = root / "data/reports/final_results_plot_data.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {output}: {len(combined)} metric rows, 40 tuning candidates, 3 densities")


if __name__ == "__main__":
    main()
