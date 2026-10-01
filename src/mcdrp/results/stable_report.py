"""Professional, portable notebook and release checks for newly fitted stable runs."""
# Notebook cell source strings preserve plotting and presentation layout.
# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mcdrp.experiments.stable_v1 import scientific_config_sha256, sha256, write_json
from mcdrp.metrics import regression_metrics
from mcdrp.models.stable_core import SPECIFICATIONS
from mcdrp.results.uncertainty import paired_bootstrap, uncertainty_report


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def build_report(root: Path, *, golden: dict | None = None) -> None:
    import joblib

    from mcdrp.features.expression import load_expression_matrix, transform_expression

    repository = Path(__file__).resolve().parents[3]
    source_paths = [
        *repository.joinpath("src").rglob("*.py"),
        *repository.joinpath("scripts").glob("*.py"),
        *repository.joinpath("scripts").glob("*.ps1"),
        *repository.joinpath("requirements").glob("*"),
        repository / "pyproject.toml",
        repository / ".gitattributes",
    ]
    write_json(
        root / "reports/source_code.json",
        {
            "recorded_at": "report construction after training",
            "files": {
                path.relative_to(repository).as_posix(): sha256(path)
                for path in sorted(source_paths)
                if path.is_file()
            },
        },
    )
    expression_path = json.loads((root / "reports/cohort.json").read_text())["inputs"][
        "depmap_expression"
    ]
    cohort_ids = set(pd.read_csv(root / "data/cohort.csv", usecols=["depmap_id"])["depmap_id"])
    expression = load_expression_matrix(expression_path, model_ids=cohort_ids)
    preprocessing_checks = []
    split_populations = []
    metric_frames, calibration, candidates, bootstrap, densities, drug_errors = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    pooled: dict[tuple[str, str], list[pd.DataFrame]] = {}
    sources = {}
    histories = []
    n_models = 0
    for case_path in sorted((root / "cases").glob("*.json")):
        case = json.loads(case_path.read_text())
        features = joblib.load(case["features"])
        case_expression = expression.loc[expression.index.isin(features["cells"])]
        restored = transform_expression(features["expression_pipeline"], case_expression)
        expected_features = np.vstack([features["cells"][k] for k in case_expression.index])
        np.testing.assert_allclose(restored, expected_features, rtol=1e-6, atol=2e-5)
        preprocessing_checks.append(
            {
                "case": case["name"],
                "max_abs_delta": float(np.max(np.abs(restored - expected_features))),
                "cells": len(restored),
                "train_cells": len(features["train_cell_ids"]),
                "genes_retained": len(features["expression_pipeline"].gene_columns),
                "pca_components": features["expression_pipeline"].n_components,
                "explained_variance": features[
                    "expression_pipeline"
                ].explained_variance_ratio_sum,
                "computation_dtype": features["expression_pipeline"].computation_dtype,
                "blas_threads": features["expression_pipeline"].blas_threads,
                "status": "verified",
            }
        )
        del features, restored, expected_features
        cohort = pd.read_csv(case["cohort"], low_memory=False)
        assigned = cohort.merge(
            pd.read_csv(case["assignments"]), on="pair_id", validate="one_to_one"
        )
        if case["split_name"] == "cold_drug":
            active = assigned.loc[assigned["split"].ne("unused")]
            if active.groupby("chemical_id")["split"].nunique().gt(1).any():
                raise ValueError("Chemical identity crosses cold-drug subsets.")
        for subset, rows in assigned.groupby("split", sort=True):
            split_populations.append(
                {
                    "case": case["name"], "subset": subset, "pairs": len(rows),
                    "drug_ids": rows["drug_id"].nunique(),
                    "chemical_identities": rows["chemical_id"].nunique(),
                    "cells": rows["depmap_id"].nunique(),
                }
            )
        fits = sorted(Path(case["output"]).glob("*/fit.json"))
        expected = 2 if "_calibration" in case["name"] else len(SPECIFICATIONS)
        if len(fits) != expected:
            raise ValueError(f"Incomplete model inventory: {case['name']}")
        test_predictions = {}
        for fit_path in fits:
            fit = json.loads(fit_path.read_text())
            if fit["replay_max_abs_delta"] > 1e-7:
                raise ValueError("Persisted model failed prediction replay.")
            n_models += 1
            directory = fit_path.parent
            if fit["stage"] in {"B6", "B7"}:
                estimator = joblib.load(directory / "model.joblib")["model"]
                values = estimator.evals_result()["validation_0"]["rmse"]
                if not np.isfinite(values).all() or fit["best_iteration"] >= len(values):
                    raise ValueError("Validation learning history is incomplete or nonfinite.")
                histories.append(
                    {
                        "case": case["name"],
                        "stage": fit["stage"],
                        "model": fit["model"],
                        "best_iteration": fit["best_iteration"],
                        "validation_rmse": values,
                    }
                )
                del estimator
            for path in (fit_path, directory / "metrics.csv", directory / "predictions.csv"):
                sources[path.relative_to(root).as_posix()] = sha256(path)
            metrics = pd.read_csv(directory / "metrics.csv")
            metrics["case"] = case["name"]
            metric_frames.append(metrics)
            prediction = pd.read_csv(directory / "predictions.csv")
            test = prediction.loc[prediction["subset"].eq("test")].merge(
                cohort[["pair_id", "drug_id", "depmap_id", "chemical_id"]],
                on="pair_id", validate="one_to_one"
            )
            label = f"{fit['stage']} · {fit['model']}"
            test_predictions[label] = test
            candidates.extend(
                [
                    {**row, **json.loads(row["params"]), "case": case["name"], "model": label}
                    for row in fit["candidates"]
                ]
            )
            if "_calibration" in case["name"]:
                cal = uncertainty_report(
                    prediction,
                    cohort,
                    str(Path(case["assignments"]).parent),
                    target=case["target"],
                    group="chemical_id" if case["split_name"] == "cold_drug" else "depmap_id",
                    bounds=None if case["target"] == "ln_ic50" else (0.0, 1.0),
                )
                cal["case"] = case["name"]
                calibration.append(cal)
                cal.to_csv(directory / "uncertainty.csv", index=False)
            elif case["name"].startswith("cold_drug"):
                pooled.setdefault((fit["stage"], fit["model"]), []).append(test)
        if "_calibration" not in case["name"]:
            ref = test_predictions["B0 · mean_effects"]
            for label, test in test_predictions.items():
                if label.startswith("B0"):
                    continue
                paired = test.merge(
                    ref[["pair_id", "y_pred"]],
                    on="pair_id",
                    suffixes=("", "_reference"),
                    validate="one_to_one",
                )
                bootstrap.append(
                    {
                        "case": case["name"],
                        "model": label,
                        **paired_bootstrap(
                            paired["y_true"].to_numpy(),
                            paired["y_pred"].to_numpy(),
                            paired["y_pred_reference"].to_numpy(),
                            paired["chemical_id"].to_numpy(),
                        ),
                    }
                )
    pooled_rows = []
    coverage = {}
    ref = pd.concat(pooled[("B0", "mean_effects")], ignore_index=True)
    for (stage, model), frames in pooled.items():
        test = pd.concat(frames, ignore_index=True)
        if test["pair_id"].duplicated().any() or set(test["pair_id"]) != set(ref["pair_id"]):
            raise ValueError("Out-of-fold test predictions must cover each cohort pair once.")
        coverage[f"{stage}_{model}"] = len(test)
        pooled_rows.append(
            {
                "stage": stage,
                "model": model,
                **regression_metrics(
                    test["y_true"].to_numpy(),
                    test["y_pred"].to_numpy(),
                    cell_keys=test["depmap_id"].tolist(),
                    drug_keys=test["drug_id"].tolist(),
                ),
            }
        )
        label = f"{stage} · {model}"
        if stage != "B0":
            paired = test.merge(
                ref[["pair_id", "y_pred"]],
                on="pair_id",
                suffixes=("", "_reference"),
                validate="one_to_one",
            )
            bootstrap.append(
                {
                    "case": "cold_drug_pooled",
                    "model": label,
                    **paired_bootstrap(
                        paired["y_true"].to_numpy(),
                        paired["y_pred"].to_numpy(),
                        paired["y_pred_reference"].to_numpy(),
                        paired["chemical_id"].to_numpy(),
                    ),
                }
            )
        edges = np.linspace(-12, 15, 71)
        histogram, _, _ = np.histogram2d(test["y_true"], test["y_pred"], bins=(edges, edges))
        densities.append(
            {
                "model": label,
                "edges": edges.tolist(),
                "counts": histogram.tolist(),
                "n": len(test),
                "outside_plot": int(len(test) - histogram.sum()),
            }
        )
        errors = (
            test.assign(squared_error=(test["y_true"] - test["y_pred"]) ** 2)
            .groupby("drug_id")["squared_error"]
            .mean()
            .pow(0.5)
        )
        drug_errors.extend(
            [{"model": label, "drug_id": str(k), "rmse": float(v)} for k, v in errors.items()]
        )
    metrics = pd.concat(metric_frames, ignore_index=True)
    main = metrics.loc[
        metrics["case"].str.fullmatch(r"cold_drug_fold_\d") & metrics["subset"].eq("test")
    ]
    aggregate = (
        main.groupby(["stage", "model"])
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            spearman_mean=("spearman", "mean"),
            r2_normalized_mean=("r2_normalized", "mean"),
            spearman_per_drug_mean=("spearman_per_drug", "mean"),
            n_folds=("case", "nunique"),
        )
        .reset_index()
    )
    if len(aggregate) != 6 or not aggregate["n_folds"].eq(5).all():
        raise ValueError("Stable primary report requires six models across five folds.")
    # Predeclared reproduction tolerance; never used to choose/tune a model.
    config = json.loads((root / "config.json").read_text())
    references = config.get("legacy_reference_rmse", {})
    if set(references) != {f"{stage}_{model}" for stage, model, _ in SPECIFICATIONS}:
        raise ValueError("The stable contract needs all six predeclared reference scores.")
    comparisons = []
    for row in records(aggregate):
        key = f"{row['stage']}_{row['model']}"
        if key in references:
            delta = abs(row["rmse_mean"] - references[key])
            comparisons.append(
                {
                    "model": key,
                    "reference": references[key],
                    "fresh": row["rmse_mean"],
                    "abs_delta": delta,
                    "within_legacy_tolerance": delta <= config["reference_rmse_absolute_tolerance"],
                }
            )
    # Legacy float32 PCA and the revised float64 recipe are distinct numerical
    # experiments. Preserve the failed migration check; never relax its tolerance.
    reproduction = []
    case_reproduction = []
    derived_data_sha256 = {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted((root / "data").rglob("*.csv"))
    }
    test_reference = {
        f"{row['case']}/{row['stage']}_{row['model']}": row["rmse"]
        for row in records(metrics.loc[metrics["subset"].eq("test")])
    }
    if len(test_reference) != 60:
        raise ValueError("The release requires one test RMSE for each of its 60 fits.")
    if golden is not None:
        if derived_data_sha256 != golden["derived_data_sha256"]:
            raise ValueError("Reconstructed cohorts, curve fits or assignments differ from the release.")
        if set(test_reference) != set(golden["test_rmse"]):
            raise ValueError("The golden reference describes another model/case inventory.")
        for key, fresh in test_reference.items():
            delta = abs(fresh - golden["test_rmse"][key])
            case_reproduction.append(
                {
                    "case_model": key,
                    "reference": golden["test_rmse"][key],
                    "fresh": fresh,
                    "abs_delta": delta,
                    "accepted": delta <= config["reference_rmse_absolute_tolerance"],
                }
            )
        for row in records(aggregate):
            key = f"{row['stage']}_{row['model']}"
            delta = abs(row["rmse_mean"] - golden["reference_rmse"][key])
            reproduction.append(
                {
                    "model": key,
                    "reference": golden["reference_rmse"][key],
                    "fresh": row["rmse_mean"],
                    "abs_delta": delta,
                    "accepted": delta <= config["reference_rmse_absolute_tolerance"],
                }
            )
        if not all(row["accepted"] for row in [*reproduction, *case_reproduction]):
            write_json(
                root / "reports/reproduction_failure.json",
                {"primary": reproduction, "all_cases": case_reproduction},
            )
            raise ValueError("Fresh fitting exceeds the frozen release-reference tolerance.")
    summary = {
        "protocol": "stable_v1",
        "status": "models_validated",
        "models_reloaded": n_models,
        "preprocessing_reloaded": preprocessing_checks,
        "split_populations": split_populations,
        "metrics": records(metrics),
        "aggregate_test": records(aggregate),
        "pooled_test": records(pd.DataFrame(pooled_rows)),
        "bootstrap": bootstrap,
        "calibration": records(pd.concat(calibration, ignore_index=True)),
        "rf_candidates": candidates,
        "learning_histories": histories,
        "densities": densities,
        "drug_errors": drug_errors,
        "out_of_fold_coverage": coverage,
        "source_sha256": sources,
        "reproduction": reproduction,
        "case_reproduction": case_reproduction,
        "derived_data_sha256": derived_data_sha256,
        "reference_status": "matched_release_golden"
        if golden
        else "establishing_release_golden",
        "legacy_checkpoint_comparison": comparisons,
        "legacy_comparison_role": "Diagnostic only: historical float32 preprocessing and drug-ID folds differ from this protocol",
        "numerical_validation": json.loads(
            (root / "reports/numerical_stability.json").read_text()
        ),
        "config": config,
        "environment": json.loads((root / "environment.json").read_text()),
        "cohort": json.loads((root / "reports/cohort.json").read_text()),
        "chemistry": json.loads((root / "reports/chemical_identity.json").read_text()),
        "harmonization": json.loads((root / "reports/harmonization.json").read_text()),
        "limits": [
            "Exploratory research; PRISM has already been examined.",
            "Overlapping CV training folds are not independent studies.",
            "Entity bootstrap is conditional on fitted models, not training uncertainty.",
            "Frozen ChemBERTa features are verified inputs, not regenerated encoder weights.",
            "Shared dose window does not equate exposure time or assay biology.",
            "Published and harmonized AUC differ in definition; compare against B0 within each endpoint.",
            "Coverage failure in PRISM remains a valid negative result.",
        ],
    }
    write_json(root / "reports/stable_v1_summary.json", summary)
    metrics.to_csv(root / "reports/metrics.csv", index=False)
    create_notebook(root)
    summary["status"] = "validated"
    write_json(root / "reports/stable_v1_summary.json", summary)
    if golden is None:
        write_json(
            root / "reports/stable_v1_golden_candidate.json",
            {
                "protocol": "stable_v1",
                "preprocessing_version": config["preprocessing_version"],
                "scientific_config_sha256": scientific_config_sha256(config),
                "reference_rmse": {
                    f"{row['stage']}_{row['model']}": row["rmse_mean"]
                    for row in records(aggregate)
                },
                "test_rmse": test_reference,
                "derived_data_sha256": derived_data_sha256,
                "status": "candidate_after_full_validation",
            },
        )


def create_notebook(root: Path) -> None:
    import nbformat
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter

    cells = []

    def md(source: str) -> None:
        cells.append(nbformat.v4.new_markdown_cell(source))

    def code(source: str) -> None:
        cells.append(nbformat.v4.new_code_cell(source))

    md("""# Respuesta farmacológica multimodal · versión estable v1
**Reproducción desde datos originales en un entorno aislado · octubre de 2026.**

Este notebook presenta los seis modelos del protocolo congelado, cinco folds de
fármacos nuevos, transferencia GDSC→PRISM, armonización y calibración independiente.
Los IDs con la misma identidad química completa (InChIKey) permanecen juntos
en cada partición cold-drug y en la reserva de calibración.
Las métricas macro por fármaco conservan los IDs de los registros originales;
el bootstrap agrupa las identidades químicas para mantener juntos sus alias.
Cada modelo se ha guardado y recargado para verificar sus predicciones.
El escalado y PCA se ajustan en float64 con BLAS fijado internamente a ocho
hilos sobre células de entrenamiento; las
características finales se convierten a float32. Se comprueba la estabilidad
de las proyecciones entre 8 y 16 hilos de BLAS sin usar respuestas del test.
Las figuras se regeneran desde el informe compacto; el notebook no reentrena.
Las «dianas» son anotaciones putativas que incluyen genes, familias y mecanismos.

RMSE menor indica mejor predicción. Los resultados de IC50 y AUC tienen escalas
distintas. PRISM es una evaluación exploratoria ya examinada; no constituye una
validación clínica. Las demás arquitecturas permanecen experimentales.
""")
    code("""from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from IPython.display import display, Image

HERE = Path.cwd()
REPORT = next((p for p in [HERE / "reports/stable_v1_summary.json",
    HERE / "data/reports/stable_v1_summary.json",
    HERE.parent / "data/reports/stable_v1_summary.json"] if p.exists()), None)
if REPORT is None:
    raise FileNotFoundError("No se encuentra el informe estable v1.")
s = json.loads(REPORT.read_text(encoding="utf-8"))
assert s["status"] in {"models_validated", "validated"} and s["models_reloaded"] == 60
metrics = pd.DataFrame(s["metrics"])
agg = pd.DataFrame(s["aggregate_test"]).sort_values("rmse_mean")
main_check = metrics.loc[metrics.case.str.fullmatch(r"cold_drug_fold_\\d") & metrics.subset.eq("test")]
for row in agg.itertuples():
    selected = main_check.loc[main_check.stage.eq(row.stage) & main_check.model.eq(row.model)]
    assert len(selected) == 5 and np.isclose(selected.rmse.mean(), row.rmse_mean, atol=1e-10)
assert len(s["out_of_fold_coverage"]) == 6
assert set(s["out_of_fold_coverage"].values()) == {149364}
MODEL_LABELS = {"mean_effects": "Efectos medios", "targets_expression_pca": "Dianas + PCA",
    "morgan_expression_pca": "Morgan + PCA", "pretrained_drug_expression_pca": "ChemBERTa + PCA"}
CASE_ORDER = ["published_transfer", "matched_published_transfer", "harmonized_transfer"]
CASE_LABELS = {"published_transfer": "AUC publicada · total",
    "matched_published_transfer": "AUC publicada · mismos pares",
    "harmonized_transfer": "AUC armonizada · mismos pares"}
METHOD_LABELS = {"raw": "Sin ajuste", "affine_validation": "Ajuste afín (validación)"}
def pretty_model(value):
    stage, name = value.split(" · ", 1)
    return stage + " · " + MODEL_LABELS.get(name, name)
def label(frame):
    return frame.stage + " · " + frame.model.map(MODEL_LABELS)
agg["label"] = label(agg)
REPORT_ROOT = REPORT.parent.parent
if REPORT_ROOT.name == "data":
    REPORT_ROOT = REPORT_ROOT.parent
OUT = REPORT_ROOT / "figures/stable_v1"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "figure.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
exports = []
def save(fig, name):
    fig.tight_layout()
    for suffix in ("png", "svg", "pdf"):
        path = OUT / f"{name}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        exports.append(str(path))
    display(Image(filename=str(OUT / f"{name}.png")))
    plt.close(fig)
display(agg[["label", "rmse_mean", "rmse_std", "spearman_mean", "r2_normalized_mean", "spearman_per_drug_mean", "n_folds"]])
""")
    md(
        "## Preprocessing y separación de datos\nEl filtrado de genes de baja varianza, escalado y PCA se ajustan solo sobre células de entrenamiento. La tabla muestra los objetos recargados y su reproducción desde la expresión original. En cold-drug se retienen fármacos nuevos, por lo que las células pueden ser conocidas. La validación selecciona los modelos; la calibración reserva entidades adicionales y el test se evalúa después."
    )
    code('''display(pd.DataFrame(s["preprocessing_reloaded"]))
print("IDs de fármacos:", s["chemistry"]["drug_ids"],
      "· identidades químicas:", s["chemistry"]["chemical_identities"])
display(pd.DataFrame(s["chemistry"]["aliases"]))''')
    code('''populations = pd.DataFrame(s["split_populations"])
display(populations.loc[populations.case.str.fullmatch(r"cold_drug_fold_\\d")])
display(populations.loc[populations.subset.eq("calibration")])''')
    md(
        "## 1. Rendimiento en fármacos nuevos\nMedia y desviación entre folds; la dispersión no es un intervalo de confianza."
    )
    code("""fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(agg.label, agg.rmse_mean, xerr=agg.rmse_std, color="#227C9D", capsize=4)
ax.invert_yaxis()
ax.set(xlabel="RMSE de ln(IC50) · media ± DE entre 5 folds", title="Comparación completa del protocolo principal")
save(fig, "01_cold_drug_rmse")
main = metrics.loc[metrics.case.str.fullmatch(r"cold_drug_fold_\\d") & metrics.subset.eq("test")].copy()
main["label"] = label(main)
fig, ax = plt.subplots(figsize=(9, 4))
for name, rows in main.groupby("label"):
    ax.plot(rows.case.str.extract(r"(\\d)$")[0].astype(int), rows.rmse, "o-", label=name)
ax.set(xlabel="Fold de fármacos retenidos", ylabel="RMSE · ln(IC50)", xticks=range(5), title="Variación por composición del fold")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
save(fig, "02_folds")
fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(agg.label, agg.spearman_mean, color="#17A398")
ax.invert_yaxis()
ax.set(xlabel="Spearman global · media de 5 folds", title="Ordenación de respuestas fuera de muestra")
save(fig, "03_spearman")
fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(agg.label, agg.r2_normalized_mean, color="#227C9D")
ax.axvline(0, color="#d1495b", linestyle="--")
ax.invert_yaxis()
ax.set(xlabel="R² normalizado respecto a B0 · media de folds", title="Mejora relativa a los efectos medios")
save(fig, "03b_r2_normalizado")
fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(agg.label, agg.spearman_per_drug_mean, color="#17A398")
ax.invert_yaxis()
ax.set(xlabel="Spearman por ID de fármaco · media macro de folds", title="Ordenación de células para un mismo registro farmacológico")
save(fig, "03c_spearman_por_farmaco")
""")
    md(
        "## 2. Predicciones fuera de muestra\nCada uno de los 149.364 pares aparece exactamente una vez por modelo. La densidad conserva todos los pares; se declara cualquier punto fuera de los límites dibujados."
    )
    code("""fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
common_max = max(np.max(row["counts"]) for row in s["densities"])
for ax, row in zip(axes.flat, s["densities"]):
    edges = np.array(row["edges"])
    im = ax.pcolormesh(edges, edges, np.array(row["counts"]).T, norm=LogNorm(vmin=1, vmax=common_max), cmap="viridis")
    ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]], "--", color="#d1495b", lw=1)
    ax.set(title=pretty_model(row["model"]), xlabel="Observado · ln(IC50)", ylabel="Predicho · ln(IC50)")
    ax.text(.02, .97, f"n={row['n']:,}; fuera={row['outside_plot']}", transform=ax.transAxes, va="top", fontsize=8)
    fig.colorbar(im, ax=ax, label="Pares por celda")
save(fig, "04_densidad_predicciones")
errors = pd.DataFrame(s["drug_errors"])
groups = [r.model for r in pd.DataFrame(s["pooled_test"]).itertuples()]
fig, ax = plt.subplots(figsize=(10, 4))
labels = errors.model.unique()
ax.boxplot([errors.loc[errors.model.eq(k), "rmse"] for k in labels], tick_labels=[pretty_model(k) for k in labels], showfliers=True)
ax.tick_params(axis="x", rotation=25)
ax.set(ylabel="RMSE por ID de fármaco", title="Heterogeneidad entre los 238 IDs farmacológicos")
save(fig, "05_error_por_farmaco")
""")
    md(
        "## 3. Incertidumbre de la comparación\nBootstrap pareado por identidad química completa (2.000 remuestreos), manteniendo juntos los IDs alias y condicionado a los modelos ya ajustados. Una diferencia negativa favorece al candidato frente a B0; no mide variación por reentrenamiento."
    )
    code("""boot = pd.DataFrame(s["bootstrap"])
pooled = boot.loc[boot.case.eq("cold_drug_pooled")].sort_values("rmse_delta")
fig, ax = plt.subplots(figsize=(9, 4))
ax.errorbar(pooled.rmse_delta, pooled.model.map(pretty_model), xerr=np.vstack([pooled.rmse_delta-pooled.ci95_low, pooled.ci95_high-pooled.rmse_delta]), fmt="o", capsize=4, color="#227C9D")
ax.axvline(0, color="#d1495b", linestyle="--")
ax.invert_yaxis()
ax.set(xlabel="Δ RMSE respecto a B0 · IC95% bootstrap por identidad química", title="Comparación pareada de todos los pares fuera de muestra")
save(fig, "06_bootstrap")
""")
    md(
        "## 4. Selección de Random Forest\nLos cuatro candidatos se comparan usando solo validación. El test se calcula después de elegir el modelo."
    )
    code("""grid = pd.DataFrame(s["rf_candidates"])
display(grid.head(12))
fig, ax = plt.subplots(figsize=(10, 4))
for name, rows in grid.loc[grid.case.str.fullmatch(r"cold_drug_fold_\\d")].groupby("model"):
    column = "validation_rmse" if "validation_rmse" in rows else "rmse"
    values = rows.groupby(["max_features", "min_samples_leaf"], dropna=False)[column].mean()
    ax.plot([str(k) for k in values.index], values.values, "o-", label=pretty_model(name))
ax.set(xlabel="(max_features, min_samples_leaf)", ylabel="RMSE medio de validación", title="Grid predefinido; cinco folds")
ax.legend(fontsize=8)
save(fig, "07_seleccion_rf")
""")
    md(
        "## Evolución del ajuste XGBoost\nRMSE de validación por iteración; el punto marca la iteración seleccionada. Se permite un máximo de 700 iteraciones y una paciencia de 40; el ajuste puede terminar por cualquiera de esos límites. Las curvas pueden continuar después de la mejor iteración durante la paciencia. No se utiliza el test para decidir dónde detener el entrenamiento."
    )
    code("""fig, axes = plt.subplots(3, 1, figsize=(10, 9))
specs = [("B6", "targets_expression_pca"), ("B6", "morgan_expression_pca"), ("B7", "pretrained_drug_expression_pca")]
for ax, (stage, model) in zip(axes, specs):
    for history in s["learning_histories"]:
        if history["stage"] != stage or history["model"] != model or not pd.Series([history["case"]]).str.fullmatch(r"cold_drug_fold_\\d").iloc[0]:
            continue
        values = history["validation_rmse"]
        chosen = history["best_iteration"]
        line, = ax.plot(range(len(values)), values, label="Fold " + history["case"].rsplit("_", 1)[1])
        ax.scatter([chosen], [values[chosen]], color=line.get_color(), s=35, zorder=3)
    ax.set(title=pretty_model(stage + " · " + model), xlabel="Iteración (índice desde 0)", ylabel="RMSE de validación")
    ax.legend(fontsize=8, ncol=5)
save(fig, "07b_validacion_xgboost")
""")
    md(
        "## 5. Transferencia GDSC→PRISM\nMapeo químico estricto: 44 coincidencias completas de InChIKey y dos coincidencias inequívocas por conectividad; estas últimas no confirman identidad estereoquímica. Se muestran AUC publicada, AUC armonizada y AUC publicada sobre exactamente los mismos pares elegibles para armonización. El ajuste 3PL exige soporte de dosis en log10(µM) [-1,9; 0,9] y no extrapola. Las definiciones de AUC son distintas: se debe comparar cada modelo con B0 dentro del mismo endpoint. Cambiar la ventana no iguala el tiempo de exposición ni la biología del ensayo."
    )
    code("""transfer = metrics.loc[metrics.case.str.endswith("transfer") & metrics.subset.eq("test")].copy()
transfer["label"] = label(transfer)
table = transfer.pivot(index="label", columns="case", values="rmse").reindex(columns=CASE_ORDER).rename(columns=CASE_LABELS)
fig, ax = plt.subplots(figsize=(10, 5))
table.plot.barh(ax=ax, color=["#227C9D", "#17A398", "#E9B44C"])
ax.set(xlabel="RMSE · AUC (escala 0–1)", title="Transferencia: endpoint y población de evaluación")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
save(fig, "08_transferencia")
display(transfer[["case", "label", "n_rows", "rmse", "spearman"]] if "n_rows" in transfer else transfer[["case", "label", "rmse", "spearman"]])
display(pd.DataFrame(s["harmonization"]["counts"]).rename(columns={
    "study": "Ensayo", "split": "Subconjunto", "n_pairs": "Pares", "n_cells": "Células", "n_drugs": "Fármacos"}))
display(pd.DataFrame(s["harmonization"]["fit_status"]).pivot(index="status", columns="study", values="n").fillna(0))
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
for ax, case in zip(axes, CASE_ORDER):
    rows = boot.loc[boot.case.eq(case)]
    rows = rows.sort_values("rmse_delta")
    ax.errorbar(rows.rmse_delta, rows.model.map(pretty_model), xerr=np.vstack([rows.rmse_delta-rows.ci95_low, rows.ci95_high-rows.rmse_delta]), fmt="o", capsize=3)
    ax.axvline(0, linestyle="--", color="#d1495b")
    ax.set(title=CASE_LABELS[case].replace(" · ", "\\n"), xlabel="Δ RMSE frente a B0")
save(fig, "09_bootstrap_transferencia")
""")
    md(
        "## 6. Calibración independiente\nSe reservan entidades de entrenamiento exclusivamente para calibración; nunca se ajusta el radio con test. Se evalúan intervalos por máximo residual de entidad, crudos y con transformación afín ajustada en validación. La línea del 90% es un objetivo, no una garantía bajo cambio de distribución."
    )
    code("""cal = pd.DataFrame(s["calibration"])
display(cal)
coverage_column = "whole_group_coverage"
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, transfer_only in zip(axes, (False, True)):
    rows = cal.loc[cal.case.str.contains("transfer").eq(transfer_only)].copy()
    rows["label"] = label(rows) + " · " + rows.method.map(METHOD_LABELS)
    column = coverage_column
    grouped = rows.groupby("label")[column].mean()
    ax.barh(grouped.index, grouped.values, color="#17A398")
    ax.axvline(.9, color="#d1495b", linestyle="--", label="Objetivo 90%")
    ax.set(xlim=(0, 1), xlabel="Cobertura simultánea por entidad", title="PRISM · cambio de ensayo" if transfer_only else "Cold-drug · media de folds")
    ax.legend(fontsize=8)
save(fig, "10_cobertura")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, transfer_only in zip(axes, (False, True)):
    rows = cal.loc[cal.case.str.contains("transfer").eq(transfer_only)].copy()
    rows["label"] = label(rows) + " · " + rows.method.map(METHOD_LABELS)
    widths = rows.groupby("label").mean_width.mean()
    ax.barh(widths.index, widths.values, color="#227C9D")
    ax.set(xlabel="Amplitud media · AUC" if transfer_only else "Amplitud media · ln(IC50)", title="PRISM (límites [0, 1])" if transfer_only else "Cold-drug (media de folds)")
save(fig, "11_radios")
""")
    md(
        "## 7. Procedencia y aceptación\nEl informe corresponde a una ejecución nueva: seis modelos × ocho escenarios, más doce ajustes independientes para calibración. Los 60 modelos recargados reproducen sus predicciones. Los hashes de entrada y el entorno están congelados. La verificación del directorio completo se ejecuta con `python -m mcdrp.experiments.stable_v1 verify --output ...`. Los archivos externos grandes son necesarios para reentrenar, no para regenerar estas gráficas."
    )
    code("""display(pd.DataFrame(s["reproduction"]))
display(pd.DataFrame(s["case_reproduction"]))
print("Referencia de versión:", s["reference_status"])
print("Comparación histórica:", s["legacy_comparison_role"])
display(pd.DataFrame(s["legacy_checkpoint_comparison"]))
display(pd.DataFrame(s["numerical_validation"]))
display(pd.DataFrame.from_dict(s["out_of_fold_coverage"], orient="index", columns=["pares_test_unicos"]))
display(pd.DataFrame.from_dict(s["config"]["inputs"], orient="index"))
display(pd.DataFrame.from_dict(s["config"]["environment"], orient="index", columns=["version"]))
display(pd.DataFrame([{key: s["environment"][key] for key in
    ["python", "platform", "logical_cpus", "gpu_driver_memory"]}]))
for limitation in s["limits"]:
    print("•", limitation)
print(f"{len(exports)} archivos de figuras exportados (PNG 300 dpi, SVG y PDF).")
""")
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3 (stable v1)",
        "language": "python",
    }
    kernel_dir = root / "kernels" / "stable-v1"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    import sys

    write_json(
        kernel_dir / "kernel.json",
        {
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Python stable v1",
            "language": "python",
            "env": {
                "IPYTHONDIR": str(root / "kernels/ipython"),
                "MPLCONFIGDIR": str(root / "kernels/matplotlib"),
            },
        },
    )
    manager = KernelSpecManager(kernel_dirs=[str(root / "kernels")])
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="stable-v1",
        resources={"metadata": {"path": str(root)}},
    )
    from jupyter_client import KernelManager

    client.km = KernelManager(
        kernel_name="stable-v1",
        kernel_spec_manager=manager,
        connection_file=str(root / "kernels/connection.json"),
    )
    client.execute()
    nbformat.write(notebook, root / "stable_v1_results.ipynb")
    exporter = HTMLExporter(
        template_name="classic",
        exclude_input=True,
        exclude_input_prompt=True,
        exclude_output_prompt=True,
    )
    html, _ = exporter.from_notebook_node(notebook)
    from bs4 import BeautifulSoup

    document = BeautifulSoup(html, "html.parser")
    for script in document.find_all("script"):
        script.decompose()
    (root / "stable_v1_results.html").write_text(str(document), encoding="utf-8")
    n_figures = sum(
        "image/png" in output.get("data", {})
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )
    if n_figures != 14:
        raise ValueError(f"Expected 14 rendered professional figures, got {n_figures}")
    n_exports = sum(
        path.suffix in {".png", ".svg", ".pdf"}
        for path in (root / "figures/stable_v1").iterdir()
    )
    if n_exports != 42:
        raise ValueError(f"Expected 42 exported figure files, got {n_exports}")
    write_json(
        root / "reports/notebook_validation.json",
        {
            "status": "verified",
            "figures": n_figures,
            "code_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
            "exports": n_exports,
            "executed_with": sys.executable,
        },
    )
    # Kernel caches are execution scratch, not fitted scientific artifacts.
    import shutil

    scratch = (root / "kernels").resolve()
    if not scratch.is_relative_to(root.resolve()) or scratch.name != "kernels":
        raise ValueError("Kernel scratch directory escapes the completed run.")
    shutil.rmtree(scratch)
