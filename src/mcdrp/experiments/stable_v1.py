"""Rebuild the stable protocol from verified inputs in an empty output directory.

No derived cohort, fitted preprocessing, model or prediction is read from older runs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def input_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or Path(relative).is_absolute():
        raise ValueError(f"Input path escapes the input bundle: {relative}")
    return path


def check_inputs(config: dict, root: Path) -> dict[str, str]:
    paths = {}
    for key, item in config["inputs"].items():
        path = input_path(root, item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Frozen input checksum failed: {key}")
        paths[key] = str(path)
    return paths


def check_environment(config: dict) -> dict:
    versions = {key: importlib.metadata.version(key) for key in config["environment"]}
    if versions != config["environment"]:
        raise ValueError("Installed packages differ from the stable environment lock.")
    if platform.python_version() != config["python"] or sys.platform != "win32":
        raise ValueError("This lock certifies Windows CPython 3.13.5 only.")
    import xgboost

    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        gpu = query.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        gpu = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "executable": sys.executable,
        "logical_cpus": os.cpu_count(),
        "xgboost_build": xgboost.build_info(),
        "gpu_driver_memory": gpu,
    }


def inventory(root: Path) -> list[dict]:
    return [
        {"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name not in {"manifest.json", "failure.json"}
    ]


def verify_run(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("The stable run is not complete.")
    if (root / "config.json").exists() and sha256(root / "config.json") != manifest[
        "config_sha256"
    ]:
        raise ValueError("Stable config hash differs from the release manifest.")
    expected = manifest["artifacts"]
    if inventory(root) != expected:
        raise ValueError("Stable run artifacts are missing, added or changed.")
    return {
        "status": "verified",
        "files": len(expected),
        "config_sha256": manifest["config_sha256"],
    }


def prepare(config: dict, inputs: dict, root: Path) -> list[Path]:
    import joblib
    import pandas as pd

    from mcdrp.data.build_cohort import build_cohort
    from mcdrp.data.build_transfer_cohort import build_transfer_cohort
    from mcdrp.data.harmonize_endpoints import harmonize_endpoints
    from mcdrp.data.target_annotations import audit_targets
    from mcdrp.features.chemical_identity import annotate_chemical_identity
    from mcdrp.features.drug_embeddings import load_drug_embeddings
    from mcdrp.features.embedding_audit import audit_embeddings
    from mcdrp.features.expression import (
        EXPRESSION_BLAS_THREADS,
        fit_expression_pipeline,
        load_expression_matrix,
        transform_expression,
    )
    from mcdrp.features.fingerprints import build_fingerprint_matrix
    from mcdrp.features.targets import build_target_matrix
    from mcdrp.splits.calibration import reserve_calibration
    from mcdrp.splits.grouped_cv import build_grouped_cv
    from mcdrp.splits.make_splits import build_splits

    if config["expression_blas_threads"] != EXPRESSION_BLAS_THREADS:
        raise ValueError("Configured expression BLAS profile differs from the implementation.")
    if config["cold_drug_grouping"] != "full_standard_inchikey":
        raise ValueError("Stable cold-drug evaluation requires complete chemical identities.")

    data, reports = root / "data", root / "reports"
    print("Rebuilding cohort and split assignments from raw inputs", flush=True)
    cohort_path = data / "cohort.csv"
    build_cohort(
        inputs["gdsc"],
        inputs["metadata"],
        inputs["expression"],
        inputs["structures"],
        cohort_path,
        reports / "cohort.json",
    )
    if sha256(cohort_path) != config["expected_cohort_sha256"]:
        raise ValueError("Raw preprocessing did not reproduce the frozen cohort.")
    write_json(
        reports / "embeddings.json",
        audit_embeddings(
            str(cohort_path),
            inputs["embeddings"],
            inputs["embedding_manifest"],
        ),
    )
    audit_targets(
        str(cohort_path),
        inputs["hgnc"],
        str(data / "annotations.csv"),
        str(reports / "annotations.json"),
    )
    chemical_cohort, chemistry = annotate_chemical_identity(pd.read_csv(cohort_path))
    chemical_cohort_path = data / "chemical_cohort.csv"
    chemical_cohort.to_csv(chemical_cohort_path, index=False)
    write_json(reports / "chemical_identity.json", chemistry)
    build_splits(
        cohort_path,
        data / "random",
        reports / "random.json",
        split_names=("random_pair",),
        seed=config["seed"],
    )
    build_grouped_cv(
        chemical_cohort_path,
        data / "cv",
        reports / "cv.json",
        split_names=("cold_drug",),
        seed=config["seed"],
        n_folds=5,
        group_columns={"cold_drug": "chemical_id"},
    )
    transfer_path = data / "published.csv"
    transfer_split = data / "transfer" / "transfer_prism.csv"
    build_transfer_cohort(
        cohort_path,
        inputs["prism"],
        data / "random/random_pair.csv",
        transfer_path,
        transfer_split,
        data / "drug_map.csv",
        reports / "transfer.json",
    )
    published, _ = annotate_chemical_identity(pd.read_csv(transfer_path, low_memory=False))
    published.to_csv(transfer_path, index=False)
    harmonization = {
        "cohort": str(transfer_path),
        "assignments": str(transfer_split),
        "drug_map": str(data / "drug_map.csv"),
        "prism_metadata": inputs["prism"],
        "sources": [
            {
                "study": "gdsc2",
                "path": inputs["gdsc_doses"],
                "dose_scale": "linear_um",
                "exposure_hours": None,
                "sha256": config["inputs"]["gdsc_doses"]["sha256"],
            },
            {
                "study": "prism",
                "path": inputs["prism_doses"],
                "dose_scale": "log10_um",
                "replicate_prefix": "MTS010",
                "exposure_hours": None,
                "sha256": config["inputs"]["prism_doses"]["sha256"],
            },
        ],
        "log10_um_lower": -1.9,
        "log10_um_upper": 0.9,
        "workers": 4,
        "curve_fits": str(data / "curve_fits.csv"),
        "output_cohort": str(data / "harmonized.csv"),
        "output_assignments": str(data / "harmonized/transfer_prism.csv"),
        "summary": str(reports / "harmonization.json"),
    }
    write_json(root / "harmonization_config.json", harmonization)
    print("Refitting dose-response curves on the shared concentration window", flush=True)
    harmonize_endpoints(str(root / "harmonization_config.json"))
    harmonized = pd.read_csv(data / "harmonized.csv", low_memory=False)
    specs = [
        (
            f"cold_drug_fold_{k}",
            chemical_cohort_path,
            data / f"cv/cold_drug/fold_{k}/cold_drug.csv",
            "cold_drug",
            "ln_ic50",
        )
        for k in range(5)
    ]
    specs += [
        ("published_transfer", transfer_path, transfer_split, "transfer_prism", "auc"),
        (
            "harmonized_transfer",
            data / "harmonized.csv",
            data / "harmonized/transfer_prism.csv",
            "transfer_prism",
            "auc",
        ),
    ]
    # Endpoint comparison uses exactly the same curve-eligible pairs and assignments.
    matched = harmonized.copy()
    published = pd.read_csv(transfer_path, low_memory=False).set_index("pair_id")
    matched["auc"] = matched["pair_id"].map(published["auc"])
    if matched["auc"].isna().any():
        raise ValueError("Matched endpoint comparison has missing published AUC.")
    matched.to_csv(data / "matched_published.csv", index=False)
    specs += [
        (
            "matched_published_transfer",
            data / "matched_published.csv",
            data / "harmonized/transfer_prism.csv",
            "transfer_prism",
            "auc",
        )
    ]
    for name, cp, ap, split, target in list(specs):
        if name.startswith("cold_drug") or name == "harmonized_transfer":
            frame, assignments = pd.read_csv(cp), pd.read_csv(ap)
            group = "chemical_id" if split == "cold_drug" else "depmap_id"
            reserved = reserve_calibration(
                frame, assignments, group=group, fraction=0.15, seed=config["seed"]
            )
            cal_path = data / "calibration" / name / f"{split}.csv"
            cal_path.parent.mkdir(parents=True, exist_ok=True)
            reserved.to_csv(cal_path, index=False)
            specs.append((f"{name}_calibration", cp, cal_path, split, target))
    cases = []
    embedding_map = load_drug_embeddings(inputs["embeddings"]).embedding_map
    all_cell_ids = set(pd.read_csv(cohort_path, usecols=["depmap_id"])["depmap_id"])
    expression = load_expression_matrix(inputs["expression"], model_ids=all_cell_ids)
    expression_cache: dict = {}
    numerical_checks = []
    for name, cp, ap, split, target in specs:
        print(f"Fitting and saving train-only preprocessing: {name}", flush=True)
        frame, assignments = pd.read_csv(cp, low_memory=False), pd.read_csv(ap)
        train = frame.merge(assignments, on="pair_id", validate="one_to_one")
        train = train.loc[train["split"].eq("train")]
        training_cells = frozenset(train["depmap_id"])
        # Reuse inside this fresh run only when the exact fitting entities agree.
        # Targets retain their own per-case training-drug vocabularies.
        if training_cells not in expression_cache:
            train_expression = expression.loc[expression.index.isin(training_cells)]
            expr = fit_expression_pipeline(
                train_expression, n_components=256, random_state=config["seed"]
            )
            transformed = transform_expression(expr, expression)
            cells = dict(zip(expression.index, transformed, strict=True))
            expression_cache[training_cells] = (expr, cells)
            # The validation uses expression values and identities only, no response labels.
            import numpy as np
            from threadpoolctl import threadpool_limits

            with threadpool_limits(limits=16, user_api="blas"):
                alternative = fit_expression_pipeline(
                    train_expression, n_components=256, random_state=config["seed"]
                )
                alternate_values = transform_expression(alternative, expression)
            delta = float(np.max(np.abs(alternate_values - transformed)))
            numerical_checks.append(
                {
                    "case": name,
                    "train_cells": len(training_cells),
                    "dtype": expr.computation_dtype,
                    "blas_threads": [8, 16],
                    "effective_expression_blas_threads": expr.blas_threads,
                    "max_abs_delta": delta,
                    "status": "verified"
                    if delta <= config["projection_absolute_tolerance"]
                    else "failed",
                }
            )
            write_json(reports / "numerical_stability.json", numerical_checks)
            if delta > config["projection_absolute_tolerance"]:
                raise ValueError(f"Expression BLAS-profile stability failed: {name}, {delta}")
        expr, cells = expression_cache[training_cells]
        catalog = frame.drop_duplicates("drug_id")
        target_pipeline, targets = build_target_matrix(
            catalog, set(train["drug_id"].astype(str))
        )
        fingerprints = build_fingerprint_matrix(catalog["canonical_smiles"])
        feature_path = root / "features" / f"{name}.joblib"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        keys = catalog["drug_id"].astype(str).tolist()
        joblib.dump(
            {
                "cells": cells,
                "targets": dict(zip(keys, targets, strict=True)),
                "morgan": dict(zip(keys, fingerprints, strict=True)),
                "embeddings": embedding_map,
                "expression_pipeline": expr,
                "target_pipeline": target_pipeline,
                "train_cell_ids": sorted(set(train["depmap_id"])),
                "train_drug_ids": sorted(set(train["drug_id"].astype(str))),
            },
            feature_path,
            compress=3,
        )
        case_path = root / "cases" / f"{name}.json"
        write_json(
            case_path,
            {
                "name": name,
                "cohort": str(cp),
                "assignments": str(ap),
                "target": target,
                "split_name": split,
                "chemical_group_column": "chemical_id",
                "features": str(feature_path),
                "output": str(root / "fits" / name),
                "rf_trees": config["rf_trees"],
                "xgb_params": config["xgb_params"],
            },
        )
        cases.append(case_path)
    write_json(reports / "numerical_stability.json", numerical_checks)
    return cases


def train_worker(root: Path, worker: str, shard: int = 0, shards: int = 1) -> None:
    from mcdrp.models.stable_core import train_case

    paths = sorted((root / "cases").glob("*.json"))
    if worker == "rf":
        paths = [path for path in paths if "_calibration" not in path.stem][shard::shards]
    for path in paths:
        calibration = "_calibration" in path.stem
        if worker == "rf" and calibration:
            continue
        families = (
            ("RF",) if worker == "rf" else (("B6",) if calibration else ("B0", "B6", "B7"))
        )
        train_case(str(path), families)


def scientific_config_sha256(config: dict) -> str:
    value = {key: item for key, item in config.items() if key != "release_reference_sha256"}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def run(
    config_path: Path, inputs_root: Path, root: Path, *, establish_golden: bool = False
) -> None:
    config = json.loads(config_path.read_text())
    environment = check_environment(config)
    inputs = check_inputs(config, inputs_root)
    golden = None
    if not establish_golden:
        reference = config_path.parent.parent / config["release_reference"]
        if sha256(reference) != config.get("release_reference_sha256"):
            raise ValueError("Release golden-reference checksum failed.")
        golden = json.loads(reference.read_text())
        if golden["scientific_config_sha256"] != scientific_config_sha256(config):
            raise ValueError("The golden reference describes another numerical/model recipe.")
    if root.exists():
        raise ValueError(
            "Choose a new output directory; stable runs never reuse cached outputs."
        )
    root.mkdir(parents=True)
    write_json(root / "config.json", config)
    write_json(root / "environment.json", environment)
    processes: list[subprocess.Popen] = []
    try:
        prepare(config, inputs, root)
        workers = [("rf", i, config["rf_workers"]) for i in range(config["rf_workers"])]
        workers.append(("gpu", 0, 1))
        for worker, shard, shards in workers:
            with (root / f"{worker}_{shard}.log").open("w", encoding="utf-8") as stream:
                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "mcdrp.experiments.stable_v1",
                            "worker",
                            "--output",
                            str(root),
                            "--worker",
                            worker,
                            "--shard",
                            str(shard),
                            "--shards",
                            str(shards),
                        ],
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        env=os.environ.copy(),
                    )
                )
        for process in processes:
            if process.wait() != 0:
                raise RuntimeError("Training worker failed; inspect rf.log/gpu.log.")
        from mcdrp.results.stable_report import build_report

        build_report(root, golden=golden)
        write_json(
            root / "manifest.json",
            {
                "status": "complete",
                "protocol": "stable_v1",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "config_sha256": sha256(root / "config.json"),
                "artifacts": inventory(root),
            },
        )
        print(json.dumps(verify_run(root)), flush=True)
    except BaseException as exc:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        write_json(
            root / "failure.json",
            {"status": "failed", "error": str(exc), "type": type(exc).__name__},
        )
        raise


def main() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "8"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "verify", "worker"])
    parser.add_argument("--config", type=Path, default=Path("configs/stable_v1.json"))
    parser.add_argument("--inputs-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("results/stable_v1"))
    parser.add_argument("--worker", choices=["rf", "gpu"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--establish-golden", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    if args.action == "run":
        run(
            args.config.resolve(),
            args.inputs_root.resolve(),
            root,
            establish_golden=args.establish_golden,
        )
    elif args.action == "worker":
        if args.worker is None:
            parser.error("worker requires --worker")
        if args.shards < 1 or not 0 <= args.shard < args.shards:
            parser.error("worker shard must be in [0, shards)")
        train_worker(root, args.worker, args.shard, args.shards)
    else:
        print(json.dumps(verify_run(root)), flush=True)


if __name__ == "__main__":
    main()
