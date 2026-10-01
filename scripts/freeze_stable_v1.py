"""Freeze local raw-input and fitted-artifact bundles after full run verification.

The archives are local, ignored outputs. Dataset files are not published to Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from mcdrp.experiments.stable_v1 import check_inputs, sha256, verify_run, write_json


def archive(paths: list[tuple[Path, str]], output: Path) -> dict:
    if output.exists():
        raise ValueError(f"Refusing to overwrite a release archive: {output}")
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for path, relative in paths:
            bundle.write(path, relative)
    with zipfile.ZipFile(output) as bundle:
        for path, relative in paths:
            digest = hashlib.sha256()
            with bundle.open(relative) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != sha256(path):
                raise ValueError(f"Archive round-trip checksum failed: {relative}")
    return {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "members": len(paths),
        "round_trip": "verified",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/stable_v1"))
    parser.add_argument("--inputs-root", type=Path, default=Path("."))
    parser.add_argument("--release-dir", type=Path, default=Path("results/releases/stable_v1"))
    parser.add_argument("--phase", choices=["inputs", "all"], default="all")
    args = parser.parse_args()
    verification = verify_run(args.output) if args.phase == "all" else None
    config = json.loads((args.output / "config.json").read_text())
    inputs = check_inputs(config, args.inputs_root.resolve())
    args.release_dir.mkdir(parents=True, exist_ok=True)
    input_manifest = args.release_dir / "input_manifest.json"
    manifest_data = {
        "protocol": "stable_v1",
        "inputs": config["inputs"],
        "provenance": "Frozen local raw files and external embeddings",
    }
    if input_manifest.exists():
        if json.loads(input_manifest.read_text()) != manifest_data:
            raise ValueError("The existing input manifest describes different frozen files.")
    else:
        write_json(input_manifest, manifest_data)
    input_paths = [(Path(inputs[key]), item["path"]) for key, item in config["inputs"].items()]
    input_paths.append((input_manifest, "input_manifest.json"))
    print("Freezing and verifying the raw-input bundle", flush=True)
    raw_record = args.release_dir / "raw_bundle.json"
    if raw_record.exists():
        raw = json.loads(raw_record.read_text())
        if sha256(Path(raw["path"])) != raw["sha256"]:
            raise ValueError("The frozen raw-input archive has changed.")
    else:
        raw = archive(input_paths, args.release_dir / "stable_v1_inputs.zip")
        write_json(raw_record, raw)
    if args.phase == "inputs":
        print(json.dumps(raw), flush=True)
        return
    print("Freezing and verifying all fitted artifacts", flush=True)
    fitted_paths = [
        (p, p.relative_to(args.output).as_posix())
        for p in sorted(args.output.rglob("*"))
        if p.is_file()
    ]
    fitted = archive(fitted_paths, args.release_dir / "stable_v1_artifacts.zip")
    write_json(
        Path("data/reports/stable_v1_release.json"),
        {
            "status": "verified",
            "protocol": "stable_v1",
            "code_tag": "v0.2.0-stable",
            "run": str(args.output),
            "run_verification": verification,
            "inputs_archive": raw,
            "artifacts_archive": fitted,
            "config_sha256": sha256(args.output / "config.json"),
            "run_manifest_sha256": sha256(args.output / "manifest.json"),
            "note": "Local archives; no remote backup or publication of datasets is implied.",
        },
    )


if __name__ == "__main__":
    main()
