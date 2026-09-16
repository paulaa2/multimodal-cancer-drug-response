"""Run reproducible experiment pipelines from JSON configs."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentStep:
    """One command-line module invocation."""

    name: str
    module: str
    args: dict[str, Any]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment config."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if "name" not in config:
        raise ValueError(f"{config_path} is missing required field: name")
    if "steps" not in config or not isinstance(config["steps"], list):
        raise ValueError(f"{config_path} must define a list of steps.")
    return config


def parse_steps(config: dict[str, Any]) -> list[ExperimentStep]:
    """Parse config steps into typed objects."""

    steps: list[ExperimentStep] = []
    for index, raw in enumerate(config["steps"], start=1):
        if "module" not in raw:
            raise ValueError(f"Step {index} is missing required field: module")
        steps.append(
            ExperimentStep(
                name=str(raw.get("name", f"step_{index}")),
                module=str(raw["module"]),
                args=dict(raw.get("args", {})),
            )
        )
    return steps


def serialize_arg(flag: str, value: Any) -> list[str]:
    """Convert a JSON argument value into CLI tokens."""

    if value is None or value is False:
        return []
    normalized_flag = flag if flag.startswith("--") else f"--{flag}"
    if value is True:
        return [normalized_flag]
    if isinstance(value, list):
        return [normalized_flag, *[str(item) for item in value]]
    return [normalized_flag, str(value)]


def build_command(step: ExperimentStep) -> list[str]:
    """Build the Python module command for one step."""

    command = [sys.executable, "-m", step.module]
    for flag, value in step.args.items():
        command.extend(serialize_arg(flag, value))
    return command


def shell_join(command: list[str]) -> str:
    """Return a readable shell command."""

    return " ".join(shlex.quote(part) for part in command)


def run_experiment(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    run_dir: str | Path = "results/experiments",
) -> dict[str, Any]:
    """Run all steps in an experiment config."""

    config = load_config(config_path)
    steps = parse_steps(config)
    experiment_dir = Path(run_dir) / str(config["name"])
    experiment_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "name": config["name"],
        "description": config.get("description", ""),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "dry_run": dry_run,
        "steps": [],
    }

    for step in steps:
        command = build_command(step)
        entry = {
            "name": step.name,
            "module": step.module,
            "command": command,
            "command_text": shell_join(command),
        }
        print(f"[{step.name}] {entry['command_text']}")
        if not dry_run:
            completed = subprocess.run(command, check=True)
            entry["returncode"] = completed.returncode
        manifest["steps"].append(entry)

    manifest_path = experiment_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Wrote experiment manifest to {manifest_path}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description="Run an mcdrp experiment config.")
    parser.add_argument("config", help="Path to a JSON experiment config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write a manifest without executing steps.",
    )
    parser.add_argument(
        "--run-dir",
        default="results/experiments",
        help="Directory for experiment manifests.",
    )
    return parser


def main() -> None:
    """Command-line entry point."""

    args = build_parser().parse_args()
    run_experiment(args.config, dry_run=args.dry_run, run_dir=args.run_dir)


if __name__ == "__main__":
    main()
