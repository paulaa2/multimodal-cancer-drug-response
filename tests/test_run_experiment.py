import re

import pytest

from mcdrp.experiments.run_experiment import (
    ExperimentStep,
    build_command,
    parse_steps,
    run_experiment,
    serialize_arg,
)


def test_failed_experiment_keeps_a_reviewable_manifest(tmp_path, monkeypatch):
    import json
    import subprocess

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"name": "failure", "steps": [
        {"name": "first", "module": "mcdrp.models.baseline_b0"}
    ]}))

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        run_experiment(config, run_dir=tmp_path)
    manifest = json.loads((tmp_path / "failure" / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["steps"][0]["returncode"] == 2
    assert len(manifest["source_sha256"]) == 64


def test_serialize_arg_handles_lists_and_flags() -> None:
    assert serialize_arg("splits", ["random_pair", "cold_scaffold"]) == [
        "--splits",
        "random_pair",
        "cold_scaffold",
    ]
    assert serialize_arg("dry-run", True) == ["--dry-run"]
    assert serialize_arg("device", "cuda") == ["--device", "cuda"]
    assert serialize_arg("skip", False) == []


def test_build_command_uses_current_python_executable() -> None:
    step = ExperimentStep(
        name="train",
        module="mcdrp.models.gnn_b5",
        args={"splits": ["random_pair"], "device": "cuda"},
    )

    command = build_command(step)

    assert command[1:4] == ["-m", "mcdrp.models.gnn_b5", "--splits"]
    assert command[-2:] == ["--device", "cuda"]


def test_parse_steps_accepts_cli_token_lists() -> None:
    steps = parse_steps(
        {
            "name": "demo",
            "steps": [
                {
                    "name": "train",
                    "module": "mcdrp.models.pretrained_b7",
                    "args": ["--splits", "random_pair", "--device", "cuda"],
                }
            ],
        }
    )

    command = build_command(steps[0])

    assert command[1:] == [
        "-m",
        "mcdrp.models.pretrained_b7",
        "--splits",
        "random_pair",
        "--device",
        "cuda",
    ]


def test_parse_steps_rejects_src_prefixed_modules() -> None:
    config = {
        "name": "demo",
        "steps": [{"name": "train", "module": "src.mcdrp.models.gnn_b5"}],
    }

    with pytest.raises(ValueError, match=re.escape("mcdrp.models.gnn_b5")):
        parse_steps(config)


def test_parse_steps_rejects_modules_outside_the_package() -> None:
    config = {"name": "demo", "steps": [{"name": "train", "module": "os"}]}

    with pytest.raises(ValueError, match="outside the mcdrp package"):
        parse_steps(config)
