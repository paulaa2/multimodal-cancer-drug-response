from mcdrp.experiments.run_experiment import (
    ExperimentStep,
    build_command,
    serialize_arg,
)


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
        module="src.mcdrp.models.gnn_b5",
        args={"splits": ["random_pair"], "device": "cuda"},
    )

    command = build_command(step)

    assert command[1:4] == ["-m", "src.mcdrp.models.gnn_b5", "--splits"]
    assert command[-2:] == ["--device", "cuda"]
