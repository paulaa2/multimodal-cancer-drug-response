"""Parse each real experiment command without loading data or fitting models."""

import argparse
import importlib
import json
from pathlib import Path

import pytest

from mcdrp.experiments.run_experiment import build_command, parse_steps

CONFIGS = sorted((Path(__file__).parents[1] / "configs" / "experiments").glob("*.json"))


class ArgumentsValidated(Exception):
    """Stop a CLI after argument parsing, before executing its pipeline."""


@pytest.mark.parametrize("config", CONFIGS, ids=lambda path: path.stem)
def test_versioned_experiment_arguments(config, monkeypatch):
    steps = parse_steps(json.loads(config.read_text(encoding="utf-8")))
    if any(
        step.module.endswith(("gnn_b4", "gnn_b5", "tune_b4", "tune_b5", "graph_only"))
        for step in steps
    ):
        pytest.importorskip("torch")
    original = argparse.ArgumentParser.parse_args
    for step in steps:
        tokens = build_command(step)[3:]
        module = importlib.import_module(step.module)
        if hasattr(module, "build_parser"):
            module.build_parser().parse_args(tokens)
            continue

        def validate_and_stop(parser, args=None, namespace=None, _tokens=tokens):
            original(parser, _tokens, namespace)
            raise ArgumentsValidated()

        with monkeypatch.context() as context:
            context.setattr(argparse.ArgumentParser, "parse_args", validate_and_stop)
            with pytest.raises(ArgumentsValidated):
                module.main()
