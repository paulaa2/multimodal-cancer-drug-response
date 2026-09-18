from pathlib import Path

import pandas as pd

from mcdrp.models.ablation_b6 import build_parser as build_b6_parser
from mcdrp.models.baseline_b2 import load_selected_xgb_params
from mcdrp.models.tune_b4 import DEFAULT_GRID as B4_GRID
from mcdrp.models.tune_b4 import iter_param_grid, tune_b4_for_split
from mcdrp.models.tune_b5 import DEFAULT_GRID as B5_GRID
from mcdrp.models.tune_b5 import tune_b5_for_split


def _metric_row(model: str, split_name: str, subset: str, rmse: float) -> dict[str, object]:
    return {
        "model": model,
        "split_name": split_name,
        "subset": subset,
        "n_rows": 6,
        "n_cell_lines": 3,
        "n_drugs": 2,
        "rmse": rmse,
        "mae": rmse / 2,
        "pearson": 0.5,
        "spearman": 0.4,
        "r2": 0.1,
        "best_epoch": 2,
        "device": "cpu",
    }


def _frame(subset: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair_id": ["pair_0000001"],
            "subset": [subset],
            "y_true": [1.0],
            "y_pred": [1.1],
        }
    )


def _fake_run(model: str, calls: list[dict[str, object]]):
    def run_split(*_args: object, **kwargs: object):
        calls.append(dict(kwargs))
        learning_rate = float(kwargs["learning_rate"])  # type: ignore[arg-type]
        dropout = float(kwargs["dropout"])  # type: ignore[arg-type]
        val_rmse = dropout + learning_rate
        split_name = str(kwargs["split_name"])
        rows = [
            _metric_row(model, split_name, "validation", val_rmse),
            _metric_row(model, split_name, "test", val_rmse + 0.05),
        ]
        return rows, [_frame("validation"), _frame("test")]

    return run_split


def _common_kwargs() -> dict[str, object]:
    cohort = pd.DataFrame({"pair_id": ["pair_0000001"]})
    assignments = pd.DataFrame({"pair_id": ["pair_0000001"], "split": ["train"]})
    return {
        "cohort": cohort,
        "assignments": assignments,
        "graph_map": {},
        "cell_feature_map": {},
        "split_name": "random_pair",
        "target": "ln_ic50",
        "batch_size": 8,
        "max_epochs": 2,
        "patience": 1,
        "weight_decay": 0.0001,
        "graph_hidden_dim": 8,
        "graph_layers": 1,
        "cell_hidden_dim": 8,
        "fusion_hidden_dim": 8,
        "gradient_clip_norm": 1.0,
        "device": "cpu",
        "random_state": 0,
        "log_every": 1,
    }


def test_iter_param_grid_is_four_candidates() -> None:
    assert len(iter_param_grid(B4_GRID)) == 4
    assert len(iter_param_grid(B5_GRID)) == 4


def test_tune_b4_selects_lowest_validation_rmse_without_retraining() -> None:
    calls: list[dict[str, object]] = []
    rows, frames = tune_b4_for_split(
        **_common_kwargs(),  # type: ignore[arg-type]
        run_split=_fake_run("gnn", calls),
    )

    assert len(calls) == 4
    test_rows = [row for row in rows if row["subset"] == "test"]
    validation_rows = [row for row in rows if row["subset"] == "validation"]
    assert len(validation_rows) == 4
    assert len(test_rows) == 1
    assert test_rows[0]["selected_by_validation"] is True
    assert test_rows[0]["params"] == '{"dropout": 0.1, "learning_rate": 0.0005}'
    assert {tuple(frame["subset"]) for frame in frames} == {("validation",), ("test",)}


def test_tune_b5_selects_lowest_validation_rmse_without_retraining() -> None:
    calls: list[dict[str, object]] = []
    kwargs = _common_kwargs()
    rows, frames = tune_b5_for_split(
        kwargs["cohort"],  # type: ignore[arg-type]
        kwargs["assignments"],  # type: ignore[arg-type]
        {},
        {},
        {},
        split_name="cold_tissue",
        target="ln_ic50",
        batch_size=8,
        max_epochs=2,
        patience=1,
        weight_decay=0.0003,
        graph_hidden_dim=8,
        graph_layers=1,
        graph_output_dim=8,
        fingerprint_hidden_dim=8,
        fingerprint_output_dim=8,
        cell_hidden_dim=8,
        shared_dim=8,
        fusion_hidden_dim=8,
        gradient_clip_norm=1.0,
        device="cpu",
        random_state=0,
        log_every=1,
        run_split=_fake_run("hybrid_gnn", calls),
    )

    assert len(calls) == 4
    test_row = next(row for row in rows if row["subset"] == "test")
    assert test_row["selected_by_validation"] is True
    assert test_row["params"] == '{"dropout": 0.15, "learning_rate": 0.0003}'
    assert test_row["split_name"] == "cold_tissue"
    assert len(frames) == 2


def test_load_selected_xgb_params_prefers_validation_selected_test_rows(
    tmp_path: Path,
) -> None:
    fallback = {"n_estimators": 700, "learning_rate": 0.05, "max_depth": 6}
    path = tmp_path / "b2_tuning.csv"
    pd.DataFrame(
        [
            {
                "split_name": "random_pair",
                "subset": "validation",
                "rmse": 0.9,
                "params": '{"learning_rate": 0.03, "max_depth": 4}',
            },
            {
                "split_name": "random_pair",
                "subset": "test",
                "rmse": 1.1,
                "params": '{"learning_rate": 0.05, "max_depth": 6}',
                "selected_by_validation": True,
            },
            {
                "split_name": "cold_cell",
                "subset": "validation",
                "rmse": 1.4,
                "params": '{"learning_rate": 0.03, "max_depth": 6}',
            },
        ]
    ).to_csv(path, index=False)

    by_split = load_selected_xgb_params(path, fallback=fallback)

    assert by_split["random_pair"]["learning_rate"] == 0.05
    assert by_split["random_pair"]["max_depth"] == 6
    assert by_split["random_pair"]["n_estimators"] == 700
    assert "cold_cell" not in by_split


def test_load_selected_xgb_params_falls_back_to_best_validation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "b2_tuning.csv"
    pd.DataFrame(
        [
            {
                "split_name": "cold_tissue",
                "subset": "validation",
                "rmse": 1.2,
                "params": '{"learning_rate": 0.03, "max_depth": 4}',
            },
            {
                "split_name": "cold_tissue",
                "subset": "validation",
                "rmse": 0.8,
                "params": '{"learning_rate": 0.05, "max_depth": 6}',
            },
        ]
    ).to_csv(path, index=False)

    by_split = load_selected_xgb_params(
        path,
        fallback={"n_estimators": 700, "subsample": 0.8},
    )

    assert by_split["cold_tissue"]["learning_rate"] == 0.05
    assert by_split["cold_tissue"]["max_depth"] == 6
    assert by_split["cold_tissue"]["subsample"] == 0.8


def test_load_selected_xgb_params_missing_file_returns_empty(tmp_path: Path) -> None:
    by_split = load_selected_xgb_params(
        tmp_path / "missing.csv",
        fallback={"n_estimators": 700},
    )
    assert by_split == {}


def test_b6_defaults_share_b2_tuning_hyperparameters() -> None:
    args = build_b6_parser().parse_args([])

    assert args.n_estimators == 700
    assert args.learning_rate == 0.05
    assert args.max_depth == 6
    assert args.b2_tuned == "results/baselines/b2_tuning_metrics.csv"
