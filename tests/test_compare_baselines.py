from pathlib import Path

import pandas as pd

from mcdrp.results.compare_baselines import compare_baselines


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def base_row(model: str, rmse: float, stage_rows: int = 10) -> dict[str, object]:
    return {
        "model": model,
        "split_name": "random_pair",
        "subset": "test",
        "n_rows": stage_rows,
        "n_cell_lines": 3,
        "n_drugs": 2,
        "rmse": rmse,
        "mae": rmse / 2,
        "pearson": 0.5,
        "spearman": 0.4,
        "r2": 0.1,
    }


def test_compare_baselines_adds_b0_improvements(tmp_path: Path) -> None:
    b0 = tmp_path / "b0.csv"
    b1 = tmp_path / "b1.csv"
    b2 = tmp_path / "b2.csv"

    write_metrics(b0, [base_row("global_mean", 3.0), base_row("drug_mean", 2.0)])
    write_metrics(b1, [base_row("ridge", 1.5)])
    write_metrics(b2, [base_row("xgboost", 1.0)])

    comparison, best = compare_baselines(
        inputs={"B0": str(b0), "B1": str(b1), "B2": str(b2)},
        output=tmp_path / "comparison.csv",
        best_output=tmp_path / "best.csv",
        summary=tmp_path / "summary.json",
    )

    assert len(comparison) == 4
    assert best.iloc[0]["model_id"] == "B2_xgboost"
    assert best.iloc[0]["b0_reference_model"] == "B0_drug_mean"
    assert best.iloc[0]["rmse_improvement_vs_b0"] == 1.0
    assert (tmp_path / "comparison.csv").exists()
    assert (tmp_path / "best.csv").exists()
    assert (tmp_path / "summary.json").exists()


def test_compare_baselines_includes_validation_selected_tuned_rows(
    tmp_path: Path,
) -> None:
    b0 = tmp_path / "b0.csv"
    b3_tuned = tmp_path / "b3_tuning.csv"

    write_metrics(b0, [base_row("drug_mean", 2.0)])
    pd.DataFrame(
        [
            {
                **base_row("mlp", 1.5),
                "subset": "validation",
                "params": '{"hidden_layer_sizes": [256]}',
            },
            {
                **base_row("mlp", 1.2),
                "subset": "validation",
                "params": '{"hidden_layer_sizes": [512, 128]}',
            },
            {
                **base_row("mlp", 1.3),
                "subset": "test",
                "params": '{"hidden_layer_sizes": [512, 128]}',
                "selected_by_validation": True,
            },
        ]
    ).to_csv(b3_tuned, index=False)

    comparison, best = compare_baselines(
        inputs={"B0": str(b0)},
        optional_tuned_inputs={"B3_tuned": str(b3_tuned)},
        output=tmp_path / "comparison.csv",
        best_output=tmp_path / "best.csv",
        summary=tmp_path / "summary.json",
    )

    assert "B3_tuned_mlp" in set(comparison["model_id"])
    assert len(comparison.loc[comparison["stage"].eq("B3_tuned")]) == 2
    assert best.iloc[0]["model_id"] == "B3_tuned_mlp"
    assert best.iloc[0]["source"] == "tuned"
