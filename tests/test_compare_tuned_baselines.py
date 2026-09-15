from pathlib import Path

import pandas as pd

from mcdrp.results.compare_tuned_baselines import compare_tuned_baselines


def row(
    model: str,
    split: str,
    subset: str,
    rmse: float,
    params: str = "{}",
    selected: bool | None = None,
) -> dict[str, object]:
    output = {
        "model": model,
        "params": params,
        "split_name": split,
        "subset": subset,
        "n_rows": 10,
        "n_cell_lines": 3,
        "n_drugs": 2,
        "rmse": rmse,
        "mae": rmse / 2,
        "pearson": 0.5,
        "spearman": 0.4,
        "r2": 0.1,
    }
    if selected is not None:
        output["selected_by_validation"] = selected
    return output


def test_compare_tuned_baselines_keeps_selected_test_rows(tmp_path: Path) -> None:
    b0 = tmp_path / "b0.csv"
    b1 = tmp_path / "b1.csv"

    pd.DataFrame(
        [
            row("global_mean", "random_pair", "test", 3.0),
            row("drug_mean", "random_pair", "test", 2.0),
            row("global_mean", "random_pair", "validation", 3.0),
            row("drug_mean", "random_pair", "validation", 2.0),
        ]
    ).to_csv(b0, index=False)

    pd.DataFrame(
        [
            row("ridge", "random_pair", "validation", 1.6, '{"alpha": 1.0}'),
            row("ridge", "random_pair", "validation", 1.4, '{"alpha": 10.0}'),
            row("ridge", "random_pair", "test", 1.5, '{"alpha": 10.0}', True),
        ]
    ).to_csv(b1, index=False)

    comparison, best = compare_tuned_baselines(
        b0=b0,
        b1_tuned=b1,
        b2_tuned=tmp_path / "missing_b2.csv",
        output=tmp_path / "comparison.csv",
        best_output=tmp_path / "best.csv",
        summary=tmp_path / "summary.json",
    )

    assert "B1_tuned_ridge" in set(comparison["model_id"])
    assert best.loc[best["subset"].eq("test"), "model_id"].iloc[0] == "B1_tuned_ridge"
    assert best.loc[best["subset"].eq("test"), "rmse"].iloc[0] == 1.5
