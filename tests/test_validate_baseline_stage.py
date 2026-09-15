import json
from pathlib import Path

import pandas as pd

from mcdrp.results.validate_baseline_stage import (
    validate_comparison_outputs,
    validate_split,
    validate_tuning_table,
)


def cohort() -> pd.DataFrame:
    """Small cohort with two drugs and two cell lines."""

    return pd.DataFrame(
        [
            {
                "pair_id": "p1",
                "depmap_id": "c1",
                "drug_id": "d1",
                "canonical_smiles": "CCO",
                "ln_ic50": 1.0,
            },
            {
                "pair_id": "p2",
                "depmap_id": "c2",
                "drug_id": "d1",
                "canonical_smiles": "CCO",
                "ln_ic50": 2.0,
            },
            {
                "pair_id": "p3",
                "depmap_id": "c1",
                "drug_id": "d2",
                "canonical_smiles": "CC",
                "ln_ic50": 3.0,
            },
        ]
    )


def metric_row(
    stage: str,
    model: str,
    subset: str,
    rmse: float,
    *,
    source: str = "standard",
) -> dict[str, object]:
    """Create one comparison metric row."""

    return {
        "stage": stage,
        "model": model,
        "model_id": f"{stage}_{model}",
        "source": source,
        "params": "{}",
        "split_name": "random_pair",
        "subset": subset,
        "n_rows": 3,
        "n_cell_lines": 2,
        "n_drugs": 2,
        "rmse": rmse,
        "mae": rmse / 2,
        "pearson": 0.5,
        "spearman": 0.4,
        "r2": 0.1,
        "b0_reference_model": "B0_drug_mean",
        "b0_reference_rmse": 2.0,
        "rmse_improvement_vs_b0": 2.0 - rmse,
    }


def test_validate_split_detects_cold_drug_leakage(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    pd.DataFrame(
        [
            {"pair_id": "p1", "split": "train"},
            {"pair_id": "p2", "split": "validation"},
            {"pair_id": "p3", "split": "test"},
        ]
    ).to_csv(split_dir / "cold_drug.csv", index=False)

    result = validate_split(cohort(), split_dir, "cold_drug")

    assert result.status == "fail"
    assert "leaks drugs" in result.message


def test_validate_tuning_table_requires_selected_test_rows(tmp_path: Path) -> None:
    path = tmp_path / "b3_tuning.csv"
    pd.DataFrame(
        [
            {
                "model": "mlp",
                "params": "{}",
                "split_name": "random_pair",
                "subset": "validation",
                "n_rows": 3,
                "n_cell_lines": 2,
                "n_drugs": 2,
                "rmse": 1.0,
                "mae": 0.5,
                "pearson": 0.5,
                "spearman": 0.4,
                "r2": 0.1,
            }
        ]
    ).to_csv(path, index=False)

    result = validate_tuning_table("B3_tuned", path)

    assert result.status == "fail"
    assert "selected test row" in result.message


def test_validate_comparison_outputs_accepts_consistent_tables(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.csv"
    best = tmp_path / "best.csv"
    summary = tmp_path / "summary.json"

    comparison_rows = [
        metric_row("B0", "drug_mean", "test", 2.0),
        metric_row("B2_tuned", "xgboost", "test", 1.0, source="tuned"),
    ]
    pd.DataFrame(comparison_rows).to_csv(comparison, index=False)
    pd.DataFrame([comparison_rows[1]]).to_csv(best, index=False)
    summary.write_text(
        json.dumps({"best_by_split_subset": [{"best_model": "B2_tuned_xgboost"}]}),
        encoding="utf-8",
    )

    results = validate_comparison_outputs(comparison, best, summary)

    assert {result.status for result in results} == {"pass"}
