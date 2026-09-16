import pandas as pd

from mcdrp.results.cold_start_report import make_performance_drop_table


def test_make_performance_drop_table_compares_against_random_pair() -> None:
    metrics = pd.DataFrame(
        [
            {
                "model_id": "B2_xgboost",
                "stage": "B2",
                "source": "standard",
                "split_name": "random_pair",
                "subset": "test",
                "n_rows": 10,
                "n_cell_lines": 2,
                "n_drugs": 3,
                "rmse": 1.0,
                "mae": 0.5,
                "pearson": 0.9,
                "spearman": 0.8,
                "r2": 0.7,
                "rmse_improvement_pct_vs_b0": 20.0,
            },
            {
                "model_id": "B2_xgboost",
                "stage": "B2",
                "source": "standard",
                "split_name": "cold_drug",
                "subset": "test",
                "n_rows": 10,
                "n_cell_lines": 2,
                "n_drugs": 3,
                "rmse": 1.5,
                "mae": 0.7,
                "pearson": 0.6,
                "spearman": 0.5,
                "r2": 0.2,
                "rmse_improvement_pct_vs_b0": 10.0,
            },
        ]
    )

    table = make_performance_drop_table(metrics, "test")

    assert len(table) == 1
    assert table.iloc[0]["split_name"] == "cold_drug"
    assert table.iloc[0]["rmse_drop_vs_random_pair"] == 0.5
    assert table.iloc[0]["rmse_drop_pct_vs_random_pair"] == 50.0
