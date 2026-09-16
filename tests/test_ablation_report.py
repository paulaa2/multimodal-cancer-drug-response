import pandas as pd
import pytest

from mcdrp.results.ablation_report import make_delta_table


def test_make_delta_table_uses_reference_within_split_subset() -> None:
    metrics = pd.DataFrame(
        [
            {
                "model": "morgan_expression_pca",
                "split_name": "random_pair",
                "subset": "test",
                "n_features": 10,
                "rmse": 1.0,
                "mae": 0.5,
                "pearson": 0.9,
                "r2": 0.7,
            },
            {
                "model": "morgan_pathways",
                "split_name": "random_pair",
                "subset": "test",
                "n_features": 8,
                "rmse": 0.9,
                "mae": 0.45,
                "pearson": 0.92,
                "r2": 0.75,
            },
        ]
    )

    table = make_delta_table(metrics, reference="morgan_expression_pca")
    improved = table.loc[table["model"].eq("morgan_pathways")].iloc[0]

    assert improved["reference_rmse"] == 1.0
    assert improved["rmse_delta_vs_reference"] == pytest.approx(-0.1)
    assert improved["rmse_delta_pct_vs_reference"] == pytest.approx(-10.0)
