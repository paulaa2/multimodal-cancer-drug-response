import numpy as np

from mcdrp.metrics import regression_metrics


def test_regression_metrics_for_perfect_prediction() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    metrics = regression_metrics(y_true, y_pred)

    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["pearson"] == 1.0
    assert metrics["spearman"] == 1.0
    assert metrics["r2"] == 1.0
