import numpy as np

from src.evaluation.metrics import mae, rmse, safe_mape


def test_mae():
    assert mae([1, 2, 3], [2, 2, 5]) == 1.0


def test_rmse():
    assert np.isclose(rmse([0, 0], [3, 4]), np.sqrt(12.5))


def test_safe_mape():
    y = np.array([0.0, 5.0, 100.0, 200.0])
    p = np.array([10.0, 50.0, 110.0, 180.0])
    assert np.isclose(safe_mape(y, p, min_flow=10), 10.0)     # zero and 5 are excluded
    assert np.isnan(safe_mape([0, 1], [1, 2], min_flow=10))
