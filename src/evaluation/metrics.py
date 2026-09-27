"""MAE, RMSE and safe MAPE (percentage error only where actual flow is meaningfully above zero)."""
import numpy as np


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def safe_mape(y, p, min_flow=10):
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    mask = np.abs(y) >= min_flow
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y[mask] - p[mask]) / y[mask])) * 100)


def all_metrics(y, p, min_flow=10):
    return {"MAE": mae(y, p), "RMSE": rmse(y, p), "MAPE": safe_mape(y, p, min_flow)}
