"""Standardised evaluation: every model predicts the same locked test samples once."""
import time

import numpy as np
import pandas as pd

from src.evaluation.metrics import all_metrics


def evaluate(predict, split, min_flow):
    t0 = time.perf_counter()
    pred = np.asarray(predict(split), dtype=np.float32)
    seconds = time.perf_counter() - t0
    return pred, all_metrics(split["y"], pred, min_flow), seconds


def timestamps(target_time, start_date, interval_minutes):
    return pd.Timestamp(start_date) + pd.to_timedelta(target_time * interval_minutes, unit="min")


def save_predictions(path, split, pred, start_date, interval_minutes, min_flow):
    y = split["y"].astype(np.float64)
    err = pred - y
    pct = np.where(y >= min_flow, np.abs(err) / np.maximum(y, 1e-9) * 100, np.nan)
    pd.DataFrame({
        "timestamp": timestamps(split["time"], start_date, interval_minutes),
        "sensor_id": split["sensor"],
        "actual_flow": y,
        "predicted_flow": pred,
        "absolute_error": np.abs(err),
        "squared_error": err ** 2,
        "percentage_error": pct,
    }).to_csv(path, index=False, float_format="%.3f")
