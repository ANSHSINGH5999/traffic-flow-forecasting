"""Error analysis by traffic regime. All thresholds are data-driven and computed from TRAINING targets."""
import numpy as np
import pandas as pd

from src.evaluation.metrics import mae, rmse

REGIME_ORDER = ["low", "normal", "high", "peak", "sudden_change"]


def regime_thresholds(train):
    y = train["y"]
    change = np.abs(train["y"] - train["X"][:, -1])
    return {
        "low_below": float(np.percentile(y, 25)),
        "high_from": float(np.percentile(y, 75)),
        "peak_from": float(np.percentile(y, 90)),
        "sudden_change_from": float(np.percentile(change, 95)),
    }


def regime_definitions(th):
    return {
        "low": f"actual flow < {th['low_below']:.0f} (training 25th percentile)",
        "normal": f"{th['low_below']:.0f} <= actual flow < {th['high_from']:.0f} (training 25th-75th percentile)",
        "high": f"{th['high_from']:.0f} <= actual flow < {th['peak_from']:.0f} (training 75th-90th percentile)",
        "peak": f"actual flow >= {th['peak_from']:.0f} (training 90th percentile)",
        "sudden_change": (f"|flow(t+3) - flow(t)| >= {th['sudden_change_from']:.0f} "
                          f"(training 95th percentile of 15-min change); overlaps the flow-level regimes"),
    }


def regime_masks(split, th):
    y = split["y"]
    return {
        "low": y < th["low_below"],
        "normal": (y >= th["low_below"]) & (y < th["high_from"]),
        "high": (y >= th["high_from"]) & (y < th["peak_from"]),
        "peak": y >= th["peak_from"],
        "sudden_change": np.abs(y - split["X"][:, -1]) >= th["sudden_change_from"],
    }


def analyse(preds, test, th):
    masks = regime_masks(test, th)
    rows = []
    for name, p in preds.items():
        for regime in REGIME_ORDER:
            m = masks[regime]
            rows.append({"Model": name, "Regime": regime, "Samples": int(m.sum()),
                         "Share (%)": float(m.mean() * 100),
                         "MAE": mae(test["y"][m], p[m]), "RMSE": rmse(test["y"][m], p[m]),
                         "Mean error (bias)": float(np.mean(p[m] - test["y"][m]))})
    return pd.DataFrame(rows)


def error_by_hour(preds, test, steps_per_day):
    hour = (test["time"] % steps_per_day) * 24 // steps_per_day
    return pd.DataFrame({name: [float(np.abs(p - test["y"])[hour == h].mean()) for h in range(24)]
                         for name, p in preds.items()})
