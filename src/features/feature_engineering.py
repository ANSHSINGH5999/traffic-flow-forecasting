"""Hand-crafted features for Random Forest, XGBoost and the hybrid's XGBoost stage."""
import numpy as np

from src.preprocessing.windowing import time_features


def feature_names(seq_len=12):
    return ([f"lag_{i}" for i in range(seq_len, 0, -1)]
            + ["rolling_mean_60m", "rolling_std_60m", "recent_mean_15m", "recent_std_15m",
               "last_change", "hour_of_day", "day_of_week"])


# Feature name -> (meaning, why it exists, how it is calculated)
FEATURE_DOCS = {
    "lag_1 ... lag_12": ("Flow 1..12 steps before the forecast origin (lag_1 = time t)",
                         "Recent traffic is the strongest signal of near-future traffic",
                         "The 12 values of the input window"),
    "rolling_mean_60m": ("Average flow over the last hour", "Captures the current traffic level",
                         "Mean of the 12 inputs"),
    "rolling_std_60m": ("Variability over the last hour", "Distinguishes stable from fluctuating traffic",
                        "Std of the 12 inputs"),
    "recent_mean_15m": ("Average flow over the last 15 minutes", "Reacts faster to change than the 60-min mean",
                        "Mean of the last 3 inputs"),
    "recent_std_15m": ("Variability over the last 15 minutes", "Signals sudden instability",
                       "Std of the last 3 inputs"),
    "last_change": ("Most recent 5-minute change", "Gives the direction of the trend",
                    "flow(t) - flow(t-1)"),
    "hour_of_day": ("Hour of the forecast target time (0-23.92)", "Traffic follows a daily cycle (rush hours)",
                    "(target step mod 288) x 5 min / 60"),
    "day_of_week": ("Weekday of the target time (0 = Monday)", "Weekday and weekend patterns differ",
                    "Derived from the dataset start date 2018-01-01"),
}


def engineer_features(X, target_time, steps_per_day=288, start_weekday=0):
    hour, dow = time_features(target_time, steps_per_day, start_weekday)
    return np.column_stack([
        X,
        X.mean(1), X.std(1),
        X[:, -3:].mean(1), X[:, -3:].std(1),
        X[:, -1] - X[:, -2],
        hour, dow,
    ]).astype(np.float32)
