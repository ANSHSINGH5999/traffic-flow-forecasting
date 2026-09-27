"""Chronological split + sliding windows (12 inputs -> value 3 steps ahead)."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def split_bounds(T, train=0.70, validation=0.15):
    a, b = int(T * train), int(T * (train + validation))
    return {"train": (0, a), "val": (a, b), "test": (b, T)}


def make_windows(filled, observed, start, end, seq_len=12, horizon=3):
    """Samples lying entirely inside [start, end). Input = steps t-seq_len+1..t, target = t+horizon.

    A sample is kept only if
      * every input value exists,
      * the reading at time t (last input) is REAL - an interpolated value at t would have been computed
        from readings after t (future leakage); with a real reading at t every gap inside the window is
        closed by time t, so interpolation only ever uses readings <= t,
      * the target is a REAL reading (interpolated values may be inputs, never targets).
    Returns dict X, y, sensor, time.
    """
    seg, obs = filled[start:end], observed[start:end]
    T, N = seg.shape
    n = T - seq_len - horizon + 1
    if n <= 0:
        raise ValueError(f"Segment of {T} steps is too short for window {seq_len} + horizon {horizon}")

    X = sliding_window_view(seg, seq_len, axis=0)[:n].reshape(-1, seq_len)
    y = seg[seq_len + horizon - 1:].reshape(-1)
    y_real = obs[seq_len + horizon - 1:].reshape(-1)
    last_real = obs[seq_len - 1:seq_len - 1 + n].reshape(-1)
    target_time = np.repeat(np.arange(n) + start + seq_len + horizon - 1, N)
    sensor = np.tile(np.arange(N), n)

    keep = ~np.isnan(X).any(axis=1) & last_real & y_real
    return {"X": X[keep], "y": y[keep], "sensor": sensor[keep], "time": target_time[keep]}


def time_features(target_time, steps_per_day, start_weekday):
    hour = (target_time % steps_per_day) * (24 / steps_per_day)
    dow = (target_time // steps_per_day + start_weekday) % 7
    return hour.astype(np.float32), dow.astype(np.float32)
