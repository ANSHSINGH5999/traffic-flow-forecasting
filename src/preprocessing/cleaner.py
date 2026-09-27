"""Missing-value handling: zeros -> missing, gaps of <= max_gap steps interpolated in time, longer gaps left as NaN."""
import numpy as np
import pandas as pd


def gap_lengths(missing):
    """For a 1-D bool array, the length of the missing run each element belongs to (0 where not missing)."""
    out = np.zeros(len(missing), dtype=np.int64)
    edges = np.flatnonzero(np.diff(np.r_[0, missing.astype(np.int8), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        out[s:e] = e - s
    return out


def clean_flow(flow, zero_is_missing=True, max_gap=12):
    """Return (filled[T, N] with NaN in long gaps, observed[T, N] = reading was real)."""
    observed = (flow > 0) if zero_is_missing else ~np.isnan(flow)
    observed &= np.isfinite(flow)
    filled = (pd.DataFrame(np.where(observed, flow, np.nan))
              .interpolate(limit_area="inside").to_numpy(np.float32, copy=True))  # writable copy (pandas CoW)
    # pandas' `limit` would still fill the first steps of a long gap, so long gaps are re-blanked explicitly
    long_gap = np.column_stack([gap_lengths(~observed[:, i]) > max_gap for i in range(flow.shape[1])])
    filled[long_gap] = np.nan
    info = {
        "missing_readings": int((~observed).sum()),
        "missing_pct": float((~observed).mean() * 100),
        "interpolated_readings": int((~observed).sum() - np.isnan(filled).sum()),
        "unfilled_long_gap_readings": int(np.isnan(filled).sum()),
    }
    return filled, observed, info
