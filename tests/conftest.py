"""Synthetic data is used ONLY to test code paths - never for research results."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import models  # noqa: E402,F401  (XGBoost/torch import order)


@pytest.fixture
def synthetic_flow():
    """14 days x 4 sensors of daily-cycle traffic with a few zero (missing) readings."""
    rng = np.random.default_rng(0)
    t = np.arange(14 * 288)
    base = 200 + 150 * np.sin(2 * np.pi * (t % 288) / 288 - np.pi / 2)
    flow = np.stack([base * (1 + 0.2 * s) + rng.normal(0, 10, t.size) for s in range(4)], axis=1)
    flow = np.clip(flow, 1, None).astype(np.float32)
    flow[100:103, 0] = 0          # short gap -> interpolated
    flow[2000:2040, 1] = 0        # long gap -> left missing
    return flow


@pytest.fixture
def synthetic_npz(tmp_path, synthetic_flow):
    data = np.stack([synthetic_flow, synthetic_flow / 1000, np.full_like(synthetic_flow, 60)], axis=2)
    path = tmp_path / "pems04.npz"
    np.savez(path, data=data)
    return path


@pytest.fixture
def splits(synthetic_flow):
    from src.features.feature_engineering import engineer_features
    from src.preprocessing.cleaner import clean_flow
    from src.preprocessing.windowing import make_windows, split_bounds
    filled, observed, _ = clean_flow(synthetic_flow)
    out = {}
    for name, (s, e) in split_bounds(len(filled)).items():
        sp = make_windows(filled, observed, s, e)
        sp["F"] = engineer_features(sp["X"], sp["time"])
        out[name] = sp
    return out, filled, observed
