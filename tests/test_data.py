import numpy as np
import pytest

from src.data.loader import load_flow
from src.preprocessing.cleaner import clean_flow


def test_dataset_loader(synthetic_npz, synthetic_flow):
    flow = load_flow(synthetic_npz)
    assert flow.dtype == np.float32
    np.testing.assert_allclose(flow, synthetic_flow)


def test_dataset_loader_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="pems04.npz"):
        load_flow(tmp_path / "nope.npz")


def test_dataset_shape(synthetic_npz):
    assert load_flow(synthetic_npz).shape == (14 * 288, 4)
    assert load_flow(synthetic_npz, max_sensors=2).shape == (14 * 288, 2)


def test_missing_value_detection(synthetic_flow):
    filled, observed, info = clean_flow(synthetic_flow, zero_is_missing=True, max_gap=12)
    assert info["missing_readings"] == 3 + 40
    assert not np.isnan(filled[100:103, 0]).any()          # short gap interpolated
    assert np.isnan(filled[2000:2040, 1]).all()            # long gap (40 > 12) left missing
    assert not observed[100:103, 0].any()                  # but still marked as not observed


def test_real_dataset_matches_documentation():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "data/raw/pems04.npz"
    if not path.exists():
        pytest.skip("real PeMSD4 not downloaded")
    assert load_flow(path).shape == (16992, 307)


def test_long_gap_is_not_partially_filled():
    flow = np.full((40, 1), 100.0, dtype=np.float32)
    flow[5:25, 0] = 0                                      # 20-step gap > max_gap 12
    flow[30:32, 0] = 0                                     # 2-step gap
    filled, _, _ = clean_flow(flow, max_gap=12)
    assert np.isnan(filled[5:25, 0]).all()
    assert not np.isnan(filled[30:32, 0]).any()
