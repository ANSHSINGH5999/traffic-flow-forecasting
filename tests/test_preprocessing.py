import numpy as np

from src.preprocessing.scaler import FlowScaler
from src.preprocessing.windowing import make_windows, split_bounds


def test_scaler(tmp_path):
    train = np.array([10.0, 20.0, 30.0, np.nan])
    s = FlowScaler().fit(train)
    assert abs(s.mean - 20) < 1e-6
    z = s.transform(np.array([20.0, 30.0]))
    assert abs(z[0]) < 1e-6
    np.testing.assert_allclose(s.inverse(z), [20, 30], rtol=1e-5)
    s.save(tmp_path / "scaler.pkl")
    assert FlowScaler.load(tmp_path / "scaler.pkl").std == s.std


def test_window_generation():
    flow = np.arange(40, dtype=np.float32).reshape(20, 2)       # value = 2*t + sensor
    obs = np.ones_like(flow, dtype=bool)
    w = make_windows(flow, obs, 0, 20, seq_len=12, horizon=3)
    assert w["X"].shape == (2 * (20 - 12 - 3 + 1), 12)
    i = 0                                                          # first sample: sensor 0, inputs t=0..11
    np.testing.assert_array_equal(w["X"][i], flow[0:12, 0])
    assert w["y"][i] == flow[14, 0] and w["time"][i] == 14         # target = t + 3 = step 14


def test_targets_must_be_real_readings():
    flow = np.ones((20, 1), dtype=np.float32)
    obs = np.ones_like(flow, dtype=bool)
    obs[14, 0] = False                                             # interpolated value at step 14
    w = make_windows(flow, obs, 0, 20)
    assert 14 not in w["time"]


def test_chronological_split(splits):
    sp, _, _ = splits
    b = split_bounds(14 * 288)
    assert b["train"][1] == b["val"][0] and b["val"][1] == b["test"][0]
    assert sp["train"]["time"].max() < sp["val"]["time"].min() <= sp["val"]["time"].max() < sp["test"]["time"].min()
    # a training window never reaches into the validation period
    assert sp["train"]["time"].max() < b["train"][1]


def test_last_input_must_be_real_reading():
    """An interpolated value at time t is computed from readings after t -> such windows are dropped."""
    flow = np.ones((20, 1), dtype=np.float32)
    obs = np.ones_like(flow, dtype=bool)
    obs[11, 0] = False                                             # reading at t = 11 was interpolated
    w = make_windows(flow, obs, 0, 20)
    assert 14 not in w["time"]                                     # origin 11 -> target 14 dropped
    obs[5, 0] = False                                              # a gap INSIDE the window is fine
    obs[11, 0] = True
    assert 14 in make_windows(flow, obs, 0, 20)["time"]
