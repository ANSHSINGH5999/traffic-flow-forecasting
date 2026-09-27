"""One function that turns the raw file into train/val/test samples. Used by the pipeline, reports and audit
so that every consumer sees exactly the same samples."""
import numpy as np

from src.data.loader import load_flow
from src.features.feature_engineering import engineer_features
from src.preprocessing.cleaner import clean_flow
from src.preprocessing.windowing import make_windows, split_bounds


def prepare(cfg, max_sensors=None):
    ds, fc = cfg["dataset"], cfg["forecast"]
    flow = load_flow(cfg["paths"]["raw_data"], ds["flow_feature_index"], max_sensors)
    filled, observed, clean_info = clean_flow(flow, cfg["preprocessing"]["zero_is_missing"],
                                              cfg["preprocessing"]["max_interpolation_gap"])
    bounds = split_bounds(flow.shape[0], cfg["split"]["train"], cfg["split"]["validation"])
    splits = {}
    for name, (s, e) in bounds.items():
        sp = make_windows(filled, observed, s, e, fc["sequence_length"], fc["horizon_steps"])
        sp["F"] = engineer_features(sp["X"], sp["time"], ds["steps_per_day"], ds["start_weekday"])
        splits[name] = sp
    return flow, filled, observed, clean_info, bounds, splits


def train_readings(filled, observed, bounds):
    """Real training-period readings only - the ONLY data the scaler is fitted on."""
    s, e = bounds["train"]
    return np.where(observed[s:e], filled[s:e], np.nan)
