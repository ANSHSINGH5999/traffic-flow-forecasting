"""Model 1 - Historical Average: mean TRAINING flow per (sensor, weekday, 5-minute slot)."""
import json
import warnings
from pathlib import Path

import numpy as np

from src.preprocessing.windowing import time_features


class HistoricalAverage:
    def __init__(self, steps_per_day=288, start_weekday=0):
        self.steps_per_day, self.start_weekday = steps_per_day, start_weekday

    def fit(self, filled, observed, start, end):
        f = np.where(observed[start:end], filled[start:end], np.nan)   # real readings only
        t = np.arange(start, end)
        slot = t % self.steps_per_day
        _, dow = time_features(t, self.steps_per_day, self.start_weekday)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)             # empty cells are handled below
            table = np.stack([np.stack([np.nanmean(f[(dow == d) & (slot == s)], axis=0)
                                        for s in range(self.steps_per_day)]) for d in range(7)])
            slot_mean = np.stack([np.nanmean(f[slot == s], axis=0) for s in range(self.steps_per_day)])
            sensor_mean = np.nanmean(f, axis=0)
        # Fallback when a cell has no training data: (slot, sensor) -> sensor -> global mean
        slot_mean = np.where(np.isnan(slot_mean), sensor_mean, slot_mean)
        slot_mean = np.where(np.isnan(slot_mean), np.nanmean(f), slot_mean)
        self.table = np.where(np.isnan(table), slot_mean, table).astype(np.float32)  # [7, slots, sensors]
        return self

    def predict(self, sensor, target_time):
        _, dow = time_features(np.asarray(target_time), self.steps_per_day, self.start_weekday)
        return self.table[dow.astype(int), np.asarray(target_time) % self.steps_per_day, sensor]

    def save(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "table.npy", self.table)
        (d / "config.json").write_text(json.dumps({
            "steps_per_day": self.steps_per_day, "start_weekday": self.start_weekday,
            "method": "mean training flow per (weekday, 5-min slot, sensor); fallback slot -> sensor -> global",
        }, indent=2))

    @classmethod
    def load(cls, directory):
        d = Path(directory)
        cfg = json.loads((d / "config.json").read_text())
        m = cls(cfg["steps_per_day"], cfg["start_weekday"])
        m.table = np.load(d / "table.npy")
        return m
