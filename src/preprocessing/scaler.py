"""Standard scaler for the neural networks. Fitted on TRAINING data only."""
import pickle

import numpy as np


class FlowScaler:
    def fit(self, values):
        values = values[np.isfinite(values)]
        self.mean, self.std = float(values.mean()), float(values.std())
        return self

    def transform(self, a):
        return ((a - self.mean) / self.std).astype(np.float32)

    def inverse(self, a):
        return a * self.std + self.mean

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"mean": self.mean, "std": self.std}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        s = cls()
        s.mean, s.std = d["mean"], d["std"]
        return s
