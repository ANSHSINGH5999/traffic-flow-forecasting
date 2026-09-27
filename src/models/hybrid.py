"""Model 6 (proposed) - Hybrid LSTM-XGBoost.

    12-step sequence -> frozen trained LSTM -> hidden state (temporal features)
                                                   + engineered features
                                                   -> XGBoost -> flow at t+3

The LSTM is a feature extractor (not an averaged second predictor).
"""
import json
import shutil
from pathlib import Path

import numpy as np

from src.models import recurrent, xgboost_model


def fused_features(lstm, scaler, X, F, device="cpu"):
    z = recurrent.run_batched(lstm, scaler.transform(X), fn=lstm.encode, device=device)
    return np.hstack([z, F]).astype(np.float32)


class HybridLSTMXGBoost:
    def __init__(self, lstm, xgb, scaler, device="cpu"):
        self.lstm, self.xgb, self.scaler, self.device = lstm, xgb, scaler, device

    def predict(self, X, F):
        return self.xgb.predict(fused_features(self.lstm, self.scaler, X, F, self.device))

    def save(self, directory, lstm_dir, extra):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(lstm_dir) / "model.pt", d / "lstm.pt")
        lstm_cfg = json.loads((Path(lstm_dir) / "config.json").read_text())
        xgboost_model.save(self.xgb, str(d / "xgboost.json"))
        (d / "config.json").write_text(json.dumps({
            "cell": "lstm", "architecture": lstm_cfg["architecture"],
            "lstm_representation": "final hidden state of the last LSTM layer",
            **extra}, indent=2))

    @classmethod
    def load(cls, directory, scaler, device="cpu"):
        lstm, _ = recurrent.load(directory, filename="lstm.pt", device=device)
        return cls(lstm, xgboost_model.load(str(Path(directory) / "xgboost.json")), scaler, device)
