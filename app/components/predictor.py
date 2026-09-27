"""Loads the SAVED models from trained_models/ and produces forecasts. Nothing is retrained here."""
from pathlib import Path

import numpy as np

from src import models  # noqa: F401  (XGBoost/torch import order)
from src.data.loader import load_flow
from src.features.feature_engineering import engineer_features
from src.models import random_forest, recurrent, xgboost_model
from src.models.historical_average import HistoricalAverage
from src.models.hybrid import HybridLSTMXGBoost
from src.preprocessing.cleaner import clean_flow
from src.preprocessing.scaler import FlowScaler

MODEL_NAMES = ["Historical Average", "Random Forest", "XGBoost", "LSTM", "GRU", "Hybrid LSTM-XGBoost"]
MODEL_FILES = {
    "Historical Average": ["historical_average/table.npy", "historical_average/config.json"],
    "Random Forest": ["random_forest/model.pkl"],
    "XGBoost": ["xgboost/model.json"],
    "LSTM": ["lstm/model.pt", "lstm/config.json"],
    "GRU": ["gru/model.pt", "gru/config.json"],
    "Hybrid LSTM-XGBoost": ["hybrid_lstm_xgboost/lstm.pt", "hybrid_lstm_xgboost/xgboost.json",
                            "hybrid_lstm_xgboost/config.json"],
}


class ForecastError(Exception):
    """Error with a message that is safe to show to users."""


def load_clean_data(cfg):
    try:
        flow = load_flow(cfg["paths"]["raw_data"], cfg["dataset"]["flow_feature_index"])
    except FileNotFoundError as e:
        raise ForecastError(str(e)) from None
    except Exception as e:
        raise ForecastError(f"The dataset file could not be read ({type(e).__name__}).") from None
    filled, observed, _ = clean_flow(flow, cfg["preprocessing"]["zero_is_missing"],
                                     cfg["preprocessing"]["max_interpolation_gap"])
    return flow, filled, observed


class Forecaster:
    def __init__(self, cfg, models_dir):
        self.cfg, self.dir = cfg, Path(models_dir)
        scaler_path = self.dir / "preprocessing" / "scaler.pkl"
        if not scaler_path.exists():
            raise ForecastError(f"No trained models found in '{self.dir}'. Run `python run_pipeline.py --mode final` first.")
        self.scaler = FlowScaler.load(scaler_path)
        self._models = {}

    def missing_files(self, name):
        return [f for f in MODEL_FILES[name] if not (self.dir / f).exists()]

    def model(self, name):
        if name not in MODEL_FILES:
            raise ForecastError(f"Unknown model '{name}'.")
        if name in self._models:
            return self._models[name]
        missing = self.missing_files(name)
        if missing:
            raise ForecastError(f"{name}: model file(s) missing: {', '.join(missing)}. Re-run the training pipeline.")
        d = self.dir
        try:
            if name == "Historical Average":
                m = HistoricalAverage.load(d / "historical_average")
            elif name == "Random Forest":
                m = random_forest.load(d / "random_forest" / "model.pkl")
            elif name == "XGBoost":
                m = xgboost_model.load(str(d / "xgboost" / "model.json"))
            elif name in ("LSTM", "GRU"):
                m, _ = recurrent.load(d / name.lower())
            else:
                m = HybridLSTMXGBoost.load(d / "hybrid_lstm_xgboost", self.scaler)
        except Exception as e:
            raise ForecastError(f"{name}: saved model could not be loaded - the file may be corrupted "
                                f"({type(e).__name__}). Re-run the training pipeline.") from None
        self._models[name] = m
        return m

    def n_sensors(self):
        return int(self.model("Historical Average").table.shape[2])

    def windows(self, filled, sensor, origins, observed=None):
        """Input windows ending at each origin t (inclusive). Returns X, target_time, valid mask.

        valid = no missing input and (if `observed` is given) a real reading at t, exactly the rule used
        in training (an interpolated value at t would depend on readings after t)."""
        seq, h = self.cfg["forecast"]["sequence_length"], self.cfg["forecast"]["horizon_steps"]
        if not 0 <= sensor < min(filled.shape[1], self.n_sensors()):
            raise ForecastError(f"Sensor {sensor} is not available.")
        origins = np.asarray(origins)
        if origins.min() < seq - 1 or origins.max() + h >= filled.shape[0]:
            raise ForecastError("Not enough historical observations before (or data after) the chosen time.")
        X = np.stack([filled[t - seq + 1:t + 1, sensor] for t in origins]).astype(np.float32)
        valid = ~np.isnan(X).any(axis=1)
        if observed is not None:
            valid &= observed[origins, sensor]
        return X, origins + h, valid

    def predict(self, name, X, target_time, sensor):
        if np.isnan(X).any():
            raise ForecastError("The input window contains missing readings (sensor outage), so no forecast is possible.")
        ds = self.cfg["dataset"]
        m = self.model(name)
        F = engineer_features(X, target_time, ds["steps_per_day"], ds["start_weekday"])
        try:
            if name == "Historical Average":
                p = m.predict(np.full(len(X), sensor), target_time)
            elif name in ("Random Forest", "XGBoost"):
                p = m.predict(F)
            elif name in ("LSTM", "GRU"):
                p = self.scaler.inverse(recurrent.run_batched(m, self.scaler.transform(X)))
            else:
                p = m.predict(X, F)
        except Exception as e:
            raise ForecastError(f"{name}: prediction failed ({type(e).__name__}).") from None
        return np.asarray(p, dtype=np.float64)
