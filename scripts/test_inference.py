"""Tests the PRODUCTION inference path (the same code the Streamlit app uses):
sensor + time -> previous 60 min from the dataset -> saved preprocessing -> saved model -> 15-min forecast.
It also checks that the saved model reproduces the prediction the pipeline stored for the same test sample.

    python scripts/test_inference.py
    python scripts/test_inference.py --sensor 12 --time "2018-02-21 17:30" --model "LSTM"
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.components.predictor import MODEL_NAMES, ForecastError, Forecaster, load_clean_data  # noqa: E402
from src.reporting import MODEL_FILES  # noqa: E402
from src.utils import load_config  # noqa: E402

TOLERANCE = 0.1   # vehicles; GPU-trained nets evaluated on CPU can differ in the last float digits


def step_of(ts, cfg):
    return int((pd.Timestamp(ts) - pd.Timestamp(cfg["dataset"]["start_date"])) / pd.Timedelta(minutes=cfg["dataset"]["interval_minutes"]))


def when(step, cfg):
    return pd.Timestamp(cfg["dataset"]["start_date"]) + pd.Timedelta(minutes=int(step) * cfg["dataset"]["interval_minutes"])


def default_origin(fc, filled, observed, sensor, cfg):
    """First test-period step at/after 08:00 on the second test day with a usable window and a real target."""
    T, spd, h = filled.shape[0], cfg["dataset"]["steps_per_day"], cfg["forecast"]["horizon_steps"]
    test_start = int(T * (cfg["split"]["train"] + cfg["split"]["validation"]))
    start = (test_start // spd + 1) * spd + 8 * 12
    for t in range(start, T - h):
        _, _, valid = fc.windows(filled, sensor, [t], observed)
        if valid[0] and observed[t + h, sensor]:
            return t
    raise ForecastError(f"No usable forecast time for sensor {sensor} in the test period.")


def run(sensor=0, time=None, models=MODEL_NAMES, models_dir="trained_models", results_dir="results", verbose=True):
    cfg = load_config()
    raw, filled, observed = load_clean_data(cfg)
    fc = Forecaster(cfg, ROOT / models_dir)
    h, seq = cfg["forecast"]["horizon_steps"], cfg["forecast"]["sequence_length"]
    origin = step_of(time, cfg) if time else default_origin(fc, filled, observed, sensor, cfg)
    X, tt, valid = fc.windows(filled, sensor, [origin], observed)
    if not valid[0]:
        raise ForecastError("The 60-minute input window is incomplete or its last reading is missing at this time.")
    target = int(tt[0])
    actual = float(raw[target, sensor]) if observed[target, sensor] else None

    results = []
    for name in models:
        pred = float(fc.predict(name, X, tt, sensor)[0])
        stored = None
        csv = ROOT / results_dir / "predictions" / f"{MODEL_FILES[name]}.csv"
        if csv.exists():
            df = pd.read_csv(csv, usecols=["timestamp", "sensor_id", "predicted_flow"])
            row = df[(df["sensor_id"] == sensor) & (df["timestamp"] == str(when(target, cfg)))]
            stored = float(row["predicted_flow"].iloc[0]) if len(row) else None
        ok = np.isfinite(pred) and (stored is None or abs(pred - stored) <= TOLERANCE)
        results.append({"model": name, "pred": pred, "stored": stored, "ok": ok})
        if verbose:
            print("=" * 44 + "\nINFERENCE TEST\n" + "=" * 44)
            print(f"Sensor:              {sensor}")
            print(f"Forecast timestamp:  {when(target, cfg):%Y-%m-%d %H:%M}")
            print(f"Historical window:   {when(origin - seq + 1, cfg):%Y-%m-%d %H:%M} -> {when(origin, cfg):%H:%M} ({seq} readings)")
            print(f"Input readings:      {', '.join(f'{v:.0f}' for v in X[0])}")
            print(f"Forecast horizon:    {cfg['forecast']['horizon_minutes']} minutes")
            print(f"Model:               {name}")
            print(f"Predicted traffic:   {pred:.2f} vehicles / 5 min")
            print(f"Actual traffic:      {actual:.0f}" if actual is not None else "Actual traffic:      not recorded")
            if actual is not None:
                print(f"Absolute error:      {abs(pred - actual):.2f}")
            if stored is not None:
                print(f"Pipeline prediction: {stored:.2f}  (saved-model reproduction diff {abs(pred - stored):.4f})")
            else:
                print("Pipeline prediction: sample not in the stored test predictions")
            print(f"STATUS: {'PASS' if ok else 'FAIL'}\n")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", type=int, default=0)
    ap.add_argument("--time", help='forecast origin (last known reading), e.g. "2018-02-21 08:00"')
    ap.add_argument("--model", choices=MODEL_NAMES, help="default: all six")
    ap.add_argument("--models-dir", default="trained_models")
    ap.add_argument("--results-dir", default="results")
    a = ap.parse_args()
    try:
        res = run(a.sensor, a.time, [a.model] if a.model else MODEL_NAMES, a.models_dir, a.results_dir)
    except ForecastError as e:
        print(f"STATUS: FAIL - {e}")
        sys.exit(1)
    passed = sum(r["ok"] for r in res)
    print(f"OVERALL: {passed}/{len(res)} models PASS")
    sys.exit(0 if passed == len(res) else 1)


if __name__ == "__main__":
    main()
