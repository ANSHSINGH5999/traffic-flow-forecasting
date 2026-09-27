"""App tests need trained models: they run against trained_models/ (final) or trained_models_debug/."""
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = next((d for d in ["trained_models", "trained_models_debug"]
               if (ROOT / d / "hybrid_lstm_xgboost" / "xgboost.json").exists()), None)
pytestmark = pytest.mark.skipif(MODELS is None or not (ROOT / "data/raw/pems04.npz").exists(),
                                reason="run the pipeline first (python run_pipeline.py --mode debug)")


@pytest.fixture(scope="module")
def env():
    from app.components.predictor import Forecaster, load_clean_data
    from src.utils import load_config
    cfg = load_config()
    raw, filled, observed = load_clean_data(cfg)
    return cfg, Forecaster(cfg, ROOT / MODELS), filled


def test_models_load(env):
    from app.components.predictor import MODEL_NAMES
    _, fc, _ = env
    for name in MODEL_NAMES:
        assert fc.model(name) is not None


def test_sensor_selection(env):
    from app.components.predictor import ForecastError
    _, fc, filled = env
    X, _, _ = fc.windows(filled, 0, [15000])
    assert X.shape == (1, 12)
    with pytest.raises(ForecastError):
        fc.windows(filled, 10_000, [15000])                    # invalid sensor
    with pytest.raises(ForecastError):
        fc.windows(filled, 0, [2])                             # not enough history


def test_prediction(env):
    from app.components.predictor import MODEL_NAMES, ForecastError
    _, fc, filled = env
    X, tt, valid = fc.windows(filled, 0, [15000, 15001, 15002])
    for name in MODEL_NAMES:
        p = fc.predict(name, X[valid], tt[valid], 0)
        assert np.isfinite(p).all() and (p > -50).all()
    with pytest.raises(ForecastError):
        fc.predict("LSTM", np.full((1, 12), np.nan, dtype=np.float32), tt[:1], 0)   # NaN input


def test_missing_model_dir():
    from app.components.predictor import ForecastError, Forecaster
    from src.utils import load_config
    with pytest.raises(ForecastError, match="No trained models"):
        Forecaster(load_config(), ROOT / "does_not_exist")


def test_streamlit_app_runs():
    from streamlit.testing.v1 import AppTest
    os.environ["TRAFFIC_MODELS_DIR"] = MODELS
    os.environ["TRAFFIC_RESULTS_DIR"] = "results" if MODELS == "trained_models" else "results_debug"
    at = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=180).run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert any("Predicted traffic flow" in m.label for m in at.metric)


def test_inference_script_reproduces_pipeline_predictions():
    sys_path = str(ROOT / "scripts")
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    import test_inference
    results_dir = "results" if MODELS == "trained_models" else "results_debug"
    res = test_inference.run(models_dir=MODELS, results_dir=results_dir, verbose=False)
    assert len(res) == 6 and all(r["ok"] for r in res)
