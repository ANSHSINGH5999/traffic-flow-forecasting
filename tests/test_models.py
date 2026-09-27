import numpy as np

from src.models import random_forest, recurrent, xgboost_model
from src.models.historical_average import HistoricalAverage
from src.models.hybrid import HybridLSTMXGBoost
from src.preprocessing.scaler import FlowScaler

ARCH = {"hidden_size": 8, "num_layers": 1, "dropout": 0.0}
TRAIN = {"learning_rate": 0.01, "batch_size": 256, "patience": 2}


def _check(pred, n):
    assert pred.shape == (n,) and np.isfinite(pred).all()


def test_historical_average_prediction(splits, tmp_path):
    sp, filled, observed = splits
    m = HistoricalAverage().fit(filled, observed, 0, len(filled) * 7 // 10)
    _check(m.predict(sp["test"]["sensor"], sp["test"]["time"]), len(sp["test"]["y"]))
    m.save(tmp_path / "ha")
    np.testing.assert_array_equal(HistoricalAverage.load(tmp_path / "ha").table, m.table)


def test_random_forest_prediction(splits, tmp_path):
    sp, _, _ = splits
    m = random_forest.build({"n_estimators": 5, "max_depth": 5}, 0).fit(sp["train"]["F"], sp["train"]["y"])
    random_forest.save(m, tmp_path / "rf.pkl")
    _check(random_forest.load(tmp_path / "rf.pkl").predict(sp["test"]["F"]), len(sp["test"]["y"]))


def test_xgboost_prediction(splits, tmp_path):
    sp, _, _ = splits
    m = xgboost_model.fit({"n_estimators": 20, "early_stopping_rounds": 5}, 0,
                          sp["train"]["F"], sp["train"]["y"], sp["val"]["F"], sp["val"]["y"])
    xgboost_model.save(m, str(tmp_path / "x.json"))
    _check(xgboost_model.load(str(tmp_path / "x.json")).predict(sp["test"]["F"]), len(sp["test"]["y"]))


def _train_rnn(cell, sp, scaler):
    return recurrent.train(cell, ARCH, TRAIN, scaler.transform(sp["train"]["X"]), scaler.transform(sp["train"]["y"]),
                           scaler.transform(sp["val"]["X"]), sp["val"]["y"], scaler, "cpu", 0, 2, log=lambda *_: None)


def _rnn_roundtrip(cell, splits, tmp_path):
    sp, filled, _ = splits
    scaler = FlowScaler().fit(filled)
    model, hist = _train_rnn(cell, sp, scaler)
    assert len(hist) == 2
    recurrent.save(model, ARCH, {}, tmp_path / cell)
    loaded, cfg = recurrent.load(tmp_path / cell)
    pred = scaler.inverse(recurrent.run_batched(loaded, scaler.transform(sp["test"]["X"])))
    _check(pred, len(sp["test"]["y"]))
    assert cfg["cell"] == cell


def test_lstm_prediction(splits, tmp_path):
    _rnn_roundtrip("lstm", splits, tmp_path)


def test_gru_prediction(splits, tmp_path):
    _rnn_roundtrip("gru", splits, tmp_path)


def test_hybrid_prediction(splits, tmp_path):
    from src.models.hybrid import fused_features
    sp, filled, _ = splits
    scaler = FlowScaler().fit(filled)
    lstm, _ = _train_rnn("lstm", sp, scaler)
    Z_tr = fused_features(lstm, scaler, sp["train"]["X"], sp["train"]["F"])
    Z_va = fused_features(lstm, scaler, sp["val"]["X"], sp["val"]["F"])
    assert Z_tr.shape[1] == ARCH["hidden_size"] + sp["train"]["F"].shape[1]   # LSTM features + engineered
    xgb = xgboost_model.fit({"n_estimators": 20, "early_stopping_rounds": 5}, 0, Z_tr, sp["train"]["y"], Z_va, sp["val"]["y"])
    recurrent.save(lstm, ARCH, {}, tmp_path / "lstm")
    HybridLSTMXGBoost(lstm, xgb, scaler).save(tmp_path / "hyb", tmp_path / "lstm", {})
    loaded = HybridLSTMXGBoost.load(tmp_path / "hyb", scaler)
    _check(loaded.predict(sp["test"]["X"], sp["test"]["F"]), len(sp["test"]["y"]))
