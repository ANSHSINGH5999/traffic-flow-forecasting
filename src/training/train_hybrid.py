"""Train the hybrid: frozen trained LSTM -> hidden features + engineered features -> XGBoost."""
import time

from src.models import xgboost_model
from src.models.hybrid import HybridLSTMXGBoost, fused_features
from src.training.tuning import grid, sensor_subset, val_mae


def train_hybrid(ctx, lstm, lstm_train_time):
    """`lstm` is Model 4, trained on the training set only; here it is frozen and only used to extract features."""
    hcfg, log, dev = ctx.cfg["models"]["hybrid"], ctx.log, ctx.device
    lstm.eval()
    for p in lstm.parameters():
        p.requires_grad_(False)

    t0 = time.perf_counter()
    Z_tr = fused_features(lstm, ctx.scaler, ctx.train["X"], ctx.train["F"], dev)
    Z_va = fused_features(lstm, ctx.scaler, ctx.val["X"], ctx.val["F"], dev)
    extract_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    sub_tr = ctx.train["sensor"] < ctx.tune_sensors
    sub_va = ctx.val["sensor"] < ctx.tune_sensors
    results = []
    for cand in grid(hcfg["xgboost_search"]):
        m = xgboost_model.fit({**hcfg["xgboost_fixed"], **cand}, ctx.seed,
                              Z_tr[sub_tr], ctx.train["y"][sub_tr], Z_va[sub_va], ctx.val["y"][sub_va])
        results.append((val_mae(m.predict(Z_va[sub_va]), ctx.val["y"][sub_va]), cand))
        log.info(f"    Hybrid XGB candidate {cand}  val MAE {results[-1][0]:.3f}")
    best = min(results, key=lambda r: r[0])[1]
    tune_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    xgb = xgboost_model.fit({**hcfg["xgboost_fixed"], **best}, ctx.seed, Z_tr, ctx.train["y"], Z_va, ctx.val["y"])
    xgb_time = time.perf_counter() - t0

    info = {"selected": {**best, "best_iteration": int(xgb.best_iteration)},
            "search": [{"params": c, "val_mae": round(s, 4)} for s, c in results],
            "tuning_time_s": tune_time,
            "time_breakdown_s": {"lstm_training": lstm_train_time, "feature_extraction": extract_time,
                                 "xgboost_training": xgb_time},
            "fused_feature_count": int(Z_tr.shape[1])}
    # Hybrid training cost = training its LSTM + extracting features + training XGBoost
    return HybridLSTMXGBoost(lstm, xgb, ctx.scaler, dev), lstm_train_time + extract_time + xgb_time, info
