"""Train Historical Average, Random Forest and XGBoost (with validation-based selection)."""
import time

from src.models import random_forest, xgboost_model
from src.models.historical_average import HistoricalAverage
from src.training.tuning import grid, sensor_subset, val_mae


def train_historical_average(ctx):
    ds = ctx.cfg["dataset"]
    t0 = time.perf_counter()
    model = HistoricalAverage(ds["steps_per_day"], ds["start_weekday"]).fit(
        ctx.filled, ctx.observed, *ctx.bounds["train"])
    return model, time.perf_counter() - t0, {"method": "weekday x 5-min slot x sensor mean"}


def train_random_forest(ctx):
    mcfg, seed, log = ctx.cfg["models"]["random_forest"], ctx.seed, ctx.log
    tr, va = sensor_subset(ctx.train, ctx.tune_sensors), sensor_subset(ctx.val, ctx.tune_sensors)
    t0 = time.perf_counter()
    results = []
    for cand in grid(mcfg["search"]):
        m = random_forest.build({**mcfg["fixed"], **cand}, seed).fit(tr["F"], tr["y"])
        results.append((val_mae(m.predict(va["F"]), va["y"]), cand))
        log.info(f"    RF candidate {cand}  val MAE {results[-1][0]:.3f}")
    best = min(results, key=lambda r: r[0])[1]
    tune_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    model = random_forest.build({**mcfg["fixed"], **best}, seed).fit(ctx.train["F"], ctx.train["y"])
    return model, time.perf_counter() - t0, {"selected": best, "search": _fmt(results), "tuning_time_s": tune_time}


def train_xgboost(ctx):
    mcfg, seed, log = ctx.cfg["models"]["xgboost"], ctx.seed, ctx.log
    tr, va = sensor_subset(ctx.train, ctx.tune_sensors), sensor_subset(ctx.val, ctx.tune_sensors)
    t0 = time.perf_counter()
    results = []
    for cand in grid(mcfg["search"]):
        m = xgboost_model.fit({**mcfg["fixed"], **cand}, seed, tr["F"], tr["y"], va["F"], va["y"])
        results.append((val_mae(m.predict(va["F"]), va["y"]), cand))
        log.info(f"    XGB candidate {cand}  val MAE {results[-1][0]:.3f}")
    best = min(results, key=lambda r: r[0])[1]
    tune_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    model = xgboost_model.fit({**mcfg["fixed"], **best}, seed,
                              ctx.train["F"], ctx.train["y"], ctx.val["F"], ctx.val["y"])
    info = {"selected": {**best, "best_iteration": int(model.best_iteration)},
            "search": _fmt(results), "tuning_time_s": tune_time}
    return model, time.perf_counter() - t0, info


def _fmt(results):
    return [{"params": c, "val_mae": round(s, 4)} for s, c in results]
