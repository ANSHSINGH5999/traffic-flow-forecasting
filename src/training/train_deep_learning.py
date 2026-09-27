"""Train LSTM and GRU: small architecture search on validation, then final training with early stopping."""
import time

from src.models import recurrent
from src.training.tuning import grid, sensor_subset


def train_recurrent(ctx, cell):
    mcfg, log = ctx.cfg["models"][cell], ctx.log
    fixed = mcfg["fixed"]
    sc = ctx.scaler

    tr, va = sensor_subset(ctx.train, ctx.tune_sensors), sensor_subset(ctx.val, ctx.tune_sensors)
    t0 = time.perf_counter()
    results = []
    for cand in grid(mcfg["search"]):
        arch = {**cand, "dropout": fixed["dropout"]}
        _, hist = recurrent.train(cell, arch, fixed, sc.transform(tr["X"]), sc.transform(tr["y"]),
                                  sc.transform(va["X"]), va["y"], sc, ctx.device, ctx.seed,
                                  min(ctx.tune_epochs, fixed["max_epochs"]), log=lambda *_: None)
        score = min(h["val_mae"] for h in hist)
        results.append((score, cand))
        log.info(f"    {cell.upper()} candidate {cand}  val MAE {score:.3f}")
    best = min(results, key=lambda r: r[0])[1]
    tune_time = time.perf_counter() - t0

    arch = {**best, "dropout": fixed["dropout"]}
    t0 = time.perf_counter()
    model, history = recurrent.train(cell, arch, fixed, sc.transform(ctx.train["X"]), sc.transform(ctx.train["y"]),
                                     sc.transform(ctx.val["X"]), ctx.val["y"], sc, ctx.device, ctx.seed,
                                     ctx.max_epochs or fixed["max_epochs"], log=log.info)
    info = {"selected": best, "architecture": arch, "training": fixed, "history": history,
            "search": [{"params": c, "val_mae": round(s, 4)} for s, c in results],
            "tuning_time_s": tune_time, "parameters": recurrent.count_parameters(model)}
    return model, time.perf_counter() - t0, info
