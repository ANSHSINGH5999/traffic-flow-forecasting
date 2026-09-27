"""STGCN as an ADDITIONAL (7th) model under the existing protocol. Reads existing files, never writes to them.

    python -m experiments.stgcn.run_stgcn --mode debug   # 2 epochs, checks the code
    python -m experiments.stgcn.run_stgcn --mode final

Protocol (identical to the six-model experiment):
  * same PeMSD4 flow, cleaning, chronological split, 12-step input, t+3 target
  * scored ONLY on the existing locked test samples (same (time, sensor) pairs, same order) - verified at runtime
  * scaler statistics = training readings only; selection + early stopping on validation only; test used once
Graph-specific input rule (the only new design choice):
  a neighbour's reading inside the window is used only if it is real, or an interpolated value whose gap had already
  closed by the forecast origin t (so no value depends on readings after t); anything else is set to 0 (= training
  mean after scaling) and flagged in a second input channel (availability mask).
"""
import argparse
import copy
import datetime as dt
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import models  # noqa: F401  (XGBoost/torch import order)
import torch

from experiments.stgcn.stgcn import STGCN, cheb_polynomials, distance_graph
from src.evaluation import error_analysis as ea
from src.evaluation.evaluator import save_predictions
from src.evaluation.metrics import all_metrics
from src.preprocessing.prepare import prepare, train_readings
from src.preprocessing.scaler import FlowScaler
from src.utils import environment_info, get_device, load_config, set_seed
from src.visualization import plots

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "stgcn"
GRAPH = ROOT / "data" / "raw" / "pems04_distance.csv"
SEARCH = {"channels": [[32, 8, 32], [64, 16, 64]], "cheb_k": [2, 3]}
FIXED = {"kt": 3, "dropout": 0.0, "learning_rate": 1e-3, "batch_size": 32, "max_epochs": 50, "patience": 8}
TUNE_EPOCHS = 8   # same per-candidate budget as the LSTM/GRU search in config.yaml


class Snapshots:
    """Builds graph snapshots [B, 2, 12, N] for forecast origins t (inputs t-11..t, target t+3)."""
    def __init__(self, filled, observed, scaler, seq, h):
        T, N = filled.shape
        self.seq, self.h = seq, h
        self.finite = np.isfinite(filled)
        self.scaled = np.where(self.finite, scaler.transform(np.nan_to_num(filled)), 0).astype(np.float32)
        idx = np.where(observed, np.arange(T)[:, None], np.iinfo(np.int32).max).astype(np.int64)
        self.next_obs = np.minimum.accumulate(idx[::-1], axis=0)[::-1]      # next real reading at/after k
        # existing sample rule: complete window, real reading at t, real target
        win_ok = np.lib.stride_tricks.sliding_window_view(self.finite, seq, axis=0).all(-1)   # [T-seq+1, N]
        self.sample_ok = np.zeros((T, N), dtype=bool)                        # indexed by origin t
        t = np.arange(seq - 1, T - h)
        self.sample_ok[t] = win_ok[t - seq + 1] & observed[t] & observed[t + h]
        self.observed = observed

    def origins(self, start, end):
        return np.arange(start + self.seq - 1, end - self.h)

    def batch(self, t):
        idx = t[:, None] + np.arange(-self.seq + 1, 1)                        # [B, 12]
        usable = self.finite[idx] & (self.next_obs[idx] <= t[:, None, None])  # causal: gap closed by t
        x = np.where(usable, self.scaled[idx], 0.0)
        X = np.stack([x, usable.astype(np.float32)], axis=1)                  # [B, 2, 12, N]
        return (torch.from_numpy(X.astype(np.float32)), torch.from_numpy(self.scaled[t + self.h]),
                torch.from_numpy(self.sample_ok[t]))


def predict(model, snaps, origins, scaler, device, bs=64):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(origins), bs):
            X, _, _ = snaps.batch(origins[i:i + bs])
            out.append(model(X.to(device)).cpu().numpy())
    return scaler.inverse(np.concatenate(out))                                # [n_origins, N]


def masked_val_mae(pred, snaps, origins, raw_y):
    m = snaps.sample_ok[origins]
    return float(np.abs(pred[m] - raw_y[origins + snaps.h][m]).mean())


def train(arch, snaps, tr_o, va_o, raw, scaler, cheb_all, n_nodes, device, seed, max_epochs, log):
    set_seed(seed)
    model = STGCN(cheb_all[:arch["cheb_k"]].to(device), n_nodes, channels=tuple(arch["channels"]),
                  kt=FIXED["kt"], dropout=FIXED["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=FIXED["learning_rate"])
    rng = np.random.default_rng(seed)
    best, best_state, bad, hist = np.inf, None, 0, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        order, tot, cnt = rng.permutation(tr_o), 0.0, 0
        for i in range(0, len(order), FIXED["batch_size"]):
            X, y, m = snaps.batch(order[i:i + FIXED["batch_size"]])
            X, y, m = X.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            loss = ((model(X) - y) ** 2)[m].mean()                               # loss only on valid samples
            loss.backward()
            opt.step()
            tot, cnt = tot + loss.item() * int(m.sum()), cnt + int(m.sum())
        vm = masked_val_mae(predict(model, snaps, va_o, scaler, device), snaps, va_o, raw)
        hist.append({"epoch": epoch, "train_mse_scaled": tot / cnt, "val_mae": vm})
        log(f"      STGCN epoch {epoch:2d}  train MSE {tot / cnt:.4f}  val MAE {vm:.3f}")
        if vm < best:
            best, best_state, bad = vm, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= FIXED["patience"]:
                break
    model.load_state_dict(best_state)
    return model, hist


def plot_actual_vs_predicted(pred, test, sensor, day, spd, label, path):
    import matplotlib.pyplot as plt
    sel = np.where((test["sensor"] == sensor) & (test["time"] // spd == day))[0]
    hours = (test["time"][sel] % spd) * 24 / spd
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(hours, test["y"][sel], color="black", lw=1.2, label="Actual")
    ax.plot(hours, pred[sel], color="#00838f", lw=1.2, label="Predicted - STGCN")
    ax.set(title=f"STGCN: actual vs 15-min-ahead forecast (sensor {sensor}, {label}, "
                 f"day MAE {np.abs(pred[sel] - test['y'][sel]).mean():.2f})",
           xlabel="Hour of day", ylabel="Traffic flow (vehicles / 5 min)", xlim=(0, 24))
    ax.legend(loc="upper left")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["debug", "final"], default="final")
    mode = ap.parse_args().mode
    debug = mode == "debug"
    out = EXP / ("results_debug" if debug else "results")
    model_dir = EXP / ("trained_model_debug" if debug else "trained_model")
    for d in [out / "figures", out / "metrics", out / "predictions", model_dir]:
        d.mkdir(parents=True, exist_ok=True)
    log = print
    cfg = load_config()
    ds, fc, seed = cfg["dataset"], cfg["forecast"], cfg["project"]["random_seed"]
    device = get_device()
    started = time.perf_counter()

    log("[1] Data (existing pipeline, read-only)")
    flow, filled, observed, _, bounds, splits = prepare(cfg)
    raw = flow.astype(np.float64)
    scaler = FlowScaler().fit(train_readings(filled, observed, bounds))
    saved = FlowScaler.load(ROOT / "trained_models" / "preprocessing" / "scaler.pkl")
    assert np.isclose(scaler.mean, saved.mean) and np.isclose(scaler.std, saved.std), "scaler differs from the main experiment"
    snaps = Snapshots(filled, observed, scaler, fc["sequence_length"], fc["horizon_steps"])
    o = {k: snaps.origins(*bounds[k]) for k in bounds}
    # the STGCN test samples must be exactly the locked test samples, in the same order
    te = splits["test"]
    m = snaps.sample_ok[o["test"]]
    tt = np.broadcast_to(o["test"][:, None] + fc["horizon_steps"], m.shape)[m]
    ss = np.broadcast_to(np.arange(flow.shape[1])[None, :], m.shape)[m]
    assert np.array_equal(tt, te["time"]) and np.array_equal(ss, te["sensor"]), "test samples differ from the locked set"
    log(f"    origins train/val/test {len(o['train'])}/{len(o['val'])}/{len(o['test'])}; "
        f"locked test samples matched: {m.sum():,}")

    log("[2] Graph")
    W, sigma = distance_graph(GRAPH, flow.shape[1])
    cheb_all, lam = cheb_polynomials(W, 3)
    graph_info = {"source": "data/raw/pems04_distance.csv (distributed with pems04.npz by the ASTGCN authors)",
                  "edges": int((W > 0).sum() // 2), "nodes": int(W.shape[0]), "sigma_m": sigma,
                  "weighting": "w = exp(-(d/sigma)^2), sigma = std of edge distances; symmetric; road edges only",
                  "laplacian": "scaled normalised Laplacian, Chebyshev polynomials", "lambda_max": lam}
    log(f"    {graph_info['edges']} edges, sigma {sigma:.1f}, lambda_max {lam:.3f}")

    max_e = 2 if debug else FIXED["max_epochs"]
    tune_e = 1 if debug else TUNE_EPOCHS
    log("[3] Search on validation")
    t0 = time.perf_counter()
    search = []
    for ch, k in itertools.product(SEARCH["channels"], SEARCH["cheb_k"]):
        arch = {"channels": ch, "cheb_k": k}
        _, hist = train(arch, snaps, o["train"], o["val"], raw, scaler, cheb_all, flow.shape[1], device, seed,
                        tune_e, lambda *_: None)
        search.append({"params": arch, "val_mae": round(min(h["val_mae"] for h in hist), 4)})
        log(f"    candidate {arch}  val MAE {search[-1]['val_mae']:.3f}")
    best = min(search, key=lambda r: r["val_mae"])["params"]
    tune_time = time.perf_counter() - t0

    log(f"[4] Final training {best}")
    t0 = time.perf_counter()
    model, history = train(best, snaps, o["train"], o["val"], raw, scaler, cheb_all, flow.shape[1], device, seed,
                           max_e, log)
    train_time = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, model_dir / "model.pt")
    (model_dir / "config.json").write_text(json.dumps({"architecture": best, "fixed": FIXED, "graph": graph_info,
                                                        "input_channels": ["scaled flow", "availability mask"],
                                                        "parameters": n_params, "history": history}, indent=2))

    log("[5] Validation metrics, then ONE pass over the locked test set")
    min_flow = cfg["evaluation"]["mape_min_flow"]
    pv = predict(model, snaps, o["val"], scaler, device)
    mv = snaps.sample_ok[o["val"]]
    val_metrics = all_metrics(raw[o["val"] + fc["horizon_steps"]][mv], pv[mv], min_flow)
    t0 = time.perf_counter()
    pt = predict(model, snaps, o["test"], scaler, device)
    infer_time = time.perf_counter() - t0
    pred = pt[m].astype(np.float32)
    test_metrics = all_metrics(te["y"], pred, min_flow)
    log(f"    STGCN test  MAE {test_metrics['MAE']:.3f}  RMSE {test_metrics['RMSE']:.3f}  MAPE {test_metrics['MAPE']:.3f}%")
    save_predictions(out / "predictions" / "stgcn.csv", te, pred, ds["start_date"], ds["interval_minutes"], min_flow)

    # comparison with the six existing models (their numbers are read from the main experiment's files)
    base = pd.read_csv(ROOT / "results" / "metrics" / "final_model_comparison.csv")
    size = sum(f.stat().st_size for f in model_dir.glob("*")) / 1e6
    row = {"Model": "STGCN", **test_metrics, "Training time (s)": train_time, "Inference time (s)": infer_time,
           "Inference per sample (ms)": infer_time / len(pred) * 1000, "Parameters": n_params, "Model size (MB)": size}
    table = pd.concat([base, pd.DataFrame([row])], ignore_index=True)
    table.to_csv(out / "metrics" / "seven_model_comparison.csv", index=False)

    th = json.loads((ROOT / "results" / "metrics" / "error_analysis_thresholds.json").read_text())["thresholds"]
    reg7 = pd.concat([pd.read_csv(ROOT / "results" / "metrics" / "error_analysis_by_regime.csv"),
                      ea.analyse({"STGCN": pred}, te, th)], ignore_index=True)
    reg7.to_csv(out / "metrics" / "error_analysis_by_regime_seven_models.csv", index=False)

    plots.COLORS["STGCN"] = "#00838f"      # runtime-only colour for the new model
    day = int(te["time"].min() // ds["steps_per_day"]) + 1
    label = (pd.Timestamp(ds["start_date"]) + pd.Timedelta(days=day)).strftime("%a %d %b %Y")
    plot_actual_vs_predicted(pred, te, 0, day, ds["steps_per_day"], label, out / "figures" / "stgcn_actual_vs_predicted.png")
    plots.fig_metric(table, "MAE", "vehicles / 5 min", "S1", out / "figures" / "seven_model_mae.png")
    plots.fig_metric(table, "RMSE", "vehicles / 5 min", "S2", out / "figures" / "seven_model_rmse.png")
    plots.fig_regimes(reg7, ea.REGIME_ORDER, out / "figures" / "seven_model_error_by_regime.png")
    plots.fig_training_curves({"STGCN": history}, out / "figures" / "stgcn_training_curve.png")

    meta = {"mode": mode, "timestamp": dt.datetime.now().isoformat(), "seed": seed, "device": device,
            **environment_info(), "graph": graph_info, "search": search, "selected": best, "fixed": FIXED,
            "tuning_time_s": tune_time, "val_metrics": val_metrics, "test_metrics": test_metrics,
            "training_time_s": train_time, "inference_time_s": infer_time, "parameters": n_params,
            "total_runtime_s": time.perf_counter() - started}
    (out / "metrics" / "stgcn_run.json").write_text(json.dumps(meta, indent=2, default=str))
    log("\n" + table[["Model", "MAE", "RMSE", "MAPE", "Training time (s)", "Inference time (s)"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
