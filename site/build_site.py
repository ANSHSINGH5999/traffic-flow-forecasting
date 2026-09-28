"""Builds site/index.html from the saved result files (six-model experiment + STGCN). No number is typed by hand.

    python site/build_site.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R, E = ROOT / "results", ROOT / "experiments" / "stgcn" / "results"
FILES = {"Historical Average": "historical_average", "Random Forest": "random_forest", "XGBoost": "xgboost",
         "LSTM": "lstm", "GRU": "gru", "Hybrid LSTM-XGBoost": "hybrid_lstm_xgboost"}
PRED = {**{m: R / "predictions" / f"{f}.csv" for m, f in FILES.items()}, "STGCN": E / "predictions" / "stgcn.csv"}
SENSORS = list(range(0, 307, 15))            # 21 sensors spread across the network
REG = ["low", "normal", "high", "peak", "sudden_change"]


def pct(a, b):
    return (b - a) / b * 100                 # positive = a has lower error than b


def main():
    fr = json.loads((R / "reports" / "final_results.json").read_text())
    summary = json.loads((R / "metrics" / "summary.json").read_text())
    stg = json.loads((E / "metrics" / "stgcn_run.json").read_text())
    comp = pd.read_csv(E / "metrics" / "seven_model_comparison.csv")
    reg = pd.read_csv(E / "metrics" / "error_analysis_by_regime_seven_models.csv")
    t0, t1 = fr["split_bounds"]["test"]
    L = t1 - t0
    start = pd.Timestamp("2018-01-01")

    series = {str(s): {} for s in SENSORS}
    hour = pd.read_csv(R / "metrics" / "error_by_hour.csv").drop(columns="hour").to_dict(orient="list")
    for name, path in PRED.items():
        df = pd.read_csv(path, usecols=["timestamp", "sensor_id", "actual_flow", "predicted_flow", "absolute_error"])
        ts = pd.to_datetime(df["timestamp"])
        if name == "STGCN":                  # MAE by hour for the new model, same definition as error_by_hour.csv
            hour["STGCN"] = df.groupby(ts.dt.hour)["absolute_error"].mean().reindex(range(24)).tolist()
        step = ((ts - start) / pd.Timedelta(minutes=5)).astype(int) - t0
        sub = df["sensor_id"].isin(SENSORS)
        for s, g in df[sub].groupby("sensor_id"):
            idx = step[g.index].to_numpy()
            arr = [None] * L
            for i, v in zip(idx, np.round(g["predicted_flow"].to_numpy(), 1)):
                arr[i] = float(v)
            series[str(s)][name] = arr
            if "Actual" not in series[str(s)]:
                act = [None] * L
                for i, v in zip(idx, g["actual_flow"].to_numpy()):
                    act[i] = int(round(v))
                series[str(s)]["Actual"] = act

    c = comp.set_index("Model")
    s, h, x = c.loc["STGCN"], c.loc["Hybrid LSTM-XGBoost"], c.loc["XGBoost"]
    pv = reg.pivot(index="Model", columns="Regime", values="MAE")[REG]
    others = pv.drop(index=["STGCN", "Historical Average"])["sudden_change"]
    stg_line = (f"STGCN vs the hybrid: MAE {pct(s['MAE'], h['MAE']):+.2f}%, RMSE {pct(s['RMSE'], h['RMSE']):+.2f}%, "
                f"MAPE {pct(s['MAPE'], h['MAPE']):+.2f}%; vs XGBoost: MAE {pct(s['MAE'], x['MAE']):+.2f}%, RMSE "
                f"{pct(s['RMSE'], x['RMSE']):+.2f}% (positive = STGCN lower error). It needed "
                f"{s['Training time (s)'] / x['Training time (s)']:.0f}x XGBoost's training time.")
    findings = [
        "Under this setup the test RMSE ranking was: " + ", ".join(f"{m} ({v:.3f})" for m, v in c["RMSE"].sort_values().items()) + ".",
        stg_line,
        summary["research_answer"][3],
        summary["research_answer"][2],
        "Lowest MAE per regime across the seven models: " + "; ".join(f"{r.replace('_', ' ')}: {pv[r].idxmin()} ({pv[r].min():.2f})" for r in REG) + ".",
        f"In sudden-change periods STGCN's MAE ({pv.loc['STGCN', 'sudden_change']:.2f}) was below every other learning model "
        f"({others.min():.2f}-{others.max():.2f}) but above the Historical Average ({pv.loc['Historical Average', 'sudden_change']:.2f}).",
        "Each model was trained once (one seed). STGCN's validation MAE varied by about ±1 between epochs, so small gaps are not a robust ranking.",
    ]
    verdict = [["Does the hybrid help?", summary["research_answer"][3]],
               ["Does spatial modelling help?", stg_line]]

    data = {
        "experiment": summary["experiment_id"], "testStart": int(t0), "len": int(L),
        "comparison": comp.fillna("").to_dict(orient="records"),
        "regime": reg.to_dict(orient="records"),
        "regimeDefs": json.loads((R / "metrics" / "error_analysis_thresholds.json").read_text())["definitions"],
        "hour": hour,
        "hybridGain": json.loads((R / "metrics" / "hybrid_feature_gain.json").read_text()),
        "findings": findings, "verdict": verdict,
        "dataset": fr["dataset"], "samples": fr["samples"], "bounds": fr["split_bounds"],
        "stgcn": {"selected": stg["selected"], "parameters": stg["parameters"], "graph": stg["graph"]},
        "sensors": series,
    }
    render_simple(c, pv, fr)
    html = (ROOT / "site" / "template.html").read_text().replace("__DATA__", json.dumps(data, separators=(",", ":")))
    (ROOT / "site" / "index.html").write_text(html)
    print(f"site/index.html {len(html) / 1e6:.2f} MB, models: {list(c.index)}")


def render_simple(c, pv, fr):
    """Kid-friendly page. Every number is computed here, and every sentence's claim is asserted against the data."""
    ha, hy, xg, st = (c.loc[m] for m in ["Historical Average", "Hybrid LSTM-XGBoost", "XGBoost", "STGCN"])
    learners = c.drop(index="Historical Average")
    assert (learners["MAE"] < ha["MAE"]).all(), "claim: every learning player beat the calendar kid"
    top3 = [list(c[k].sort_values().index[:3]) for k in ("MAE", "RMSE", "MAPE")]
    assert top3[0] == top3[1] == top3[2], "claim: same top-3 order on all metrics"
    sc = pv["sudden_change"].sort_values()
    assert list(sc.index[:2]) == ["Historical Average", "STGCN"], "claim: calendar kid first, STGCN second in sudden changes"
    assert hy["MAE"] < xg["MAE"] and st["MAE"] < hy["MAE"], "claim: hybrid beat XGBoost, STGCN beat hybrid"

    # the real last hour of machine 0 before 2018-02-21 08:00 (the example used by scripts/test_inference.py)
    flow = np.load(ROOT / "data" / "raw" / "pems04.npz")["data"][:, 0, 0]
    origin = int((pd.Timestamp("2018-02-21 08:00") - pd.Timestamp("2018-01-01")) / pd.Timedelta(minutes=5))
    hour = flow[origin - 11:origin + 1]
    bars = "".join(f'<div class="c"><i style="height:{max(8, v / hour.max() * 90):.0f}px"></i>{v:.0f}</div>' for v in hour)

    order = c["MAE"].sort_values()
    worst = order.max()
    css = {"Historical Average": "--ha", "Random Forest": "--rf", "XGBoost": "--xgb", "LSTM": "--lstm", "GRU": "--gru",
           "Hybrid LSTM-XGBoost": "--hyb", "STGCN": "--stg"}
    race = "".join(f'<div class="lane"><span>{m}</span><div class="bar"><i style="width:{v / worst * 100:.1f}%;background:var({css[m]})">'
                   f'</i></div><b>{v:.1f}</b></div>' for m, v in order.items())
    vals = {
        "counts_per_machine": f"{fr['dataset']['time_steps']:,}",
        "missing_pct": f"{fr['dataset']['zero_pct']:.1f}",
        "count_bars": bars,
        "test_samples": f"{fr['samples']['test']:,}",
        "race": race,
        "top3": ", ".join(top3[0]),
        "best_vs_ha": f"{pct(c['MAE'].min(), ha['MAE']):.0f}",
        "hyb_vs_xgb": f"{pct(hy['MAE'], xg['MAE']):.1f}",
        "hyb_time": f"{hy['Training time (s)'] / xg['Training time (s)']:.0f}",
        "stg_vs_hyb": f"{pct(st['MAE'], hy['MAE']):.0f}",
        "stg_time": f"{st['Training time (s)'] / xg['Training time (s)']:.0f}",
    }
    html = (ROOT / "site" / "simple_template.html").read_text()
    for k, v in vals.items():
        html = html.replace("{{" + k + "}}", v)
    assert "{{" not in html, "unfilled placeholder"
    (ROOT / "site" / "simple.html").write_text(html)
    print("site/simple.html", {k: v for k, v in vals.items() if k not in ("count_bars", "race")})


if __name__ == "__main__":
    main()
