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
    html = (ROOT / "site" / "template.html").read_text().replace("__DATA__", json.dumps(data, separators=(",", ":")))
    (ROOT / "site" / "index.html").write_text(html)
    print(f"site/index.html {len(html) / 1e6:.2f} MB, models: {list(c.index)}")


if __name__ == "__main__":
    main()
