"""Traffic Flow Forecasting System - demo app (inference only, never retrains).  Run:  streamlit run app/app.py"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.components.descriptions import ABOUT  # noqa: E402
from app.components.predictor import MODEL_NAMES, ForecastError, Forecaster, load_clean_data  # noqa: E402
from src.utils import load_config  # noqa: E402

COLORS = {"Historical Average": "#8c8c8c", "Random Forest": "#2e7d32", "XGBoost": "#f9a825",
          "LSTM": "#1565c0", "GRU": "#6a1b9a", "Hybrid LSTM-XGBoost": "#c62828"}
FLOW = "Flow (vehicles / 5 min)"

st.set_page_config(page_title="Traffic Flow Forecasting System", page_icon="🚦", layout="wide")
cfg = load_config()
MODELS_DIR = os.environ.get("TRAFFIC_MODELS_DIR", cfg["paths"]["models_dir"])
RESULTS_DIR = Path(os.environ.get("TRAFFIC_RESULTS_DIR", cfg["paths"]["results_dir"]))
ds, fc = cfg["dataset"], cfg["forecast"]
SPD, SEQ, H = ds["steps_per_day"], fc["sequence_length"], fc["horizon_steps"]


@st.cache_resource(show_spinner="Loading PeMSD4 data ...")
def get_data():
    return load_clean_data(cfg)


@st.cache_resource(show_spinner="Loading trained models ...")
def get_forecaster():
    return Forecaster(cfg, MODELS_DIR)


def when(step):
    return pd.Timestamp(ds["start_date"]) + pd.Timedelta(minutes=int(step) * ds["interval_minutes"])


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def line_chart(title, series, marks=(), height=3.3):
    fig, ax = plt.subplots(figsize=(11, height))
    for label, (x, y, style) in series.items():
        ax.plot(x, y, label=label, **style)
    for label, (x, y, style) in marks:
        ax.scatter([x], [y], label=label, zorder=3, s=90, **style)
    ax.set(ylabel=FLOW, title=title)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


st.title("🚦 Traffic Flow Forecasting System")
st.caption("15-minute traffic-flow forecasting demonstration on **historical** PeMSD4 data (not live traffic) · "
           "forecasts come from the saved trained models, nothing is retrained")

try:
    raw, filled, observed = get_data()
    fcst = get_forecaster()
    n_sensors = fcst.n_sensors()
except ForecastError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Start-up failed ({type(e).__name__}). Check that data/raw/pems04.npz and {MODELS_DIR}/ exist.")
    st.stop()

if Path(MODELS_DIR).name.endswith("_debug"):
    st.warning("Using DEBUG models (a few sensors, few epochs) - these are not the final research models.")

T = filled.shape[0]
test_start = int(T * (cfg["split"]["train"] + cfg["split"]["validation"]))
first_day, last_day = -(-test_start // SPD), (T - 1) // SPD

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Forecast settings")
    sensor = st.selectbox("Sensor", range(n_sensors), format_func=lambda s: f"Sensor {s}")
    day = st.selectbox("Date (unseen test period)", range(first_day, last_day + 1),
                       format_func=lambda d: (pd.Timestamp(ds["start_date"]) + pd.Timedelta(days=d)).strftime("%a %d %b %Y"))
    times = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, ds["interval_minutes"])]
    hhmm = st.select_slider("Time (last known reading)", options=times, value="08:00")
    model_name = st.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index("Hybrid LSTM-XGBoost"))
    generate = st.button("Generate Forecast", type="primary", width="stretch")
    st.divider()
    st.caption(f"Dataset: PeMSD4 ({n_sensors} sensors, 5-min readings)  \n"
               f"Input: last {SEQ * 5} min ({SEQ} readings)  \nHorizon: {fc['horizon_minutes']} min (t + {H})")

origin = min(day * SPD + times.index(hhmm), T - H - 1)
target = origin + H

tab_demo, tab_research = st.tabs(["Forecast demonstration", "Research results"])

# ======================= Forecast demonstration =======================
with tab_demo:
    st.subheader(f"Recent traffic history — sensor {sensor}")
    ctx = np.arange(max(0, origin - 3 * 12 + 1), origin + 1)
    win = ctx[-SEQ:]
    line_chart(f"Observed flow up to {when(origin):%a %d %b %Y %H:%M} (bold = the {SEQ * 5}-minute model input)",
               {"Earlier readings": ([when(t) for t in ctx], filled[ctx, sensor], {"color": "#999", "lw": 1}),
                f"Model input (last {SEQ * 5} min)": ([when(t) for t in win], filled[win, sensor],
                                                      {"color": "black", "lw": 2.2})})

    st.subheader(f"15-minute forecast — {model_name}")
    key = (sensor, model_name, origin)
    if generate:
        st.session_state["last"] = key
    pred = None
    if st.session_state.get("last") != key:
        st.info("Choose sensor, date, time and model in the sidebar, then press **Generate Forecast**.")
    else:
        try:
            X, tt, valid = fcst.windows(filled, sensor, [origin], observed)
            if not valid[0]:
                raise ForecastError("No forecast possible: the sensor's reading at this time (or part of the last hour) "
                                    "is missing because of a sensor outage. Try another time or sensor.")
            pred = float(fcst.predict(model_name, X, tt, sensor)[0])
        except ForecastError as e:
            st.error(str(e))

    if pred is not None:
        actual = float(raw[target, sensor]) if observed[target, sensor] else None
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Predicted traffic flow at {when(target):%H:%M}", f"{pred:.0f} vehicles")
        if actual is not None:
            c2.metric(f"Actual traffic flow at {when(target):%H:%M}", f"{actual:.0f} vehicles")
            c3.metric("Absolute error", f"{abs(pred - actual):.1f} vehicles", f"{(pred - actual) / actual * 100:+.1f}%",
                      delta_color="off")
        else:
            c2.metric("Actual traffic flow", "not recorded (sensor outage)")
        marks = [(f"Forecast ({model_name})", (when(target), pred, {"color": COLORS[model_name]}))]
        if actual is not None:
            marks.append(("Actual", (when(target), actual, {"color": "black", "marker": "X"})))
        line_chart("Last hour of input and the 15-minute-ahead forecast",
                   {"Input": ([when(t) for t in win], filled[win, sensor], {"color": "black", "lw": 1.8}),
                    "": ([when(origin), when(target)], [filled[origin, sensor], pred],
                         {"color": COLORS[model_name], "ls": "--"})}, marks)

        with st.expander("All six models at this exact moment"):
            rows = []
            for name in MODEL_NAMES:
                try:
                    p = float(fcst.predict(name, X, tt, sensor)[0])
                    rows.append({"Model": name, "Forecast (vehicles)": round(p, 1),
                                 "Absolute error": round(abs(p - actual), 1) if actual is not None else None})
                except ForecastError as e:
                    rows.append({"Model": name, "Forecast (vehicles)": None, "Absolute error": None, "Note": str(e)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("Actual vs predicted — whole selected day")
    day_origins = np.arange(max(day * SPD - H, SEQ - 1), min((day + 1) * SPD - H, T - H))
    try:
        X, tt, valid = fcst.windows(filled, sensor, day_origins, observed)
        keep = valid & observed[tt, sensor]
        if not keep.any():
            st.warning("No usable readings for this sensor on this day (sensor outage).")
        else:
            p = fcst.predict(model_name, X[keep], tt[keep], sensor)
            y = raw[tt[keep], sensor]
            ts = [when(t) for t in tt[keep]]
            line_chart(f"Sensor {sensor} · {when(day * SPD):%a %d %b %Y} · MAE for this day: {np.abs(p - y).mean():.2f}",
                       {"Actual": (ts, y, {"color": "black", "lw": 1.1}),
                        f"Predicted ({model_name})": (ts, p, {"color": COLORS[model_name], "lw": 1.1})}, height=3.6)
    except ForecastError as e:
        st.warning(str(e))

    metrics_file = RESULTS_DIR / "metrics" / "final_model_comparison.csv"
    if not metrics_file.exists():
        metrics_file = RESULTS_DIR / "metrics" / "model_metrics.csv"
    st.subheader("Model performance (locked test set, all sensors)")
    if not metrics_file.exists():
        st.warning("No result files found - run `python run_pipeline.py --mode final`.")
    else:
        table = pd.read_csv(metrics_file)
        r = table.set_index("Model").loc[model_name]
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{r['MAE']:.2f}")
        c2.metric("RMSE", f"{r['RMSE']:.2f}")
        c3.metric("MAPE", f"{r['MAPE']:.2f}%")

        st.subheader("Model comparison")
        show = table[["Model", "MAE", "RMSE", "MAPE", "Training time (s)", "Inference time (s)"]].round(3)
        st.dataframe(show.style.highlight_min(subset=["MAE", "RMSE", "MAPE"], color="#c8e6c9"),
                     hide_index=True, width="stretch")

    st.subheader(f"About the model — {model_name}")
    st.write(ABOUT[model_name])

# ======================= Research results =======================
with tab_research:
    summary = read_json(RESULTS_DIR / "metrics" / "summary.json")
    final = read_json(RESULTS_DIR / "reports" / "final_results.json")
    if summary is None or final is None:
        st.warning("Research result files not found - run the pipeline and `python -m src.reporting`.")
    else:
        if summary["mode"] != "final":
            st.warning("These are DEBUG results, not research results.")
        st.caption(f"Experiment `{summary['experiment_id']}` · every value below is read from `{RESULTS_DIR}/`")
        d = final["dataset"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sensors × time steps", f"{d['sensors']} × {d['time_steps']:,}")
        c2.metric("Zero (missing) readings", f"{d['zero_pct']:.2f}%")
        c3.metric("Test samples", f"{final['samples']['test']:,}")
        c4.metric("Forecast", f"{SEQ * 5} min → +{fc['horizon_minutes']} min")

        st.subheader("Final model comparison")
        comp = pd.DataFrame(final["final_model_comparison"])
        st.dataframe(comp.round(4).style.highlight_min(subset=["MAE", "RMSE", "MAPE"], color="#c8e6c9"),
                     hide_index=True, width="stretch")
        fig_dir = RESULTS_DIR / "figures"
        cols = st.columns(3)
        for col, f in zip(cols, ["fig2_mae.png", "fig3_rmse.png", "fig4_mape.png"]):
            if (fig_dir / f).exists():
                col.image(str(fig_dir / f))

        st.subheader("Research findings")
        for line in summary["research_answer"]:
            st.markdown(f"- {line}")

        st.subheader("Computational cost")
        cols = st.columns(2)
        for col, f in zip(cols, ["fig5_training_time.png", "fig6_inference_time.png"]):
            if (fig_dir / f).exists():
                col.image(str(fig_dir / f))

        st.subheader("Error analysis")
        ea = final["error_analysis"]
        st.markdown("\n".join(f"- **{k}**: {v}" for k, v in ea["definitions"].items()))
        reg = pd.DataFrame(ea["by_regime"]).pivot(index="Model", columns="Regime", values="MAE")
        st.dataframe(reg.loc[MODEL_NAMES, ["low", "normal", "high", "peak", "sudden_change"]].round(3), width="stretch")
        for f in ["fig8_error_by_regime.png", "fig9_error_by_hour.png", "fig7_residual_distribution.png"]:
            if (fig_dir / f).exists():
                st.image(str(fig_dir / f))

        figs = sorted(p.name for p in fig_dir.glob("*.png"))
        if figs:
            st.subheader("All figures")
            choice = st.selectbox("Figure", figs)
            st.image(str(fig_dir / choice))
