"""Academic-style figures. Every figure has a title, labelled axes with units and a legend where needed."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"Historical Average": "#8c8c8c", "Random Forest": "#2e7d32", "XGBoost": "#f9a825",
          "LSTM": "#1565c0", "GRU": "#6a1b9a", "Hybrid LSTM-XGBoost": "#c62828"}
FLOW = "Traffic flow (vehicles / 5 min)"

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight"})


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)


def _bars(ax, names, values, fmt="%.2f"):
    bars = ax.bar(names, values, color=[COLORS.get(n, "#555") for n in names])
    ax.bar_label(bars, fmt=fmt, fontsize=9)
    ax.tick_params(axis="x", rotation=25)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")


# ---------- dataset ----------
def dataset_figures(flow, observed, out, steps_per_day, start_weekday=0):
    week = slice(0, 7 * steps_per_day)
    fig, ax = plt.subplots(figsize=(13, 4))
    days = np.arange(week.stop) / steps_per_day
    for s in range(3):
        ax.plot(days, flow[week, s], lw=0.8, label=f"Sensor {s}")
    ax.set(title="Raw PeMSD4 traffic flow - first week (1-7 Jan 2018)", xlabel="Day", ylabel=FLOW)
    ax.legend()
    _save(fig, out / "data_raw_flow_week.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(flow[observed], bins=80, color="#1565c0")
    ax.set(title="Distribution of observed traffic flow (all sensors)", xlabel=FLOW, ylabel="Number of readings")
    _save(fig, out / "data_flow_distribution.png")

    t = np.arange(flow.shape[0])
    slot, weekday = t % steps_per_day, ((t // steps_per_day + start_weekday) % 7) < 5
    f = np.where(observed, flow, np.nan)
    hours = np.arange(steps_per_day) * 24 / steps_per_day
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, m in [("Weekdays", weekday), ("Weekends", ~weekday)]:
        ax.plot(hours, [np.nanmean(f[m & (slot == s)]) for s in range(steps_per_day)], label=label)
    ax.set(title="Average daily traffic profile (all sensors)", xlabel="Hour of day", ylabel=FLOW)
    ax.legend()
    _save(fig, out / "data_daily_profile.png")

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.bar(np.arange(flow.shape[1]), (~observed).mean(0) * 100, color="#c62828", width=1)
    ax.set(title="Missing (zero) readings per sensor", xlabel="Sensor ID", ylabel="Missing readings (%)")
    _save(fig, out / "data_missing_per_sensor.png")


# ---------- results ----------
def fig_actual_vs_predicted(preds, test, sensor, day, steps_per_day, path, day_label):
    sel = np.where((test["sensor"] == sensor) & (test["time"] // steps_per_day == day))[0]
    hours = (test["time"][sel] % steps_per_day) * 24 / steps_per_day
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), sharex=True, sharey=True)
    for ax, (name, p) in zip(axes.flat, preds.items()):
        ax.plot(hours, test["y"][sel], color="black", lw=1.1, label="Actual")
        ax.plot(hours, p[sel], color=COLORS[name], lw=1.1, label="Predicted")
        ax.set_title(name)
        ax.legend(fontsize=9, loc="upper left")
    fig.supxlabel("Hour of day")
    fig.supylabel(FLOW)
    fig.suptitle(f"Figure 1 - Actual vs predicted flow, 15-min ahead (sensor {sensor}, {day_label})", fontsize=14)
    _save(fig, path)


def fig_metric(table, col, unit, num, path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _bars(ax, table["Model"], table[col])
    ax.set(title=f"Figure {num} - {col} on the test set (lower is better)", ylabel=f"{col} ({unit})")
    _save(fig, path)


def fig_time(table, col, num, title, path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _bars(ax, table["Model"], table[col], fmt="%.2f")
    ax.set_yscale("log")
    ax.set(title=f"Figure {num} - {title} (log scale)", ylabel="Seconds")
    _save(fig, path)


def fig_residuals(preds, test, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(-150, 150, 121)
    for name, p in preds.items():
        ax.hist(np.clip(p - test["y"], -150, 150), bins=bins, histtype="step", lw=1.4,
                color=COLORS[name], label=name, density=True)
    ax.set(title="Figure 7 - Residual distribution (predicted - actual, clipped to +/-150)",
           xlabel="Residual (vehicles / 5 min)", ylabel="Density")
    ax.legend()
    _save(fig, path)


def fig_regimes(regime_df, order, path):
    models = list(dict.fromkeys(regime_df["Model"]))
    fig, ax = plt.subplots(figsize=(13, 5))
    w = 0.8 / len(models)
    x = np.arange(len(order))
    for i, m in enumerate(models):
        d = regime_df[regime_df["Model"] == m].set_index("Regime").loc[order]
        ax.bar(x + i * w - 0.4 + w / 2, d["MAE"], w, color=COLORS[m], label=m)
    ax.set_xticks(x, [r.replace("_", " ") for r in order])
    ax.set(title="Figure 8 - MAE by traffic regime (data-driven thresholds from training data)",
           xlabel="Regime", ylabel="MAE (vehicles / 5 min)")
    ax.legend(ncol=3, fontsize=9)
    _save(fig, path)


def fig_error_by_hour(hour_df, path):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for name in hour_df.columns:
        ax.plot(hour_df.index, hour_df[name], marker="o", ms=3, color=COLORS[name], label=name)
    ax.set(title="Figure 9 - MAE by hour of day", xlabel="Hour of day", ylabel="MAE (vehicles / 5 min)")
    ax.legend(ncol=3, fontsize=9)
    _save(fig, path)


def fig_importance(names, values, title, xlabel, path, top=15):
    order = np.argsort(values)[-top:]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(np.array(names)[order], np.array(values)[order], color="#1565c0")
    ax.set(title=title, xlabel=xlabel)
    _save(fig, path)


def fig_training_curves(histories, path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, hist in histories.items():
        ax.plot([h["epoch"] for h in hist], [h["val_mae"] for h in hist], marker="o", color=COLORS[name], label=name)
    ax.set(title="Figure 12 - Validation MAE per epoch (early stopping)", xlabel="Epoch",
           ylabel="Validation MAE (vehicles / 5 min)")
    ax.legend()
    _save(fig, path)


def fig_single_actual_vs_predicted(name, pred, test, sensor, day, steps_per_day, path, day_label, idx):
    sel = np.where((test["sensor"] == sensor) & (test["time"] // steps_per_day == day))[0]
    hours = (test["time"][sel] % steps_per_day) * 24 / steps_per_day
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(hours, test["y"][sel], color="black", lw=1.2, label="Actual")
    ax.plot(hours, pred[sel], color=COLORS[name], lw=1.2, label=f"Predicted - {name}")
    mae_day = np.abs(pred[sel] - test["y"][sel]).mean()
    ax.set(title=f"Figure 1{'abcdef'[idx - 1]} - {name}: actual vs 15-min-ahead forecast "
                 f"(sensor {sensor}, {day_label}, day MAE {mae_day:.2f})",
           xlabel="Hour of day", ylabel=FLOW, xlim=(0, 24))
    ax.legend(loc="upper left")
    _save(fig, path)
