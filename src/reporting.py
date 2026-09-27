"""Builds every report from the SAVED result files (no number is typed by hand).

    python -m src.reporting                   # rebuild reports for results/ + trained_models/
    python -m src.reporting results_debug trained_models_debug
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import models  # noqa: F401  (XGBoost/torch import order)
from src.evaluation import error_analysis as ea
from src.features.feature_engineering import FEATURE_DOCS
from src.preprocessing.prepare import prepare
from src.utils import load_config
from src.visualization import plots

MODEL_FILES = {"Historical Average": "historical_average", "Random Forest": "random_forest", "XGBoost": "xgboost",
               "LSTM": "lstm", "GRU": "gru", "Hybrid LSTM-XGBoost": "hybrid_lstm_xgboost"}
HYBRID = "Hybrid LSTM-XGBoost"


def pct_lower(a, b):
    """How much lower a is than b, in % of b (positive = a has lower error)."""
    return (b - a) / b * 100


def load_predictions(results, test):
    preds = {}
    for name, stem in MODEL_FILES.items():
        df = pd.read_csv(results / "predictions" / f"{stem}.csv", usecols=["sensor_id", "actual_flow", "predicted_flow"])
        if len(df) != len(test["y"]) or not np.array_equal(df["sensor_id"].to_numpy(), test["sensor"]):
            raise ValueError(f"{stem}.csv does not match the test samples - rerun the pipeline")
        preds[name] = df["predicted_flow"].to_numpy(np.float32)
    return preds


def large_error_analysis(preds, test, th):
    """Where do each model's largest errors (top 1% absolute error) occur?"""
    masks = ea.regime_masks(test, th)
    rows = []
    for name, p in preds.items():
        err = np.abs(p - test["y"])
        cut = np.percentile(err, 99)
        big = err >= cut
        for regime in ea.REGIME_ORDER:
            rows.append({"Model": name, "Top-1% error threshold": float(cut), "Regime": regime,
                         "Share of all test samples (%)": float(masks[regime].mean() * 100),
                         "Share of top-1% errors (%)": float(masks[regime][big].mean() * 100)})
    return pd.DataFrame(rows)


def hybrid_gain_split(models_dir):
    """Share of the hybrid XGBoost's total gain (sum over all splits) that comes from LSTM features vs engineered features."""
    from src.models import xgboost_model
    d = Path(models_dir) / "hybrid_lstm_xgboost"
    cfg = json.loads((d / "config.json").read_text())
    h = cfg["architecture"]["hidden_size"]
    gain = xgboost_model.load(str(d / "xgboost.json")).get_booster().get_score(importance_type="total_gain")
    total = sum(gain.values())
    lstm = sum(v for k, v in gain.items() if int(k[1:]) < h)
    return {"lstm_hidden_size": h, "lstm_feature_gain_share": lstm / total,
            "engineered_feature_gain_share": 1 - lstm / total}


def research_answer(t, reg, hy):
    """Answer to the research question, phrased only from measured values."""
    singles = t.drop(index=HYBRID)
    best_single = singles["RMSE"].idxmin()
    h, b, ha = t.loc[HYBRID], t.loc[best_single], t.loc["Historical Average"]
    order = ", ".join(f"{m} ({v:.3f})" for m, v in t["RMSE"].sort_values().items())
    lines = [
        f"Under this experimental setup the test RMSE ranking was: {order}.",
        f"All learning models are compared with the Historical Average baseline (MAE {ha['MAE']:.3f}). "
        f"The measured MAE reduction relative to the baseline ranged from "
        f"{pct_lower(singles.drop(index='Historical Average')['MAE'].max(), ha['MAE']):.2f}% to "
        f"{pct_lower(t['MAE'].min(), ha['MAE']):.2f}%.",
        f"The tree models, which receive engineered features including time of day and weekday, measured "
        f"MAE {t.loc['Random Forest', 'MAE']:.3f} (RF) and {t.loc['XGBoost', 'MAE']:.3f} (XGBoost). The sequence-only "
        f"recurrent models measured {t.loc['LSTM', 'MAE']:.3f} (LSTM) and {t.loc['GRU', 'MAE']:.3f} (GRU).",
    ]
    better = [k for k in ("MAE", "RMSE", "MAPE") if h[k] < b[k]]
    worse = [k for k in ("MAE", "RMSE", "MAPE") if h[k] >= b[k]]
    verdict = ("lower error on all three metrics" if not worse else
               "no improvement on any metric" if not better else
               f"a mixed result: lower {', '.join(better)} but not lower {', '.join(worse)}")
    lines.append(
        f"Hybrid LSTM-XGBoost vs the strongest single model ({best_single}): MAE {pct_lower(h['MAE'], b['MAE']):+.2f}%, "
        f"RMSE {pct_lower(h['RMSE'], b['RMSE']):+.2f}%, MAPE {pct_lower(h['MAPE'], b['MAPE']):+.2f}% "
        f"(positive = hybrid lower error): {verdict}. It came at "
        f"{h['Training time (s)'] / b['Training time (s)']:.1f}x the training time and "
        f"{h['Inference time (s)'] / max(b['Inference time (s)'], 1e-9):.1f}x the inference time.")
    lines.append(
        f"Inside the hybrid, LSTM-derived features account for {hy['lstm_feature_gain_share'] * 100:.1f}% of the XGBoost "
        f"gain (the engineered features account for the rest). This measures how much the model uses them, not causality.")
    levels = reg[reg["Regime"] != "sudden_change"]
    worst = levels.loc[levels.groupby("Model")["MAE"].idxmax()].set_index("Model")["Regime"]
    sc = reg[reg["Regime"] == "sudden_change"].set_index("Model")["MAE"]
    lvl_max = levels.groupby("Model")["MAE"].max().reindex(sc.index)
    worst_txt = (f"all six models had their highest flow-regime MAE in the '{worst.iloc[0]}' regime"
                 if worst.nunique() == 1 else
                 "highest flow-regime MAE by model: " + ", ".join(f"{m}: {r}" for m, r in worst.items()))
    sc_txt = ("for every model the sudden-change MAE exceeded its worst flow-regime MAE"
              if (sc > lvl_max).all() else
              f"sudden-change MAE exceeded the worst flow-regime MAE for {int((sc > lvl_max).sum())} of 6 models")
    lines.append(f"Error analysis: {worst_txt}; {sc_txt}.")
    return lines


def build_reports(results_dir="results", models_dir="trained_models"):
    results, models_dir = Path(results_dir), Path(models_dir)
    cfg = load_config()
    ds, fc = cfg["dataset"], cfg["forecast"]
    meta = json.loads((results / "run_metadata.json").read_text())
    mode = meta["mode"]
    m = results / "metrics"

    table = pd.read_csv(m / "model_metrics.csv")
    val = pd.read_csv(m / "validation_metrics.csv")
    reg = pd.read_csv(m / "error_analysis_by_regime.csv")
    fi = pd.read_csv(m / "feature_importance.csv")
    params = json.loads((m / "best_parameters.json").read_text())
    dataset = json.loads((results / "reports" / "dataset_report.json").read_text())

    # recompute the exact samples (deterministic) to align predictions and regimes
    _, _, _, clean_info, bounds, splits = prepare(cfg, cfg["debug"]["max_sensors"] if mode == "debug" else None)
    test = splits["test"]
    th_file = m / "error_analysis_thresholds.json"
    th = (json.loads(th_file.read_text())["thresholds"] if th_file.exists()
          else ea.regime_thresholds(splits["train"]))
    defs = ea.regime_definitions(th)
    th_file.write_text(json.dumps({"thresholds": th, "definitions": defs}, indent=2))
    preds = load_predictions(results, test)

    # final comparison table
    cmp_cols = ["Model", "MAE", "RMSE", "MAPE", "Training time (s)", "Inference time (s)",
                "Inference per sample (ms)", "Parameters", "Model size (MB)"]
    comparison = table[cmp_cols]
    comparison.to_csv(m / "final_model_comparison.csv", index=False)

    # large errors + hybrid internals
    big = large_error_analysis(preds, test, th)
    big.to_csv(m / "large_error_analysis.csv", index=False)
    hy = hybrid_gain_split(models_dir)
    (m / "hybrid_feature_gain.json").write_text(json.dumps(hy, indent=2))

    # one actual-vs-predicted figure per model (+ the combined grid)
    fig_dir = results / "figures"
    day = int(test["time"].min() // ds["steps_per_day"]) + 1
    day_label = (pd.Timestamp(ds["start_date"]) + pd.Timedelta(days=day)).strftime("%a %d %b %Y")
    for i, name in enumerate(MODEL_FILES, 1):
        plots.fig_single_actual_vs_predicted(name, preds[name], test, 0, day, ds["steps_per_day"],
                                             fig_dir / f"fig1{'abcdef'[i - 1]}_actual_vs_predicted_{MODEL_FILES[name]}.png",
                                             day_label, i)

    t = table.set_index("Model")
    answer = research_answer(t, reg, hy)
    summary = {"experiment_id": meta["experiment_id"], "mode": mode,
               "best_model_by_rmse": t["RMSE"].idxmin(), "best_model_by_mae": t["MAE"].idxmin(),
               "test_metrics": table[["Model", "MAE", "RMSE", "MAPE"]].to_dict(orient="records"),
               "research_answer": answer}
    (m / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---------- final_results.json ----------
    (results / "reports" / "final_results.json").write_text(json.dumps({
        **summary, "dataset": dataset, "preprocessing": clean_info,
        "samples": {k: len(v["y"]) for k, v in splits.items()}, "split_bounds": bounds,
        "forecast": fc, "split": cfg["split"], "hyperparameters": params,
        "final_model_comparison": comparison.to_dict(orient="records"),
        "validation_metrics": val.to_dict(orient="records"),
        "error_analysis": {"definitions": defs, "by_regime": reg.to_dict(orient="records"),
                           "large_errors": big.to_dict(orient="records")},
        "hybrid_feature_gain": hy, "environment": {k: meta[k] for k in ("python", "os", "libraries", "device")},
    }, indent=2, default=str))

    # ---------- final_results.md ----------
    def md(df, fmt="{:.3f}"):
        cols = list(df.columns)
        out = "| " + " | ".join(cols) + " |\n|" + "|".join("---" if c == cols[0] else "---:" for c in cols) + "|\n"
        for _, r in df.iterrows():
            out += "| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |\n"
        return out

    samples = {k: len(v["y"]) for k, v in splits.items()}
    reg_mae = reg.pivot(index="Model", columns="Regime", values="MAE")[ea.REGIME_ORDER].loc[table["Model"]].reset_index()
    big_sc = big[big["Regime"] == "sudden_change"][["Model", "Top-1% error threshold", "Share of all test samples (%)",
                                                    "Share of top-1% errors (%)"]]
    big_peak = big[big["Regime"] == "peak"][["Model", "Share of all test samples (%)", "Share of top-1% errors (%)"]]
    top = lambda col: ", ".join(fi.sort_values(col, ascending=False)["feature"].head(5))  # noqa: E731
    banner = "" if mode == "final" else "> **DEBUG MODE - these are NOT research results.**\n\n"
    ft = fc["sequence_length"]

    text = f"""# Final Experimental Results

{banner}Experiment `{meta['experiment_id']}` · generated from the files in `{results}/` · run on {meta['timestamp'][:19]}

## 1. Dataset
| Property | Measured value |
|---|---|
| File | `{dataset['file']}` ({dataset['file_size_mb']} MB) |
| Array shape | {dataset['array_shape']} (time steps x sensors x channels: flow, occupancy, speed) |
| Sensors used | {int(test['sensor'].max()) + 1} |
| Interval | {dataset['sampling_interval_minutes']} min - {dataset['interval_check']} |
| NaN / inf / negative | {dataset['nan_count']} / {dataset['inf_count']} / {dataset['negative_count']} |
| Zero readings (treated as missing) | {dataset['zero_count']:,} ({dataset['zero_pct']:.2f}%) |
| Flow min / max / mean / median / std | {dataset['flow_stats']['min']:.0f} / {dataset['flow_stats']['max']:.0f} / {dataset['flow_stats']['mean']:.2f} / {dataset['flow_stats']['median']:.0f} / {dataset['flow_stats']['std']:.2f} |

## 2. Experimental Configuration
- Chronological split {cfg['split']['train']:.0%} / {cfg['split']['validation']:.0%} / {cfg['split']['test']:.0%} of the time axis (steps {bounds['train']}, {bounds['val']}, {bounds['test']}); no shuffling.
- Samples: train {samples['train']:,} · validation {samples['val']:,} · test {samples['test']:,}.
- Missing values: gaps of <= {cfg['preprocessing']['max_interpolation_gap']} steps linearly interpolated; longer gaps dropped ({clean_info['interpolated_readings']:,} readings interpolated, {clean_info['unfilled_long_gap_readings']:,} left missing).
- Leakage rules: the reading at the forecast origin t must be real (so interpolation never uses readings after t); targets are real readings only; windows never cross split boundaries; the scaler is fitted on training readings only; model selection and early stopping use validation only; the test set was evaluated once, after all models were saved.
- One global model per algorithm across all sensors (sensor ID is not a feature). Seed {meta['config']['project']['random_seed']}; device `{meta['device']}`.

## 3. Forecasting Task
Input: the previous {ft} readings ({ft * ds['interval_minutes']} minutes, t-{ft - 1} ... t) → target: traffic flow at t+{fc['horizon_steps']} ({fc['horizon_minutes']} minutes ahead). Only this horizon is in the comparison.

## 4. Model Configurations (selected on the validation set)
""" + "".join(f"- **{name}**: `{json.dumps(p.get('selected', p.get('method')))}`" +
              (f" - validation-search results: {', '.join(str(c['val_mae']) for c in p['search'])}" if 'search' in p else "") + "\n"
              for name, p in params.items()) + f"""
- Hybrid LSTM representation: final hidden state of the last LSTM layer ({hy['lstm_hidden_size']} values), concatenated with the {len(fi)} engineered features and passed to XGBoost. It is not an average of predictions.

Engineered features (RF, XGBoost, hybrid):
""" + "".join(f"- **{k}** - {v[0]}. *Why:* {v[1]}. *How:* {v[2]}.\n" for k, v in FEATURE_DOCS.items()) + f"""
## 5. Evaluation Metrics
- MAE = mean|y - ŷ|; RMSE = sqrt(mean(y - ŷ)²) - computed on **all** test targets.
- MAPE = mean|(y - ŷ)/y| x 100 computed **only where y >= {cfg['evaluation']['mape_min_flow']}** vehicles / 5 min (safe MAPE: avoids division by near-zero flow; never inf/NaN).
- Training time (final fit; hybrid includes its LSTM training + feature extraction), inference time over the whole test set.

## 6. Final Results (locked test set)
{md(comparison[['Model', 'MAE', 'RMSE', 'MAPE']])}
Validation metrics of the frozen models (for reference; not used for ranking):

{md(val)}
## 7. Model Comparison
""" + "".join(f"- {line}\n" for line in answer[:3]) + f"""
Figures: `fig1a-f_actual_vs_predicted_*.png` (one per model), `fig2_mae.png`, `fig3_rmse.png`, `fig4_mape.png`.

## 8. Error Analysis
Regimes (thresholds from **training** targets):
""" + "".join(f"- **{k}**: {v}\n" for k, v in defs.items()) + f"""
MAE by regime:

{md(reg_mae)}
Largest errors (top 1% absolute error per model): share that falls in the sudden-change regime vs that regime's share of all test samples:

{md(big_sc, '{:.2f}')}
Share of top-1% errors in the peak regime:

{md(big_peak, '{:.2f}')}
Feature importance (how much the model uses a feature, not causal evidence): Random Forest top-5 = {top('random_forest_impurity')}; XGBoost (gain) top-5 = {top('xgboost_gain_share')}.

Figures: `fig7_residual_distribution.png`, `fig8_error_by_regime.png`, `fig9_error_by_hour.png`, `fig10/11_feature_importance_*.png`.

## 9. Computational Comparison
{md(comparison[['Model', 'Training time (s)', 'Inference time (s)', 'Inference per sample (ms)', 'Parameters', 'Model size (MB)']], '{:.4g}')}
Figures: `fig5_training_time.png`, `fig6_inference_time.png`, `fig12_training_curves.png`.

## 10. Hybrid Model Analysis
- {answer[3]}
- {answer[4]}
- Hybrid time breakdown (s): `{json.dumps({k: round(v, 1) for k, v in params[HYBRID]['time_breakdown_s'].items()})}`.

## 11. Research Findings
""" + "".join(f"- {line}\n" for line in answer) + """
These findings apply to this dataset, horizon and setup; they are not a general ranking of the methods.

## 12. Limitations
- One dataset (PeMSD4, Bay Area, 59 days) and one horizon (15 min); generalisation to other cities/seasons is untested.
- Only flow history and calendar time are used - no weather, incidents, events or road-network (graph) structure.
- Small validation grid; candidates compared on the first 40 sensors to keep the search affordable.
- A single run per model (one seed); GPU (MPS) training is not bit-for-bit deterministic, so no confidence intervals are reported.
- Samples whose forecast origin reading was missing are excluded, so outage periods are not evaluated.

## 13. Reproducibility
- `python run_pipeline.py --mode final` reproduces the experiment; `python -m src.reporting` rebuilds this report from the saved files.
- Environment: Python """ + meta["python"] + f""", {meta['os']}; libraries: `{json.dumps(meta['libraries'])}`.
- Full configuration and seed: `results/run_metadata.json`; per-experiment record: `results/reports/experiment_{meta['experiment_id']}.json`.
"""
    (results / "reports" / "final_results.md").write_text(text)
    return answer


if __name__ == "__main__":
    args = sys.argv[1:]
    for line in build_reports(*(args or ["results", "trained_models"])):
        print("*", line)
