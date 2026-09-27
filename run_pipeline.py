"""End-to-end experiment.

    python run_pipeline.py --mode debug   # 5 sensors, few epochs: checks the code (NOT research results)
    python run_pipeline.py --mode final   # full PeMSD4 experiment
"""
import argparse
import datetime as dt
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src import models  # noqa: F401  (sets the XGBoost/torch import order)
from src.data.validator import save_report, validate_dataset
from src.evaluation import error_analysis as ea
from src.evaluation.evaluator import evaluate, save_predictions
from src.evaluation.metrics import all_metrics
from src.features.feature_engineering import feature_names
from src.models import random_forest, recurrent, xgboost_model
from src.preprocessing.prepare import prepare, train_readings
from src.preprocessing.scaler import FlowScaler
from src.reporting import build_reports
from src.training.train_baselines import train_historical_average, train_random_forest, train_xgboost
from src.training.train_deep_learning import train_recurrent
from src.training.train_hybrid import train_hybrid
from src.utils import environment_info, get_device, get_logger, load_config, set_seed
from src.visualization import plots

MODEL_DIRS = {"Historical Average": "historical_average", "Random Forest": "random_forest", "XGBoost": "xgboost",
              "LSTM": "lstm", "GRU": "gru", "Hybrid LSTM-XGBoost": "hybrid_lstm_xgboost"}


def dir_size_mb(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["debug", "final"], default="final")
    mode = ap.parse_args().mode

    cfg = load_config()
    ds, fc, seed = cfg["dataset"], cfg["forecast"], cfg["project"]["random_seed"]
    debug = mode == "debug"
    results = Path(cfg["debug"]["results_dir"] if debug else cfg["paths"]["results_dir"])
    models_dir = Path(cfg["debug"]["models_dir"] if debug else cfg["paths"]["models_dir"])
    dirs = {k: results / k for k in ["metrics", "predictions", "figures", "reports", "logs"]}
    for d in [*dirs.values(), models_dir / "preprocessing"]:
        d.mkdir(parents=True, exist_ok=True)

    experiment_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{mode}"
    log = get_logger(dirs["logs"] / "pipeline.log")
    log.info(f"===== {'DEBUG MODE (not research results)' if debug else 'FINAL EXPERIMENT'}  id={experiment_id} =====")
    set_seed(seed)
    device = get_device()
    started = time.perf_counter()

    # 1. Validate dataset
    log.info("[1] Validating dataset")
    report, _ = validate_dataset(cfg)
    save_report(report, dirs["reports"])
    if not report["checks_passed"]:
        raise SystemExit("Dataset validation failed - see results/reports/dataset_report.txt")
    log.info(f"    shape {report['array_shape']}, zero readings {report['zero_pct']:.2f}%")

    # 2. Preprocess
    log.info("[2] Preprocessing")
    max_sensors = cfg["debug"]["max_sensors"] if debug else None
    # 3. Chronological split, windows, features, scaler (fit on training data only)
    flow, filled, observed, clean_info, bounds, splits = prepare(cfg, max_sensors)
    plots.dataset_figures(flow, observed, dirs["figures"], ds["steps_per_day"], ds["start_weekday"])
    log.info("[3] Split, windowing, features")
    scaler = FlowScaler().fit(train_readings(filled, observed, bounds))
    scaler.save(models_dir / "preprocessing" / "scaler.pkl")
    data_info = {"sensors": flow.shape[1], "time_steps": flow.shape[0], **clean_info,
                 "bounds": bounds, "samples": {k: len(v["y"]) for k, v in splits.items()}}
    log.info(f"    samples {data_info['samples']}  missing {clean_info['missing_pct']:.2f}%  device {device}")

    ctx = SimpleNamespace(cfg=cfg, seed=seed, log=log, device=device, scaler=scaler, bounds=bounds,
                          filled=filled, observed=observed, train=splits["train"], val=splits["val"],
                          tune_sensors=cfg["tuning"]["max_sensors"],
                          tune_epochs=cfg["debug"]["max_epochs"] if debug else cfg["tuning"]["max_epochs"],
                          max_epochs=cfg["debug"]["max_epochs"] if debug else None)

    # 4-9. Train (test set untouched)
    trained, train_times, params = {}, {}, {}

    log.info("[4] Historical Average")
    trained["Historical Average"], train_times["Historical Average"], params["Historical Average"] = train_historical_average(ctx)
    trained["Historical Average"].save(models_dir / "historical_average")

    log.info("[5] Random Forest (search on validation)")
    trained["Random Forest"], train_times["Random Forest"], params["Random Forest"] = train_random_forest(ctx)
    (models_dir / "random_forest").mkdir(exist_ok=True)
    random_forest.save(trained["Random Forest"], models_dir / "random_forest" / "model.pkl")

    log.info("[6] XGBoost (search on validation)")
    trained["XGBoost"], train_times["XGBoost"], params["XGBoost"] = train_xgboost(ctx)
    (models_dir / "xgboost").mkdir(exist_ok=True)
    xgboost_model.save(trained["XGBoost"], str(models_dir / "xgboost" / "model.json"))

    histories = {}
    for name, cell in [("LSTM", "lstm"), ("GRU", "gru")]:
        log.info(f"[{7 if cell == 'lstm' else 8}] {name} (search on validation, then early stopping)")
        trained[name], train_times[name], params[name] = train_recurrent(ctx, cell)
        histories[name] = params[name]["history"]
        recurrent.save(trained[name], params[name]["architecture"],
                       {"training": params[name]["training"], "history": histories[name],
                        "parameters": params[name]["parameters"]}, models_dir / cell)

    log.info("[9] Hybrid LSTM-XGBoost (frozen LSTM features + engineered features -> XGBoost)")
    hybrid, train_times["Hybrid LSTM-XGBoost"], params["Hybrid LSTM-XGBoost"] = train_hybrid(
        ctx, trained["LSTM"], train_times["LSTM"])
    trained["Hybrid LSTM-XGBoost"] = hybrid
    hybrid.save(models_dir / "hybrid_lstm_xgboost", models_dir / "lstm",
                {"xgboost": params["Hybrid LSTM-XGBoost"]["selected"]})

    predictors = {
        "Historical Average": lambda sp: trained["Historical Average"].predict(sp["sensor"], sp["time"]),
        "Random Forest": lambda sp: trained["Random Forest"].predict(sp["F"]),
        "XGBoost": lambda sp: trained["XGBoost"].predict(sp["F"]),
        "LSTM": lambda sp: scaler.inverse(recurrent.run_batched(trained["LSTM"], scaler.transform(sp["X"]), device=device)),
        "GRU": lambda sp: scaler.inverse(recurrent.run_batched(trained["GRU"], scaler.transform(sp["X"]), device=device)),
        "Hybrid LSTM-XGBoost": lambda sp: hybrid.predict(sp["X"], sp["F"]),
    }

    # 10-12. Frozen models: validation metrics, then ONE pass over the locked test set
    log.info("[10] Evaluating on validation and the locked test set")
    min_flow = cfg["evaluation"]["mape_min_flow"]
    test = splits["test"]
    rows, val_rows, preds = [], [], {}
    for name, predict in predictors.items():
        val_rows.append({"Model": name, **all_metrics(splits["val"]["y"], predict(splits["val"]), min_flow)})
        pred, m, infer = evaluate(predict, test, min_flow)
        preds[name] = pred
        size = dir_size_mb(models_dir / MODEL_DIRS[name])
        rows.append({"Model": name, **m, "Training time (s)": train_times[name], "Inference time (s)": infer,
                     "Inference per sample (ms)": infer / len(pred) * 1000,
                     "Parameters": params[name].get("parameters", ""), "Model size (MB)": size})
        log.info(f"    {name:20s} MAE {m['MAE']:.3f}  RMSE {m['RMSE']:.3f}  MAPE {m['MAPE']:.3f}%  "
                 f"train {train_times[name]:.1f}s  infer {infer:.2f}s")
        save_predictions(dirs["predictions"] / f"{MODEL_DIRS[name]}.csv", test, pred,
                         ds["start_date"], ds["interval_minutes"], min_flow)
    table, val_table = pd.DataFrame(rows), pd.DataFrame(val_rows)
    table.to_csv(dirs["metrics"] / "model_metrics.csv", index=False)
    val_table.to_csv(dirs["metrics"] / "validation_metrics.csv", index=False)
    (dirs["metrics"] / "best_parameters.json").write_text(json.dumps(
        {m: {k: v for k, v in p.items() if k != "history"} for m, p in params.items()}, indent=2, default=str))

    # 13. Error analysis + feature importance
    log.info("[11] Error analysis and feature importance")
    th = ea.regime_thresholds(splits["train"])
    regime_df = ea.analyse(preds, test, th)
    regime_df.to_csv(dirs["metrics"] / "error_analysis_by_regime.csv", index=False)
    hour_df = ea.error_by_hour(preds, test, ds["steps_per_day"])
    hour_df.to_csv(dirs["metrics"] / "error_by_hour.csv", index_label="hour")

    names = feature_names(fc["sequence_length"])
    rf_imp = trained["Random Forest"].feature_importances_
    gain = trained["XGBoost"].get_booster().get_score(importance_type="gain")
    xgb_imp = np.array([gain.get(f"f{i}", 0.0) for i in range(len(names))])
    xgb_imp = xgb_imp / xgb_imp.sum()
    pd.DataFrame({"feature": names, "random_forest_impurity": rf_imp, "xgboost_gain_share": xgb_imp}).to_csv(
        dirs["metrics"] / "feature_importance.csv", index=False)

    # 14. Figures
    log.info("[12] Figures")
    f = dirs["figures"]
    day = int(test["time"].min() // ds["steps_per_day"]) + 1
    day_label = (pd.Timestamp(ds["start_date"]) + pd.Timedelta(days=day)).strftime("%a %d %b %Y")
    plots.fig_actual_vs_predicted(preds, test, 0, day, ds["steps_per_day"], f / "fig1_actual_vs_predicted.png", day_label)
    plots.fig_metric(table, "MAE", "vehicles / 5 min", 2, f / "fig2_mae.png")
    plots.fig_metric(table, "RMSE", "vehicles / 5 min", 3, f / "fig3_rmse.png")
    plots.fig_metric(table, "MAPE", "%", 4, f / "fig4_mape.png")
    plots.fig_time(table, "Training time (s)", 5, "Training time", f / "fig5_training_time.png")
    plots.fig_time(table, "Inference time (s)", 6, f"Inference time for {len(test['y']):,} test samples",
                   f / "fig6_inference_time.png")
    plots.fig_residuals(preds, test, f / "fig7_residual_distribution.png")
    plots.fig_regimes(regime_df, ea.REGIME_ORDER, f / "fig8_error_by_regime.png")
    plots.fig_error_by_hour(hour_df, f / "fig9_error_by_hour.png")
    plots.fig_importance(names, rf_imp, "Figure 10 - Random Forest feature importance (impurity decrease)",
                         "Importance (share)", f / "fig10_feature_importance_rf.png")
    plots.fig_importance(names, xgb_imp, "Figure 11 - XGBoost feature importance (gain)",
                         "Gain (share)", f / "fig11_feature_importance_xgb.png")
    plots.fig_training_curves(histories, f / "fig12_training_curves.png")

    # 15. Reports (built from the saved result files, so they can be regenerated with `python -m src.reporting`)
    log.info("[13] Reports")
    th = {k: float(v) for k, v in th.items()}
    (dirs["metrics"] / "error_analysis_thresholds.json").write_text(json.dumps(
        {"thresholds": th, "definitions": ea.regime_definitions(th)}, indent=2))
    meta = {"experiment_id": experiment_id, "mode": mode, "timestamp": dt.datetime.now().isoformat(),
            "random_seed": seed, "device": device, **environment_info(), "config": cfg,
            "total_runtime_s": time.perf_counter() - started}
    (results / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    (dirs["reports"] / f"experiment_{experiment_id}.json").write_text(json.dumps({
        **meta, "dataset": data_info, "model_parameters": params, "training_time_s": train_times,
        "validation_metrics": val_table.to_dict(orient="records"),
        "test_metrics": table.to_dict(orient="records")}, indent=2, default=str))
    facts = build_reports(results, models_dir)

    log.info("\n" + table[["Model", "MAE", "RMSE", "MAPE", "Training time (s)", "Inference time (s)"]]
             .round(3).to_string(index=False))
    for fct in facts:
        log.info("  * " + fct)
    log.info(f"Done in {(time.perf_counter() - started) / 60:.1f} min. Reports: {dirs['reports'].resolve()}")


if __name__ == "__main__":
    main()
