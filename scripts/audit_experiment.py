"""Final research audit: re-checks the data, leakage rules, saved models, results and the app.

    python scripts/audit_experiment.py
    python scripts/audit_experiment.py --results-dir results_debug --models-dir trained_models_debug
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.components.predictor import MODEL_FILES as APP_FILES, MODEL_NAMES, Forecaster  # noqa: E402
from src.data.loader import load_raw  # noqa: E402
from src.evaluation.metrics import all_metrics  # noqa: E402
from src.preprocessing.prepare import prepare, train_readings  # noqa: E402
from src.preprocessing.scaler import FlowScaler  # noqa: E402
from src.reporting import MODEL_FILES  # noqa: E402
from src.utils import load_config  # noqa: E402

FIGURES = ["fig1a_actual_vs_predicted_historical_average.png", "fig1b_actual_vs_predicted_random_forest.png",
           "fig1c_actual_vs_predicted_xgboost.png", "fig1d_actual_vs_predicted_lstm.png",
           "fig1e_actual_vs_predicted_gru.png", "fig1f_actual_vs_predicted_hybrid_lstm_xgboost.png",
           "fig2_mae.png", "fig3_rmse.png", "fig4_mape.png", "fig5_training_time.png", "fig6_inference_time.png",
           "fig7_residual_distribution.png", "fig8_error_by_regime.png"]


class Audit:
    def __init__(self):
        self.rows = []

    def check(self, label, fn):
        try:
            detail = fn()
            self.rows.append((label, True, detail or ""))
        except AssertionError as e:
            self.rows.append((label, False, str(e)))
        except Exception as e:
            self.rows.append((label, False, f"{type(e).__name__}: {e}"))

    def report(self):
        print("=" * 60 + "\nFINAL RESEARCH AUDIT\n" + "=" * 60)
        for label, ok, detail in self.rows:
            print(f"{label} {'.' * (24 - len(label))} {'PASS' if ok else 'FAIL'}   {detail}")
        ok = all(r[1] for r in self.rows)
        print(f"\nOVERALL STATUS: {'PASS' if ok else 'FAIL'}")
        return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--models-dir", default="trained_models")
    ap.add_argument("--skip-app", action="store_true", help="skip the (slower) Streamlit run")
    a = ap.parse_args()
    R, M = Path(a.results_dir), Path(a.models_dir)
    cfg = load_config()
    meta = json.loads((R / "run_metadata.json").read_text())
    debug = meta["mode"] == "debug"
    audit = Audit()
    state = {}

    def dataset():
        data = load_raw(cfg["paths"]["raw_data"])
        flow = data[:, :, cfg["dataset"]["flow_feature_index"]]
        assert data.ndim == 3 and data.shape[2] == 3, f"unexpected shape {data.shape}"
        assert not np.isnan(flow).any(), "NaN in flow"
        assert (flow >= 0).all(), "negative flow"
        rep = json.loads((R / "reports" / "dataset_report.json").read_text())
        assert rep["array_shape"] == list(data.shape), "dataset report does not match the file"
        return f"shape {data.shape}, zero readings {(flow == 0).mean() * 100:.2f}%"

    def preprocessing():
        flow, filled, observed, info, bounds, splits = prepare(cfg, cfg["debug"]["max_sensors"] if debug else None)
        state.update(filled=filled, observed=observed, bounds=bounds, splits=splits)
        tr, va, te = splits["train"], splits["val"], splits["test"]
        assert tr["time"].max() < va["time"].min() and va["time"].max() < te["time"].min(), "splits overlap in time"
        for name, (s, e) in bounds.items():
            sp = splits[name]
            assert (sp["time"] - cfg["forecast"]["sequence_length"] - cfg["forecast"]["horizon_steps"] + 1 >= s).all() \
                and (sp["time"] < e).all(), f"{name} windows cross the split boundary"
        return f"chronological, samples train/val/test = {len(tr['y']):,}/{len(va['y']):,}/{len(te['y']):,}"

    def leakage():
        filled, observed, bounds, splits = (state[k] for k in ("filled", "observed", "bounds", "splits"))
        h = cfg["forecast"]["horizon_steps"]
        # scaler fitted on training readings only
        saved = FlowScaler.load(M / "preprocessing" / "scaler.pkl")
        train_fit = FlowScaler().fit(train_readings(filled, observed, bounds))
        all_fit = FlowScaler().fit(np.where(observed, filled, np.nan))
        assert np.isclose(saved.mean, train_fit.mean) and np.isclose(saved.std, train_fit.std), "scaler != training-only fit"
        assert not np.isclose(saved.mean, all_fit.mean), "scaler matches an all-data fit"
        for name, sp in splits.items():
            origin = sp["time"] - h
            # the window ends at t = target - 3 and equals the stored readings (no future values inside)
            assert np.allclose(sp["X"][:, -1], filled[origin, sp["sensor"]]), f"{name}: last input is not flow(t)"
            assert observed[origin, sp["sensor"]].all(), f"{name}: origin reading interpolated (would use future)"
            assert observed[sp["time"], sp["sensor"]].all(), f"{name}: interpolated target"
            # engineered features derive only from the window + calendar
            F, X = sp["F"], sp["X"]
            assert np.allclose(F[:, :X.shape[1]], X) and np.allclose(F[:, X.shape[1]], X.mean(1), atol=1e-3), \
                f"{name}: features not computed from the input window"
        th = json.loads((R / "metrics" / "error_analysis_thresholds.json").read_text())["thresholds"]
        assert np.isclose(th["peak_from"], np.percentile(splits["train"]["y"], 90)), "regime thresholds not from training"
        params = json.loads((R / "metrics" / "best_parameters.json").read_text())
        assert all("search" in params[m] for m in MODEL_NAMES[1:]), "a model has no validation-search record"
        return "train-only scaler, causal windows/features, real targets, validation-only selection"

    table = pd.read_csv(R / "metrics" / "model_metrics.csv").set_index("Model")

    def model_check(name):
        def fn():
            missing = [f for f in APP_FILES[name] if not (M / f).exists()]
            assert not missing, f"missing files {missing}"
            df = pd.read_csv(R / "predictions" / f"{MODEL_FILES[name]}.csv")
            test = state["splits"]["test"]
            assert len(df) == len(test["y"]), "prediction rows != test samples"
            assert df["predicted_flow"].notna().all(), "NaN predictions"
            m = all_metrics(df["actual_flow"], df["predicted_flow"], cfg["evaluation"]["mape_min_flow"])
            row = table.loc[name]
            for k in ("MAE", "RMSE", "MAPE"):
                assert np.isfinite(row[k]) and abs(m[k] - row[k]) < 1e-2, f"{k} in metrics file != recomputed ({m[k]:.4f})"
            return f"MAE {row['MAE']:.3f}  RMSE {row['RMSE']:.3f}  MAPE {row['MAPE']:.2f}%"
        return fn

    def evaluation():
        cmp_ = pd.read_csv(R / "metrics" / "final_model_comparison.csv")
        assert list(cmp_["Model"]) == MODEL_NAMES, "comparison table does not list the six models"
        assert cmp_[["MAE", "RMSE", "MAPE", "Training time (s)", "Inference time (s)"]].notna().all().all()
        missing = [f for f in FIGURES if not (R / "figures" / f).exists()]
        assert not missing, f"missing figures {missing}"
        for f in ["reports/final_results.md", "reports/final_results.json", "reports/dataset_report.txt",
                  "metrics/error_analysis_by_regime.csv", "metrics/feature_importance.csv"]:
            assert (R / f).exists(), f"missing {f}"
        return f"6 metric records, {len(list((R / 'figures').glob('*.png')))} figures, reports present"

    def saved_models():
        fc = Forecaster(cfg, M)
        for n in MODEL_NAMES:
            fc.model(n)
        return "6 models + scaler load from disk"

    def inference():
        import test_inference
        res = test_inference.run(models_dir=str(M), results_dir=str(R), verbose=False)
        assert all(r["ok"] for r in res), f"failed: {[r['model'] for r in res if not r['ok']]}"
        return "saved models reproduce stored test predictions (max diff " + \
               f"{max(abs(r['pred'] - r['stored']) for r in res if r['stored'] is not None):.4f})"

    def streamlit_app():
        from streamlit.testing.v1 import AppTest
        os.environ["TRAFFIC_MODELS_DIR"], os.environ["TRAFFIC_RESULTS_DIR"] = str(M), str(R)
        at = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=300).run()
        at.sidebar.button[0].click().run()
        assert not at.exception, f"app raised: {at.exception}"
        assert any("Predicted traffic flow" in m.label for m in at.metric), "no forecast shown"
        return "app starts, forecast generated, no errors"

    def configuration():
        assert "config" in meta and meta["random_seed"] == cfg["project"]["random_seed"], "config/seed not recorded"
        assert meta["mode"] == "final" or debug
        return f"experiment {meta['experiment_id']}, seed {meta['random_seed']}, mode {meta['mode'].upper()}"

    audit.check("Dataset", dataset)
    audit.check("Preprocessing", preprocessing)
    audit.check("Leakage Check", leakage)
    for n in MODEL_NAMES:
        audit.check("Hybrid" if n.startswith("Hybrid") else n, model_check(n))
    audit.check("Evaluation", evaluation)
    audit.check("Saved Models", saved_models)
    audit.check("Inference", inference)
    if not a.skip_app:
        audit.check("Streamlit", streamlit_app)
    audit.check("Configuration", configuration)
    if debug:
        print("NOTE: auditing DEBUG outputs - not research results.")
    sys.exit(0 if audit.report() else 1)


if __name__ == "__main__":
    main()
