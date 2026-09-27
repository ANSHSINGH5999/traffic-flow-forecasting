# An Enhanced Machine Learning Framework for Short-Term Traffic Flow Forecasting

An end-to-end, reproducible research system. It trains **six forecasting models** on the real **PeMSD4**
traffic dataset under one identical experimental setup, evaluates them on the same unseen test period,
saves the trained models, and serves them in a **Streamlit** demo app.

```
PeMSD4 → validate → clean → chronological split → 60-min windows → features
      → HA / Random Forest / XGBoost / LSTM / GRU / Hybrid LSTM-XGBoost
      → validation-based selection → frozen models → locked test set
      → MAE / RMSE / MAPE + time → error analysis → figures & reports → Streamlit
```

---

## 1. Problem statement
Short-term traffic-flow forecasts support signal control, routing and congestion management. Published models are
hard to compare because each paper uses a different dataset, horizon, preprocessing and test split.

## 2. Research objective
Develop and compare machine-learning and deep-learning models for 15-minute-ahead traffic-flow forecasting,
evaluating all of them under **one controlled setup** so that differences come from the algorithm alone.

## 3. Research question
Does a hybrid LSTM-XGBoost model, which combines learned temporal features with nonlinear tree regression,
forecast 15-minute-ahead traffic flow more accurately than HA, RF, XGBoost, LSTM and GRU under identical
conditions? What does it cost in computation? The hybrid is **not** assumed to be better. The test decides.

## 4. Dataset
PeMSD4 (Caltrans PeMS, San Francisco Bay Area). Values below were verified from the file by `src/data/validator.py`:

| Property | Verified value |
|---|---|
| Array shape | 16,992 time steps × 307 sensors × 3 channels (flow, occupancy, speed) |
| Interval | 5 minutes (16,992 / 288 = 59 whole days, 1 Jan – 28 Feb 2018) |
| Target | Traffic flow (channel 0), vehicles per 5 minutes |
| NaN / inf / negative | 0 / 0 / 0 |
| Zero readings | 82,935 (1.59%), treated as sensor failures |
| Flow range | 0 – 919, mean 211.7, median 180, std 158.1 |

Full report: `results/reports/dataset_report.txt`.

## 5. Dataset preparation
See [`data/README.md`](data/README.md). One file is needed: `data/raw/pems04.npz`.

## 6. Methodology

| Step | Decision |
|---|---|
| Cleaning | Flow = 0 is treated as missing (sensor failure) |
| Missing values | Gaps of ≤ 12 steps (1 h) are linearly interpolated in time. Longer gaps are left missing, and windows touching them are dropped |
| Targets | Only **real** readings are used as targets. Interpolated values may be inputs but are never scored |
| No future in inputs | The reading at the forecast origin t must be real. Every interpolated gap inside the window is then closed by time t, so no input uses a reading after t |
| Split | Chronological 70 / 15 / 15 on the time axis, never shuffled. Windows never cross a split boundary |
| Input window | 12 steps = 60 minutes (methodology default, configurable in `config.yaml`) |
| Horizon | t + 3 = 15 minutes. Only this horizon is in the primary comparison |
| Scaling | Standard scaler fitted on **training** readings only (used by LSTM/GRU/hybrid) |
| Sensor strategy | **One global model per algorithm**, trained on samples pooled from all 307 sensors. Sensor ID is **not** a feature (HA is the exception: it is per sensor by definition) |
| Selection | Small grid search per model, scored on the **validation** set. The test set is used once, after all models are frozen |

## 7. Architecture
```
data/raw/pems04.npz ─► src/data (loader, validator)
                      ─► src/preprocessing (cleaner, windowing, scaler)
                      ─► src/features (engineered features)
                      ─► src/training (search on validation → final fit)  ─► trained_models/
                      ─► src/evaluation (metrics, evaluator, error analysis) ─► results/
                      ─► src/visualization (figures)                       ─► results/figures/
trained_models/ + data ─► app/app.py (Streamlit, inference only)
```

## 8. Models

| # | Model | Input | Role |
|---|---|---|---|
| 1 | Historical Average | sensor, weekday, time slot | Baseline: mean training flow for the same sensor/weekday/5-min slot |
| 2 | Random Forest | engineered features | Classical nonlinear ensemble (bagging) |
| 3 | XGBoost | same engineered features | Boosted trees; only the learner changes vs RF |
| 4 | LSTM | raw 12-step sequence | Recurrent temporal learner |
| 5 | GRU | raw 12-step sequence | Lighter recurrent learner, same conditions |
| 6 | **Hybrid LSTM-XGBoost** | LSTM hidden state + engineered features | Proposed model |

LSTM and GRU share one implementation (`src/models/recurrent.py`). Only the recurrent cell differs,
which keeps their comparison fair.

## 9. Hybrid architecture
```
12-step sequence ─► trained LSTM (frozen) ─► final hidden state (temporal features)
                                                   │
engineered features ───────────────────────────────┤ concatenate
                                                   ▼
                                               XGBoost ─► flow at t+3
```
- **What:** the LSTM is used as a *feature extractor*, not as a second predictor to average with.
- **Why:** traffic has sequential dependencies that are hard to hand-engineer, while trees handle nonlinear
  interactions and the time-of-day/weekday context well.
- **How:** Model 4 (trained on the training set only) is frozen. Its last-layer hidden state is computed for
  train/val/test windows and concatenated with the 19 engineered features. XGBoost is trained on the
  training rows, with early stopping and parameter choice on validation rows.
- **Cost accounting:** hybrid training time = LSTM training + feature extraction + XGBoost training.

## 10. Feature engineering (RF, XGBoost, hybrid)

| Feature | Meaning | Why | How |
|---|---|---|---|
| lag_12 … lag_1 | Flow 12…1 steps back (lag_1 = time t) | Recent traffic is the strongest signal | The 12 window values |
| rolling_mean_60m | Mean of the last hour | Current traffic level | mean(12 inputs) |
| rolling_std_60m | Variability of the last hour | Stable vs fluctuating traffic | std(12 inputs) |
| recent_mean_15m | Mean of the last 15 min | Reacts faster than the 60-min mean | mean(last 3) |
| recent_std_15m | Variability of the last 15 min | Sudden instability | std(last 3) |
| last_change | Latest 5-min change | Trend direction | flow(t) − flow(t−1) |
| hour_of_day | Hour of the target time | Daily cycle (rush hours) | (step mod 288) × 5 / 60 |
| day_of_week | Weekday of the target time | Weekday vs weekend patterns | from start date 2018-01-01 |

Time features are listed in the methodology and are known in advance (they come from the clock, not the data).

## 11. Training procedure
1. Build identical windows for every model.
2. HA statistics are computed from the training period only.
3. **Search:** each learner tries a small grid (RF: depth × min leaf; XGBoost: depth × learning rate;
   LSTM/GRU: hidden size × layers; hybrid: XGBoost depth × learning rate). Candidates are scored by
   **validation MAE** on the first 40 sensors, to keep the search affordable.
4. **Final fit:** the selected configuration is retrained on all training samples. XGBoost, LSTM, GRU
   and hybrid use early stopping on the full validation set.
5. Models are frozen and saved. Then every model predicts the locked test set **once**.

Selected values: `results/metrics/best_parameters.json`.

## 12. Evaluation metrics
- **MAE** = mean |y − ŷ|
- **RMSE** = √mean (y − ŷ)²
- **MAPE** = mean |y − ŷ| / y × 100, computed **only where actual flow ≥ 10** vehicles/5 min. Near-zero night
  values would otherwise inflate MAPE without meaning. MAE and RMSE use all test targets.
- Also reported: training time, inference time (whole test set and per sample), parameter count (neural nets),
  saved model size.

## 13. Results
Final experiment `20260927_202736_final`: all 307 sensors, locked test set of 765,921 samples (period 20 Feb 2018 03:35 – 28 Feb 2018 23:55).
Generated by the pipeline. Source: `results/metrics/final_model_comparison.csv`.

| Model | MAE | RMSE | MAPE (%) | Training time (s) | Inference time (s) |
|---|---:|---:|---:|---:|---:|
| Historical Average | 24.469 | 39.675 | 13.583 | 0.1 | 0.01 |
| Random Forest | 20.349 | 32.438 | 11.260 | 131.3 | 1.96 |
| XGBoost | 19.979 | 31.844 | 11.183 | 67.1 | 1.81 |
| LSTM | 22.097 | 34.116 | 12.587 | 517.8 | 1.17 |
| GRU | 22.030 | 34.084 | 12.462 | 760.0 | 3.09 |
| Hybrid LSTM-XGBoost | 19.894 | 31.712 | 11.134 | 650.6 | 3.07 |

Findings, generated from the result files (`results/metrics/summary.json`):
- Under this experimental setup the test RMSE ranking was: Hybrid LSTM-XGBoost (31.712), XGBoost (31.844), Random Forest (32.438), GRU (34.084), LSTM (34.116), Historical Average (39.675).
- All learning models are compared with the Historical Average baseline (MAE 24.469). The measured MAE reduction relative to the baseline ranged from 9.69% to 18.70%.
- The tree models, which receive engineered features including time of day and weekday, measured MAE 20.349 (RF) and 19.979 (XGBoost). The sequence-only recurrent models measured 22.097 (LSTM) and 22.030 (GRU).
- Hybrid LSTM-XGBoost vs the strongest single model (XGBoost): MAE +0.42%, RMSE +0.41%, MAPE +0.43% (positive = hybrid lower error): lower error on all three metrics. It came at 9.7x the training time and 1.7x the inference time.
- Inside the hybrid, LSTM-derived features account for 73.1% of the XGBoost gain (the engineered features account for the rest). This measures how much the model uses them, not causality.
- Error analysis: all six models had their highest flow-regime MAE in the 'peak' regime; for every model the sudden-change MAE exceeded its worst flow-regime MAE.
- Lowest MAE per regime: low: Random Forest (7.27); normal: Hybrid LSTM-XGBoost (18.33); high: Hybrid LSTM-XGBoost (28.69); peak: Hybrid LSTM-XGBoost (38.28); sudden_change: Historical Average (63.06).

These findings hold for this dataset, horizon and setup. They are not a general ranking of the methods. The hybrid's margin
over XGBoost is small (under 0.5% on each metric) and comes from a single run, without repeated seeds or significance testing.
Full report: [`results/reports/final_results.md`](results/reports/final_results.md).


## 14. Error analysis
All thresholds are computed from **training** targets, so they are data-driven and leak nothing from the test set:

| Regime | Rule |
|---|---|
| low | actual < 25th percentile |
| normal | 25th – 75th percentile |
| high | 75th – 90th percentile |
| peak | ≥ 90th percentile |
| sudden change | \|flow(t+3) − flow(t)\| ≥ 95th percentile of the 15-minute change (overlaps the others) |

Outputs: `results/metrics/error_analysis_by_regime.csv`, `error_by_hour.csv`, figures 7–9.
Feature importance (RF impurity decrease, XGBoost gain) is in `feature_importance.csv` and figures 10–11.
It shows how much each model *uses* a feature, not causality.

## 15. Installation
```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows
pip install -r requirements.txt
```
macOS only: XGBoost needs OpenMP, so run `brew install libomp`.

## 16. Dataset setup
```bash
mkdir -p data/raw
curl -L -o data/raw/pems04.npz https://github.com/Davidham3/ASTGCN/raw/master/data/PEMS04/pems04.npz
python -m src.data.validator       # prints and saves the dataset report
```

## 17. Training
```bash
python run_pipeline.py --mode debug   # ~1 min, 5 sensors, 3 epochs: checks the code. NOT research results.
python run_pipeline.py --mode final   # full experiment (33 min measured on an Apple M3, 16 GB)
python -m src.reporting              # rebuild reports/figures from the saved results (no training)
```
Debug output goes to `results_debug/` and `trained_models_debug/`, so it can never be mistaken for final results.

## 18. Evaluation
Evaluation runs as part of the pipeline, after all models are frozen. Outputs:
- `results/metrics/model_metrics.csv`: test MAE/RMSE/MAPE, times, sizes
- `results/metrics/validation_metrics.csv`
- `results/predictions/<model>.csv`: timestamp, sensor_id, actual, predicted, absolute/squared/percentage error
- `results/reports/final_results.md` and `.json`, and `experiment_<id>.json`

Verification:
```bash
python -m pytest -q tests          # 26 tests (synthetic data only tests code paths)
python scripts/audit_experiment.py # data, leakage, models, results, inference, app
python scripts/test_inference.py   # saved models reproduce stored test predictions
```

## 19. Streamlit app
```bash
streamlit run app/app.py
```
Pick a sensor, model, date and time from the **test period**, then press **Predict**. The app loads the saved
models from `trained_models/` (it never retrains) and shows:
- historical traffic
- the 15-minute forecast next to the actual value
- the whole day of actual vs predicted
- test metrics
- a six-model comparison
- a plain-language explanation of the model

## 20. Project structure
```
config.yaml            all settings
run_pipeline.py        orchestrates the experiment
src/data/              loader.py, validator.py
src/preprocessing/     cleaner.py, windowing.py, scaler.py
src/features/          feature_engineering.py
src/models/            historical_average.py, random_forest.py, xgboost_model.py, recurrent.py (LSTM+GRU), hybrid.py
src/training/          tuning.py, train_baselines.py, train_deep_learning.py, train_hybrid.py
src/evaluation/        metrics.py, evaluator.py, error_analysis.py
src/visualization/     plots.py
src/reporting.py       final_results.md / .json
app/                   app.py, components/predictor.py, components/descriptions.py
scripts/               audit_experiment.py, test_inference.py
tests/                 unit, model and app tests
trained_models/        saved models + preprocessing/scaler.pkl
results/               metrics/, predictions/, figures/, reports/, logs/, run_metadata.json
```

## 21. Trained model files
| Model | Files |
|---|---|
| Historical Average | `trained_models/historical_average/table.npy`, `config.json` |
| Random Forest | `trained_models/random_forest/model.pkl` |
| XGBoost | `trained_models/xgboost/model.json` |
| LSTM | `trained_models/lstm/model.pt`, `config.json` |
| GRU | `trained_models/gru/model.pt`, `config.json` |
| Hybrid | `trained_models/hybrid_lstm_xgboost/lstm.pt`, `xgboost.json`, `config.json` |
| Scaler | `trained_models/preprocessing/scaler.pkl` |

## 22. Reproducibility
- Seed 42 for Python, NumPy, scikit-learn, XGBoost and PyTorch.
- `results/run_metadata.json` records Python/OS/library versions, device, seed, full config and timestamp.
- Every setting lives in `config.yaml`.
- Neural nets train on Apple MPS (GPU). GPU kernels are not bit-for-bit deterministic, so a rerun can differ
  slightly in the last decimals.

## 23. Limitations
- One dataset, one region (Bay Area) and a 59-day period. Results may not generalise to other cities or seasons.
- Only flow history and calendar time are used. No weather, incidents, events or road-network (graph) information.
- During sudden-change periods, every learning model had a higher MAE than the Historical Average baseline (see error analysis).
- Single run per model (one seed), so no confidence intervals. MPS GPU training is not bit-for-bit deterministic.
- Samples whose forecast-origin reading was missing (sensor outage) are excluded, so outages are not evaluated.
- The hyperparameter search is intentionally small, and candidates were compared on 40 of 307 sensors.
- A single global model per algorithm. Per-sensor models or sensor embeddings were not explored.
- Single 15-minute horizon in the primary comparison.

## 24. Future work
- Additional horizons (30 / 60 min) as separate experiments.
- Spatial models using the sensor graph (e.g. STGCN / ASTGCN / Graph WaveNet from the literature survey).
- External factors (weather, incidents) and cross-dataset validation (PeMSD8, other cities).
- Sensor embeddings or per-sensor fine-tuning.
- Larger, validation-only hyperparameter search and repeated runs with different seeds for confidence intervals.
