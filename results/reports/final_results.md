# Final Experimental Results

Experiment `20260927_202736_final` · generated from the files in `results/` · run on 2026-09-27T21:00:48

## 1. Dataset
| Property | Measured value |
|---|---|
| File | `data/raw/pems04.npz` (32.96 MB) |
| Array shape | [16992, 307, 3] (time steps x sensors x channels: flow, occupancy, speed) |
| Sensors used | 307 |
| Interval | 5 min - 16992 steps / 288 steps per day = 59.00 days (2018-01-01 onward); a whole number of days, consistent with the documented 5-minute interval |
| NaN / inf / negative | 0 / 0 / 0 |
| Zero readings (treated as missing) | 82,935 (1.59%) |
| Flow min / max / mean / median / std | 0 / 919 / 211.70 / 180 / 158.07 |

## 2. Experimental Configuration
- Chronological split 70% / 15% / 15% of the time axis (steps (0, 11894), (11894, 14443), (14443, 16992)); no shuffling.
- Samples: train 3,569,218 · validation 763,134 · test 765,921.
- Missing values: gaps of <= 12 steps linearly interpolated; longer gaps dropped (46,960 readings interpolated, 35,975 left missing).
- Leakage rules: the reading at the forecast origin t must be real (so interpolation never uses readings after t); targets are real readings only; windows never cross split boundaries; the scaler is fitted on training readings only; model selection and early stopping use validation only; the test set was evaluated once, after all models were saved.
- One global model per algorithm across all sensors (sensor ID is not a feature). Seed 42; device `mps`.

## 3. Forecasting Task
Input: the previous 12 readings (60 minutes, t-11 ... t) → target: traffic flow at t+3 (15 minutes ahead). Only this horizon is in the comparison.

## 4. Model Configurations (selected on the validation set)
- **Historical Average**: `"weekday x 5-min slot x sensor mean"`
- **Random Forest**: `{"max_depth": 20, "min_samples_leaf": 5}` - validation-search results: 22.4259, 22.3504, 21.9226, 21.8972
- **XGBoost**: `{"max_depth": 8, "learning_rate": 0.05, "best_iteration": 936}` - validation-search results: 21.4414, 21.4627, 21.3355, 21.3965
- **LSTM**: `{"hidden_size": 32, "num_layers": 2}` - validation-search results: 23.6695, 23.6467, 23.71, 23.6646
- **GRU**: `{"hidden_size": 64, "num_layers": 2}` - validation-search results: 23.6262, 23.6051, 23.6136, 23.422
- **Hybrid LSTM-XGBoost**: `{"max_depth": 8, "learning_rate": 0.05, "best_iteration": 721}` - validation-search results: 21.3427, 21.388, 21.2326, 21.2506

- Hybrid LSTM representation: final hidden state of the last LSTM layer (32 values), concatenated with the 19 engineered features and passed to XGBoost. It is not an average of predictions.

Engineered features (RF, XGBoost, hybrid):
- **lag_1 ... lag_12** - Flow 1..12 steps before the forecast origin (lag_1 = time t). *Why:* Recent traffic is the strongest signal of near-future traffic. *How:* The 12 values of the input window.
- **rolling_mean_60m** - Average flow over the last hour. *Why:* Captures the current traffic level. *How:* Mean of the 12 inputs.
- **rolling_std_60m** - Variability over the last hour. *Why:* Distinguishes stable from fluctuating traffic. *How:* Std of the 12 inputs.
- **recent_mean_15m** - Average flow over the last 15 minutes. *Why:* Reacts faster to change than the 60-min mean. *How:* Mean of the last 3 inputs.
- **recent_std_15m** - Variability over the last 15 minutes. *Why:* Signals sudden instability. *How:* Std of the last 3 inputs.
- **last_change** - Most recent 5-minute change. *Why:* Gives the direction of the trend. *How:* flow(t) - flow(t-1).
- **hour_of_day** - Hour of the forecast target time (0-23.92). *Why:* Traffic follows a daily cycle (rush hours). *How:* (target step mod 288) x 5 min / 60.
- **day_of_week** - Weekday of the target time (0 = Monday). *Why:* Weekday and weekend patterns differ. *How:* Derived from the dataset start date 2018-01-01.

## 5. Evaluation Metrics
- MAE = mean|y - ŷ|; RMSE = sqrt(mean(y - ŷ)²) - computed on **all** test targets.
- MAPE = mean|(y - ŷ)/y| x 100 computed **only where y >= 10** vehicles / 5 min (safe MAPE: avoids division by near-zero flow; never inf/NaN).
- Training time (final fit; hybrid includes its LSTM training + feature extraction), inference time over the whole test set.

## 6. Final Results (locked test set)
| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Historical Average | 24.469 | 39.675 | 13.583 |
| Random Forest | 20.349 | 32.438 | 11.260 |
| XGBoost | 19.979 | 31.844 | 11.183 |
| LSTM | 22.097 | 34.116 | 12.587 |
| GRU | 22.030 | 34.084 | 12.462 |
| Hybrid LSTM-XGBoost | 19.894 | 31.712 | 11.134 |

Validation metrics of the frozen models (for reference; not used for ranking):

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Historical Average | 25.850 | 41.164 | 14.249 |
| Random Forest | 19.653 | 31.300 | 11.197 |
| XGBoost | 19.333 | 30.747 | 11.121 |
| LSTM | 21.294 | 32.853 | 12.524 |
| GRU | 21.246 | 32.850 | 12.394 |
| Hybrid LSTM-XGBoost | 19.255 | 30.619 | 11.080 |

## 7. Model Comparison
- Under this experimental setup the test RMSE ranking was: Hybrid LSTM-XGBoost (31.712), XGBoost (31.844), Random Forest (32.438), GRU (34.084), LSTM (34.116), Historical Average (39.675).
- All learning models are compared with the Historical Average baseline (MAE 24.469). The measured MAE reduction relative to the baseline ranged from 9.69% to 18.70%.
- The tree models, which receive engineered features including time of day and weekday, measured MAE 20.349 (RF) and 19.979 (XGBoost). The sequence-only recurrent models measured 22.097 (LSTM) and 22.030 (GRU).

Figures: `fig1a-f_actual_vs_predicted_*.png` (one per model), `fig2_mae.png`, `fig3_rmse.png`, `fig4_mape.png`.

## 8. Error Analysis
Regimes (thresholds from **training** targets):
- **low**: actual flow < 84 (training 25th percentile)
- **normal**: 84 <= actual flow < 318 (training 25th-75th percentile)
- **high**: 318 <= actual flow < 440 (training 75th-90th percentile)
- **peak**: actual flow >= 440 (training 90th percentile)
- **sudden_change**: |flow(t+3) - flow(t)| >= 81 (training 95th percentile of 15-min change); overlaps the flow-level regimes

MAE by regime:

| Model | low | normal | high | peak | sudden_change |
|---|---:|---:|---:|---:|---:|
| Historical Average | 8.824 | 21.918 | 36.170 | 48.804 | 63.061 |
| Random Forest | 7.268 | 18.666 | 29.556 | 39.583 | 82.856 |
| XGBoost | 7.330 | 18.393 | 28.824 | 38.481 | 79.869 |
| LSTM | 8.436 | 20.749 | 31.690 | 40.464 | 87.878 |
| GRU | 8.179 | 20.621 | 32.021 | 40.463 | 87.936 |
| Hybrid LSTM-XGBoost | 7.288 | 18.331 | 28.694 | 38.280 | 79.501 |

Largest errors (top 1% absolute error per model): share that falls in the sudden-change regime vs that regime's share of all test samples:

| Model | Top-1% error threshold | Share of all test samples (%) | Share of top-1% errors (%) |
|---|---:|---:|---:|
| Historical Average | 147.33 | 5.40 | 37.00 |
| Random Forest | 120.37 | 5.40 | 90.12 |
| XGBoost | 118.33 | 5.40 | 89.06 |
| LSTM | 123.79 | 5.40 | 91.63 |
| GRU | 123.75 | 5.40 | 91.91 |
| Hybrid LSTM-XGBoost | 117.85 | 5.40 | 89.02 |

Share of top-1% errors in the peak regime:

| Model | Share of all test samples (%) | Share of top-1% errors (%) |
|---|---:|---:|
| Historical Average | 11.64 | 36.60 |
| Random Forest | 11.64 | 40.57 |
| XGBoost | 11.64 | 38.93 |
| LSTM | 11.64 | 38.79 |
| GRU | 11.64 | 38.98 |
| Hybrid LSTM-XGBoost | 11.64 | 38.99 |

Feature importance (how much the model uses a feature, not causal evidence): Random Forest top-5 = recent_mean_15m, hour_of_day, lag_1, lag_12, rolling_std_60m; XGBoost (gain) top-5 = recent_mean_15m, lag_1, lag_2, lag_4, rolling_mean_60m.

Figures: `fig7_residual_distribution.png`, `fig8_error_by_regime.png`, `fig9_error_by_hour.png`, `fig10/11_feature_importance_*.png`.

## 9. Computational Comparison
| Model | Training time (s) | Inference time (s) | Inference per sample (ms) | Parameters | Model size (MB) |
|---|---:|---:|---:|---:|---:|
| Historical Average | 0.06783 | 0.01155 | 1.508e-05 | nan | 2.476 |
| Random Forest | 131.3 | 1.959 | 0.002558 | nan | 99.81 |
| XGBoost | 67.08 | 1.805 | 0.002357 | nan | 22.95 |
| LSTM | 517.8 | 1.167 | 0.001523 | 1.296e+04 | 0.05777 |
| GRU | 760 | 3.093 | 0.004038 | 3.789e+04 | 0.1573 |
| Hybrid LSTM-XGBoost | 650.6 | 3.067 | 0.004004 | nan | 16.87 |

Figures: `fig5_training_time.png`, `fig6_inference_time.png`, `fig12_training_curves.png`.

## 10. Hybrid Model Analysis
- Hybrid LSTM-XGBoost vs the strongest single model (XGBoost): MAE +0.42%, RMSE +0.41%, MAPE +0.43% (positive = hybrid lower error): lower error on all three metrics. It came at 9.7x the training time and 1.7x the inference time.
- Inside the hybrid, LSTM-derived features account for 73.1% of the XGBoost gain (the engineered features account for the rest). This measures how much the model uses them, not causality.
- By regime: Lowest MAE per regime: low: Random Forest (7.27); normal: Hybrid LSTM-XGBoost (18.33); high: Hybrid LSTM-XGBoost (28.69); peak: Hybrid LSTM-XGBoost (38.28); sudden_change: Historical Average (63.06).
- Hybrid time breakdown (s): `{"lstm_training": 517.8, "feature_extraction": 9.8, "xgboost_training": 122.9}`.

## 11. Research Findings
- Under this experimental setup the test RMSE ranking was: Hybrid LSTM-XGBoost (31.712), XGBoost (31.844), Random Forest (32.438), GRU (34.084), LSTM (34.116), Historical Average (39.675).
- All learning models are compared with the Historical Average baseline (MAE 24.469). The measured MAE reduction relative to the baseline ranged from 9.69% to 18.70%.
- The tree models, which receive engineered features including time of day and weekday, measured MAE 20.349 (RF) and 19.979 (XGBoost). The sequence-only recurrent models measured 22.097 (LSTM) and 22.030 (GRU).
- Hybrid LSTM-XGBoost vs the strongest single model (XGBoost): MAE +0.42%, RMSE +0.41%, MAPE +0.43% (positive = hybrid lower error): lower error on all three metrics. It came at 9.7x the training time and 1.7x the inference time.
- Inside the hybrid, LSTM-derived features account for 73.1% of the XGBoost gain (the engineered features account for the rest). This measures how much the model uses them, not causality.
- Error analysis: all six models had their highest flow-regime MAE in the 'peak' regime; for every model the sudden-change MAE exceeded its worst flow-regime MAE.
- Lowest MAE per regime: low: Random Forest (7.27); normal: Hybrid LSTM-XGBoost (18.33); high: Hybrid LSTM-XGBoost (28.69); peak: Hybrid LSTM-XGBoost (38.28); sudden_change: Historical Average (63.06).

These findings apply to this dataset, horizon and setup; they are not a general ranking of the methods.

## 12. Limitations
- One dataset (PeMSD4, Bay Area, 59 days) and one horizon (15 min); generalisation to other cities/seasons is untested.
- Only flow history and calendar time are used - no weather, incidents, events or road-network (graph) structure.
- Small validation grid; candidates compared on the first 40 sensors to keep the search affordable.
- A single run per model (one seed); GPU (MPS) training is not bit-for-bit deterministic, so no confidence intervals are reported.
- Samples whose forecast origin reading was missing are excluded, so outage periods are not evaluated.

## 13. Reproducibility
- `python run_pipeline.py --mode final` reproduces the experiment; `python -m src.reporting` rebuilds this report from the saved files.
- Environment: Python 3.14.6, Darwin 25.6.0 (arm64); libraries: `{"numpy": "2.5.3", "pandas": "3.0.6", "scikit-learn": "1.9.1", "xgboost": "3.4.1", "torch": "2.14.0", "matplotlib": "3.11.2", "streamlit": "1.64.0", "pyyaml": "6.0.3"}`.
- Full configuration and seed: `results/run_metadata.json`; per-experiment record: `results/reports/experiment_20260927_202736_final.json`.
