# Demo Guide

The whole demo uses the **saved** models, so nothing is retrained. It takes about 10 minutes.

## 1. Activate environment
```bash
cd ~/Desktop/ML/traffic-forecasting
source .venv/bin/activate
```

## 2. Run tests
```bash
python -m pytest -q tests              # expect: 26 passed
python scripts/audit_experiment.py     # expect: OVERALL STATUS: PASS
python scripts/test_inference.py       # expect: OVERALL: 6/6 models PASS
```
What to say:
- *"The audit re-checks the dataset and the leakage rules: training-only scaler, causal windows, real targets,
  and validation-only model selection. It also checks that every model, prediction and figure exists."*
- *"The inference test loads each saved model and reproduces the exact prediction the experiment stored for the same test sample."*

## 3. Launch application
```bash
streamlit run app/app.py
```
Open http://localhost:8501 in the browser. It is a **historical-data forecast demonstration**, not live traffic.

## 4. Select sensor
Sidebar → **Sensor** (e.g. 0). "Recent traffic history" shows the last 3 hours. The **bold** part is the
60-minute window that the model actually receives.

## 5. Select date/time
Choose a date (only the unseen **test period** is offered) and a time, for example **08:00** for the morning peak
or **17:30** for the evening peak. The time is the last known reading t, and the forecast is for t + 15 min.

## 6. Select model
Start with **Hybrid LSTM-XGBoost**, then compare it with **Historical Average** and **XGBoost**.

## 7. Generate forecast
Press **Generate Forecast**. You will see:
- the predicted flow
- the actual flow recorded 15 minutes later
- the absolute error

Open **"All six models at this exact moment"** to compare all models on the same input.
If the sensor had an outage at that time, the app says so instead of guessing. The same rule was used in training.

## 8. Explain graph
- *Last hour of input → forecast*: the dashed line joins the last known reading to the forecast point, and ✕ is the actual value.
- *Actual vs predicted, whole day*: every 5-minute forecast for the selected day, with that day's MAE in the title.

## 9. Show model comparison
"Model performance" and "Model comparison" show MAE / RMSE / MAPE on the **locked test set** (all 307 sensors,
from `results/metrics/final_model_comparison.csv`). The best value in each column is highlighted.

## 10. Explain hybrid architecture
```
last 60 min (12 readings) ─► trained LSTM (frozen) ─► hidden state (learned temporal features)
                                                            + 19 engineered features (lags, rolling stats, hour, weekday)
                                                            ─► XGBoost ─► flow 15 min later
```
It is **not** an average of two predictions. The LSTM is only a feature extractor.

## 11. Show final research results
Open the **Research results** tab (every number is read from `results/`), or:
```bash
open results/reports/final_results.md
open results/figures
```
Answer "which model is best?" only from those files. The report states whether the hybrid improved, for each metric.

## Likely questions
- **Is there leakage?** No, and the audit checks it:
  - the split is chronological
  - the scaler is fitted on training data only
  - the reading at forecast time t must be real, so gap-filling never uses values after t
  - targets are real readings, and windows never cross split boundaries
  - tuning and early stopping use validation only, and the test set was evaluated once
- **Why MAPE only for flow ≥ 10?** Dividing by near-zero night-time flow gives huge, meaningless percentages.
- **Is this real-time?** No. It forecasts from historical PeMSD4 readings, which demonstrates the method.
- **Why one model for all sensors?** It is computationally practical and learns from 3.5 M training samples.
  HA is per sensor by design.
