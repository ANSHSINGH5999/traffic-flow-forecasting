# STGCN as a seventh model — experiment report

Branch `stgcn-experiment` · run 2026-09-27T23:53:39 · device `mps` · seed 42

## 1. What was added and what was left untouched
- **Added (new files only):** `experiments/stgcn/` (model, training script, results, trained model) and `data/raw/pems04_distance.csv`.
- **Untouched:** all six existing models, their predictions, metrics, reports, `src/`, `app/`, `site/`, `run_pipeline.py`, `config.yaml`,
  the `main` branch and both deployments. SHA-256 checksums of the 20 existing result/model files were recorded before the work (`baseline_checksums.sha256`) and re-verified afterwards with `shasum -c`
  and all 20 still match. The six-model numbers below are read from `results/metrics/final_model_comparison.csv`.

## 2. Protocol (identical to the six-model experiment)
- Same PeMSD4 flow, cleaning, chronological 70/15/15 split, 12-step (60 min) input and t+3 (15 min) target.
- **Same locked test set:** STGCN forecasts every sensor at once. Only the 765,921 existing test
  (time, sensor) pairs are scored, in the same order. The script asserts that they match the existing test samples exactly.
- Scaler is refitted on training readings and asserted equal to the saved one. Selection and early stopping use validation only, and the test set was used once.
- **One new design choice (graph input):** a neighbouring sensor's reading inside the window is used only if it is real, or an
  interpolated value whose gap had already closed by the forecast origin t. Anything else is set to 0 (the training mean after scaling) and
  flagged in a second input channel. No input depends on readings after t.

## 3. Graph
- Source: `data/raw/pems04_distance.csv (distributed with pems04.npz by the ASTGCN authors)`. It has 340 undirected road edges over 307 sensors (12 connected components).
- Ordering check: on training data, connected sensors' flow deviations from their daily profiles are more correlated than random pairs
  (median 0.629 vs 0.446, and 78.5% of edges lie above the random median). This supports the assumption that `distance.csv` ids match the npz column order.
  It is evidence, not proof, because no coordinates are available.
- Weights: w = exp(-(d/sigma)^2), sigma = std of edge distances; symmetric; road edges only (sigma = 257.5). Scaled normalised Laplacian, Chebyshev graph convolution (lambda_max = 2.000).

## 4. Model and selection
STGCN (Yu, Yin & Zhu, IJCAI 2018): 2 ST-Conv blocks (gated temporal conv Kt=3 → Chebyshev graph conv → gated temporal conv → LayerNorm),
then an output layer. Input `[batch, 2 channels, 12 steps, 307 sensors]`, output `[batch, 307]`. Fixed settings: `{"kt": 3, "dropout": 0.0, "learning_rate": 0.001, "batch_size": 32, "max_epochs": 50, "patience": 8}`.
The validation search used the same budget as LSTM/GRU (8 epochs per candidate):

| Channels | Chebyshev K | Validation MAE |
|---|---:|---:|
| [32, 8, 32] | 2 | 20.135 |
| [32, 8, 32] | 3 | 20.205 |
| [64, 16, 64] | 2 | 19.833 |
| [64, 16, 64] | 3 | 20.048 |

Selected: `{"channels": [64, 16, 64], "cheb_k": 2}`, 141,825 parameters. Early stopping kept epoch 23 (validation MAE 18.780) and stopped after 31 epochs.

## 5. Results on the locked test set

| Model | MAE | RMSE | MAPE (%) | Training time (s) | Inference time (s) |
|---|---:|---:|---:|---:|---:|
| Historical Average | 24.469 | 39.675 | 13.583 | 0.1 | 0.01 |
| Random Forest | 20.349 | 32.438 | 11.260 | 131.3 | 1.96 |
| XGBoost | 19.979 | 31.844 | 11.183 | 67.1 | 1.81 |
| LSTM | 22.097 | 34.116 | 12.587 | 517.8 | 1.17 |
| GRU | 22.030 | 34.084 | 12.462 | 760.0 | 3.09 |
| Hybrid LSTM-XGBoost | 19.894 | 31.712 | 11.134 | 650.6 | 3.07 |
| STGCN | 19.494 | 30.773 | 10.997 | 2399.2 | 3.54 |

STGCN training time excludes the 2053 s search (the other models are reported the same way).

Relative error reduction by STGCN (positive = STGCN lower error):
- vs XGBoost: MAE +2.43%, RMSE +3.36%, MAPE +1.66%; training 35.8x, inference 1.96x
- vs Hybrid LSTM-XGBoost: MAE +2.01%, RMSE +2.96%, MAPE +1.23%; training 3.7x, inference 1.16x
- vs LSTM (same sequence input, no graph): MAE +11.78%, RMSE +9.80%, MAPE +12.63%

## 6. Error analysis (MAE by regime; thresholds from training data, same as the main experiment)

| Model | low | normal | high | peak | sudden change |
|---|---:|---:|---:|---:|---:|
| Historical Average | 8.82 | 21.92 | 36.17 | 48.80 | 63.06 |
| Random Forest | 7.27 | 18.67 | 29.56 | 39.58 | 82.86 |
| XGBoost | 7.33 | 18.39 | 28.82 | 38.48 | 79.87 |
| LSTM | 8.44 | 20.75 | 31.69 | 40.46 | 87.88 |
| GRU | 8.18 | 20.62 | 32.02 | 40.46 | 87.94 |
| Hybrid LSTM-XGBoost | 7.29 | 18.33 | 28.69 | 38.28 | 79.50 |
| STGCN | 7.00 | 18.19 | 27.58 | 37.57 | 69.67 |

Lowest MAE per regime: low: STGCN; normal: STGCN; high: STGCN; peak: STGCN; sudden change: Historical Average.

## 7. Interpretation
- Under this setup, STGCN measured the lowest MAE, RMSE and MAPE of the seven models. It reduced error by about 2–3% relative to the hybrid and XGBoost.
  That margin is larger than the hybrid's margin over XGBoost (about 0.4%).
- STGCN had the lowest MAE in the low, normal, high and peak regimes. In sudden-change periods, its MAE (69.67) was clearly below
  every other learning model (79.50–87.94),
  but still above the Historical Average (63.06). This is consistent with, but does not prove, that
  information from neighbouring sensors helps anticipate changes the target sensor's own history does not yet show.
- The cost is substantial: 2399 s training (plus 2053 s search), versus 67 s for XGBoost.
- **Explicit spatial modelling helped in this experiment.** STGCN is the only model that sees other sensors. Every other model sees one sensor's history.

## 8. Caveats
- Single run and single seed. The validation MAE fluctuates by about ±1 between epochs (see `stgcn_training_curve.png`), so part of the
  2–3% margin could be run-to-run variance. Repeated seeds would be needed to claim a robust ranking.
- STGCN is also the only model with an availability-mask input channel and a graph-specific missing-data rule.
- The graph's geography cannot be verified directly (no sensor coordinates). Ordering is supported by the correlation check above.
- No comparison with published PEMS04 numbers: those use different horizons, splits and preprocessing.

## 9. Recommendation for the paper
Include STGCN as a **seventh model, reported as an additional spatial baseline**. It uses the same locked test set, and it is the only model
that addresses the paper's stated limitation (no road-network information).

State the main finding honestly:
- The proposed hybrid improved only marginally on XGBoost.
- A graph-based spatio-temporal model reduced error further under the same protocol, at a much higher training cost.

Do not present the hybrid as the best model overall.

## 10. Reproduce
```bash
git checkout stgcn-experiment
python -m experiments.stgcn.run_stgcn --mode debug   # 2 epochs, checks the code
python -m experiments.stgcn.run_stgcn --mode final   # ~75 min on an Apple M3 (search + training)
shasum -a 256 -c experiments/stgcn/baseline_checksums.sha256   # existing results unchanged
```
Outputs: `experiments/stgcn/results/metrics/` (seven_model_comparison.csv, error_analysis_by_regime_seven_models.csv, stgcn_run.json),
`results/figures/`, `results/predictions/stgcn.csv` (not committed, 30+ MB; regenerated by the script), and `trained_model/` (model.pt, config.json).
