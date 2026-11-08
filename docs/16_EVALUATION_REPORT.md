# 16 — Evaluation Report

All numbers in this report are read from the committed files [`reports/metrics_baseline.json`](../reports/metrics_baseline.json), [`reports/metrics_primary.json`](../reports/metrics_primary.json), [`reports/model_comparison.json`](../reports/model_comparison.json), [`reports/residuals_by_rul_bucket.json`](../reports/residuals_by_rul_bucket.json) and [`models/model_metadata.json`](../models/model_metadata.json); the document was generated from them in Milestone 7. The served model is `primary-20261108-f4c86df`.

## 1. Dataset and protocol

| Item | Value |
|---|---|
| Data | NASA C-MAPSS **FD001**: 100 training engines run to failure, 100 test engines truncated before failure, one operating condition, one fault mode (04) |
| Data identity | train SHA-256 `963b5e22825b34d8…`, test `3cda7109ce17bafb…`, RUL `a19c8ec94931949d…` (04 §3) |
| Target | `RUL = max_cycle − cycle`, capped at **125** (ADR-004) |
| Features | `cycle` + raw value, trailing mean and std over a **30-cycle** window for each of 14 kept sensors = 43 features; 7 near-constant sensors and the 3 settings dropped (ADR-005) |
| Training rows | 17,731 (rows with a full 30-cycle window) |
| Validation | **5-fold GroupKFold by engine** on the training set: no engine is in both a training and a validation fold (ADR-008). CV labels are capped. |
| Final evaluation | Official test set: one prediction per engine at its **last observed cycle** (100 engines) vs `RUL_FD001`. **Unclipped** truth is primary; truth clipped at 125 is reported for comparability with literature. |
| Post-processing | Predictions clamped to [0, 125] everywhere, in CV, test evaluation and the API alike (ADR-027) |
| Seed | 42; repeat runs give identical metrics and a byte-identical artifact (05 §7a; M7 fresh clone) |

## 2. Baseline vs primary

| Model | Version | CV RMSE (mean ± std) | CV MAE (mean ± std) | Test RMSE | Test MAE | NASA score | Test RMSE / MAE, truth clipped |
|---|---|---|---|---|---|---|---|
| Baseline — Ridge(alpha=1.0) | `ridge-20261108T150138Z` | 19.152 ± 1.182 | 15.112 ± 1.156 | **19.144** | **15.347** | 590.6 | 17.845 / 14.277 |
| Primary — HistGradientBoosting **(served)** | `hgb-20261108T150441Z` | 16.559 ± 0.695 | 11.843 ± 0.676 | **16.610** | **12.567** | 441.5 | 15.333 / 11.497 |

On the official test set the primary model reduces RMSE by 2.53 cycles (13 %) and MAE by 2.78 cycles (18 %). CV and test agree closely for both models: primary CV RMSE 16.559 vs test 16.610; baseline 19.152 vs 19.144. There is no sign of leakage or of overfitting to the validation protocol.

## 3. Selection rule and outcome (ADR-024)

The served model is the candidate with the lowest 5-fold GroupKFold **CV** RMSE (tie-break CV MAE, then the simpler model). Test metrics are reported but do not drive the choice, which keeps the official test set a held-out report.

- CV ranking: primary → baseline; test ranking: primary → baseline; rankings agree: **True**.
- Recorded reason: *Selected primary because its 5-fold GroupKFold CV RMSE (16.559) is lower than baseline's (19.152) (ADR-024). Official test metrics are reported but do not drive the choice. The official test ranking agrees with the CV ranking.*

## 4. Primary-model search (ADR-026)

One bounded search, 12 configurations, scored by cv_rmse_mean (tie-break cv_mae_mean), with `early_stopping=False` and `random_state=42`. No second round was run.

| learning_rate | max_iter | max_leaf_nodes | l2_regularization | CV RMSE (mean ± std) | CV MAE |
|---|---|---|---|---|---|
| 0.1 | 100 | 15 | 0.0 | 16.788 ± 0.614 | 11.962 |
| 0.1 | 100 | 15 | 1.0 | 16.717 ± 0.679 | 11.915 |
| 0.1 | 100 | 31 | 0.0 | 16.740 ± 0.636 | 11.808 |
| 0.1 | 100 | 31 | 1.0 | 16.717 ± 0.653 | 11.826 |
| 0.05 | 200 | 15 | 0.0 | **16.559 ± 0.695** | 11.843 |
| 0.05 | 200 | 15 | 1.0 | 16.688 ± 0.686 | 11.958 |
| 0.05 | 200 | 31 | 0.0 | 16.828 ± 0.679 | 11.840 |
| 0.05 | 200 | 31 | 1.0 | 16.726 ± 0.646 | 11.789 |
| 0.03 | 400 | 15 | 0.0 | 16.642 ± 0.634 | 11.849 |
| 0.03 | 400 | 15 | 1.0 | 16.629 ± 0.652 | 11.843 |
| 0.03 | 400 | 31 | 0.0 | 16.683 ± 0.611 | 11.742 |
| 0.03 | 400 | 31 | 1.0 | 16.798 ± 0.634 | 11.815 |

Selected: `{'learning_rate': 0.05, 'max_iter': 200, 'max_leaf_nodes': 15, 'l2_regularization': 0.0}`. The spread across all configurations is small (CV RMSE 16.559–16.828, i.e. 0.270, less than one fold standard deviation). The gain comes from moving from a linear model to boosted trees, not from tuning.

## 5. NASA score (secondary) — ADR-009 caveat

The PHM08 asymmetric score (lower is better, late predictions penalised more) is **441.5** for the primary model and **590.6** for the baseline. Caveat: the paper's printed equation pairs its coefficients opposite to its own text and figure. This project follows the stated intent (late: `exp(d/10) − 1`, early: `exp(−d/13) − 1`). Scores computed with the literal printed equation would differ, so compare them with published numbers only after checking which convention those used. The score is never used for selection.

## 6. Residual analysis of the served model (official test set)

Error = predicted − true (positive = late, i.e. optimistic). Buckets use the **unclipped** true RUL. Overall: n = 100, MAE 12.567, RMSE 16.610, mean signed error -0.486.

| True RUL bucket | Engines | MAE | RMSE | Mean signed error |
|---|---|---|---|---|
| 0-20 | 16 | 3.430 | 5.821 | +1.618 |
| 21-60 | 23 | 8.820 | 12.476 | +5.380 |
| 61-125 | 50 | 14.597 | 17.586 | +1.418 |
| >125 | 11 | 24.464 | 26.972 | -24.464 |

Reading the table:

- **Near failure (0–20):** the most accurate region (MAE 3.430), where accuracy matters most for maintenance.
- **21–60:** a mean signed error of +5.380 means predictions are on average *later* than reality, the optimistic and riskier direction for a maintenance planner. This is the main weakness to address before any operational use.
- **61–125:** largest group (50 engines), MAE 14.597, small positive bias.
- **> 125:** every engine is under-predicted (mean signed error -24.464 = −MAE), as designed: the model cannot predict above the cap of 125, so long-RUL engines look younger than they are. These 11 engines (11 % of the test set) contribute 29% of the total squared error.

## 7. Limitations

- **Single operating condition / fault mode:** FD001 only; not valid for FD002–FD004 or other engines.
- **Simulated data:** no claim about real turbofans.
- **Cap saturation:** predictions are bounded at 125, so early-life engines are under-predicted (§6, > 125 bucket); the demo engine 1 (true RUL 112) predicts exactly 125.
- **Train-set optimism:** dashboard views of *training* engines (e.g. train engine 1) are in-sample and look better than out-of-sample performance; only the official test set and CV are honest estimates.
- **Optimistic bias at 21–60 cycles** (§6).
- **No uncertainty estimate:** point predictions only; health bands are illustrative thresholds.
- **Selection bias:** small (best of 12 configurations on CV; test-set agreement suggests it is minor).
- **Pickle tied to the scikit-learn pin:** `model.joblib` requires scikit-learn 1.9.1; a test fails on a version mismatch and retraining reproduces the artifact byte-identically.

## 8. Optional extensions — NOT IMPLEMENTED

None of the following is part of the MVP. Each would need its own change record and milestone (15).

| Extension (NOT IMPLEMENTED) | What it would need |
|---|---|
| FD002–FD004 (multiple conditions / fault modes) | Operating-condition normalisation (per-regime sensor scaling), regime-aware features, re-validated data contract and checksums, larger evaluation |
| Anomaly detection | A separate unsupervised model (e.g. reconstruction error on healthy early cycles), its own threshold calibration, evaluation protocol and API endpoint |
| Sequence models (LSTM/CNN/Transformer) | A deep-learning dependency (ADR), GPU-optional training, sequence windows in the shared feature path, longer training |
| Prediction intervals | Quantile gradient boosting or conformal prediction on CV residuals; API field for the interval |
| Bias correction for the 21–60 range | Residual analysis on CV folds, asymmetric loss or post-hoc calibration, then re-selection |
| Drift monitoring / retraining | Logging of inputs, distribution checks against `training_ranges`, a retraining trigger and model registry |
| Cloud deployment | Container registry, managed runtime, auth/TLS, secrets management, cost controls |
