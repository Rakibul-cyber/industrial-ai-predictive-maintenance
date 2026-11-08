# 18 — Model Card: FD001 Remaining-Useful-Life Model

Structure after Mitchell et al., *Model Cards for Model Reporting* (FAT* 2019). **Every number in a table below is generated from the file named on its `Source:` line** (checked by `tests/test_model_card.py`); prose numbers repeat table values. Generated 2026-11-09 for the committed artifacts.

## 1. Model details

| Field | Value |
|---|---|
| Model | `primary`: scikit-learn `HistGradientBoostingRegressor` in a `Pipeline` (feature selection step → estimator) |
| Served version | `primary-20261108-f4c86df` (candidate `hgb-20261108T150441Z`) |
| Trained / promoted | 2026-11-08T15:04:41Z / 2026-11-08T15:04:43Z (git `f4c86df`) |
| Hyperparameters | learning_rate 0.05, max_iter 200, max_leaf_nodes 15, l2_regularization 0.0; early stopping off; random_state 42 |
| Selection | `cv_rmse` over a Ridge baseline (ADR-024): Selected primary because its 5-fold GroupKFold CV RMSE (16.559) is lower than baseline's (19.152) (ADR-024). Official test metrics are reported but do not drive the choice. The official test ranking agrees with the CV ranking. |
| Target | RUL = max_cycle − cycle, capped at 125; predictions clamped to [0, 125] |
| Inputs | 43 features: `cycle` + raw, trailing 30-cycle mean and std of 14 sensors (s_2, s_3, s_4, s_7, s_8, s_9, s_11, s_12, s_13, s_14, s_15, s_17, s_20, s_21) |
| Libraries | scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.4, joblib 1.6.0; Python 3.13.3 |
| Native feature importance | None: `HistGradientBoostingRegressor` exposes no `feature_importances_` in scikit-learn, so §7b uses permutation importance |

Source: `models/model_metadata.json`, `reports/model_comparison.json`

| Artifact | SHA-256 |
|---|---|
| `models/model.joblib` | `39bfa6fd4a168b263acac4ccefa0be81a4fc838afa7eb14bf7306ad1099128c1` |
| `models/model_metadata.json` | `57f80600cf66b8cff78a10e7f9f46b8351b67dca4cb392ce6653c854cd06dcfd` |
| `models/model_intervals.joblib` | `9c7cfc5ea175516ca5a2e06177f2eebd3c2159768c692d3b046bc9939ae96d12` |
| `models/model_intervals_metadata.json` | `69b09c5b29f020f401aac4afc1634e6bb152967241a63640ebd0ebe536a8087a` |
| `models/model_anomaly.joblib` | `3f39354ff4f9805999efa8bd63e4bd10d79ef210ada60d512d6f843d7e2289b9` |
| `models/model_anomaly_metadata.json` | `3a0129bfcf159203dae1f828fc67fb1a6295447d30f05f6991d583fb8c82eb64` |
| `models/model_feature_importance.json` | `407c36131ac140032a799e17870bb462fd627284da56a756466abd715087b4c1` |

Source: `models/model.joblib`, `models/model_metadata.json`, `models/model_intervals.joblib`, `models/model_intervals_metadata.json`, `models/model_anomaly.joblib`, `models/model_anomaly_metadata.json`, `models/model_feature_importance.json` (hashes of the files themselves; `tests/test_model_card.py` re-checks them)

## 2. Intended use

- **Intended:** a portfolio demonstration of an end-to-end RUL pipeline. It estimates the remaining cycles of a turbofan engine **from the NASA C-MAPSS FD001 simulation** (one operating condition, one fault mode), given at least 30 consecutive cycles of the 14 kept sensors, via `POST /predict`.
- **Users:** engineers and reviewers evaluating ML engineering practice; not operators of real equipment.
- **Out of scope:** any real aircraft, engine or maintenance decision; other C-MAPSS subsets (FD002–FD004: several operating conditions or fault modes); engines with fewer than 30 cycles of history; use of the anomaly score as a validated fault detector; long-horizon RUL beyond the cap.

## 3. Factors

Performance depends on how far an engine is from failure (true-RUL range, §7a). Predictions are capped, so long-RUL engines are systematically under-predicted. Only one operating condition and fault mode are represented; no other demographic or environmental factors apply.

## 4. Training data

| Field | Value |
|---|---|
| Dataset | NASA C-MAPSS `FD001` training set (run to failure) |
| Eligible rows (cycle ≥ 30) / engines | 17,731 / 100 (sum over the CV folds' validation sets) |
| Data identity (SHA-256) | train `963b5e22825b34d8…`, test `3cda7109ce17bafb…`, RUL `a19c8ec94931949d…` |
| Dropped columns | setting_1, setting_2, setting_3, s_1, s_5, s_6, s_10, s_16, s_18, s_19 |

Source: `models/model_metadata.json`

## 5. Evaluation data and protocol

- **Cross-validation:** 5-fold `GroupKFold` by engine on the training set (no engine in both train and validation). This is the selection criterion.
- **Official test set:** the 100 FD001 test engines, truncated before failure, scored at their last observed cycle against `RUL_FD001` (unclipped truth). Reported only; never used for selection or tuning.

## 6. Metrics

### 6a. Point prediction (cycles)

| Model | CV RMSE (mean ± std) | CV MAE (mean ± std) | Test RMSE | Test MAE | NASA score |
|---|---|---|---|---|---|
| Ridge baseline | 19.152 ± 1.182 | 15.112 ± 1.156 | 19.144 | 15.347 | 590.6 |
| HistGradientBoosting (served) | 16.559 ± 0.695 | 11.843 ± 0.676 | 16.610 | 12.567 | 441.5 |

Source: `reports/model_comparison.json`

### 6b. 80 % prediction interval (quantile HGB, ADR-034; CQR calibration, ADR-037)

| Evaluation | Coverage, capped target | Coverage, unclipped truth | Mean width |
|---|---|---|---|
| CV out-of-fold, uncalibrated | 0.737 | 0.509 | 37.2 |
| Held-out engines (50), uncalibrated | 0.734 | 0.504 | 37.2 |
| Held-out engines, calibrated (Q_eval = 1.320; FR-020) | 0.777 | 0.536 | 39.2 |
| Official test, uncalibrated | 0.820 | 0.740 | 42.1 |
| Official test, calibrated (shipped Q = 1.742) | 0.850 | 0.750 | 44.7 |

Source: `models/model_intervals_metadata.json`

FR-020 (held-out coverage within 0.80 ± 0.03): **met**. Split-conformal coverage assumes exchangeable rows; rows of one engine are correlated, so the guarantee is approximate: the effective sample size is closer to the number of engines than to the number of rows.

### 6c. Anomaly score (unsupervised; ADR-035, smoothing rejected ADR-039)

| Measure (train out-of-fold) | Raw score (served) | Smoothed EWM half-life 10 (evaluated, not served) |
|---|---|---|
| Median per-engine Spearman(score, cycle), target > 0.7 | 0.680 | 0.693 |
| False-alarm rate, healthy rows | 0.0101 | 0.0101 |
| Median true RUL at first sustained detection | 49.0 | 44.0 |
| Engines never flagged | 0.020 | 0.160 |
| Acceptance | FR-019 not met | FR-021 not met |

Source: `reports/metrics_anomaly.json`

## 7. Quantitative analyses

### 7a. Served-model residuals on the official test set, by true RUL (error = predicted − true)

| True RUL | Engines | MAE | RMSE | Mean signed error | Share at cap |
|---|---|---|---|---|---|
| 0-20 | 16 | 3.430 | 5.821 | +1.618 | 0.000 |
| 21-60 | 23 | 8.820 | 12.476 | +5.380 | 0.000 |
| 61-125 | 50 | 14.597 | 17.586 | +1.418 | 0.040 |
| >125 | 11 | 24.464 | 26.972 | -24.464 | 0.000 |
| **All** | 100 | 12.567 | 16.610 | -0.486 | — |

Source: `reports/residuals_by_rul_bucket.json`

### 7b. Permutation feature importance (ADR-040; n_repeats 10, seed 42)

Importance = increase in RMSE (cycles) of the served, clamped predictions when one column is permuted; mean ± std over repeats. Per-sensor values sum a sensor's raw, mean and std columns within each repeat.

| Rank | Train population (17,731 rows, in-sample) | Test population (100 engines, out-of-sample) |
|---|---|---|
| 1 | s_3: 4.586 ± 0.045 | s_14: 2.534 ± 0.844 |
| 2 | s_14: 4.181 ± 0.044 | s_3: 1.884 ± 0.616 |
| 3 | s_2: 2.850 ± 0.037 | s_11: 1.801 ± 0.603 |
| 4 | s_9: 2.556 ± 0.026 | s_13: 1.382 ± 0.412 |
| 5 | s_11: 2.377 ± 0.026 | s_2: 1.135 ± 0.513 |
| 6 | s_13: 2.359 ± 0.028 | s_7: 1.097 ± 0.430 |
| 7 | s_8: 2.165 ± 0.034 | s_4: 0.956 ± 0.422 |
| 8 | s_12: 2.020 ± 0.040 | s_8: 0.699 ± 0.451 |
| 9 | s_7: 1.899 ± 0.032 | s_21: 0.384 ± 0.141 |
| 10 | s_4: 1.616 ± 0.027 | s_17: 0.199 ± 0.270 |
| 11 | s_17: 1.393 ± 0.016 | s_9: 0.164 ± 0.532 |
| 12 | s_15: 1.072 ± 0.015 | s_20: 0.131 ± 0.216 |
| 13 | s_20: 1.038 ± 0.018 | s_12: 0.125 ± 0.472 |
| 14 | s_21: 0.584 ± 0.012 | s_15: 0.030 ± 0.272 |

Source: `reports/feature_importance.json`

| Feature type | Train | Test |
|---|---|---|
| rolling mean (14 features) | 15.807 ± 0.092 | 5.868 ± 1.304 |
| rolling std (14 features) | 12.043 ± 0.097 | 4.946 ± 1.143 |
| cycle (1 feature) | 6.869 ± 0.055 | 2.991 ± 1.287 |
| raw sensor (14 features) | 2.846 ± 0.055 | 1.707 ± 0.747 |
| Spearman, Ridge per-sensor abs(coef) vs permutation per sensor | 0.226 | 0.059 |

Source: `reports/feature_importance.json`

Caveats (from the report): Permutation importance measures how much the served model relies on a feature, not a causal effect. Correlated features share importance: permuting one of a sensor's raw/mean/std columns or one of several correlated sensors can understate the group (hence per-sensor sums). Training population: in-sample, so the RMSE level is optimistic; use it for ranking. Test population: 100 engines at their last cycle, out-of-sample but noisy; see the std across repeats. HistGradientBoostingRegressor has no native feature_importances_ in scikit-learn.

## 8. Limitations

- Simulated data, one subset (FD001): no claim about real engines or other operating conditions.
- The RUL cap hides long-RUL engines: every test engine with true RUL > 125 is under-predicted (§7a: MAE equals minus the mean signed error), and the interval cannot cover that bucket against unclipped truth.
- Predictions for engines with 21–60 cycles left are on average late (positive mean signed error, §7a), the risky direction for maintenance.
- Interval coverage is marginal, not per RUL range, and approximate because rows of one engine are correlated (§6b).
- The anomaly score missed its target, raw and smoothed (§6c). It has no fault labels and is a coarse indicator only.
- Feature importance describes reliance, not causation. The test-population ranking is noisy beyond the top few sensors, and Ridge coefficients rank sensors differently (§7b).

## 9. Ethical considerations and recommendations

- Do not use for safety-relevant decisions; a real deployment would need data from the target fleet, validation against labelled failures, and human review of every maintenance call.
- Retrain and regenerate the interval, anomaly and importance artifacts together after any change of the served model (10 §6); this card must then be regenerated from the new files.
- Treat the 80 % interval as approximate, prefer the raw anomaly score only as a secondary signal, and report results against unclipped truth.
