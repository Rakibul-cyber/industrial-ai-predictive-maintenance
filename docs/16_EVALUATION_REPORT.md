# 16 — Evaluation Report

All numbers in this report are read from the committed files [`reports/metrics_anomaly.json`](../reports/metrics_anomaly.json) (§10), [`reports/metrics_intervals.json`](../reports/metrics_intervals.json) (§9), [`reports/metrics_baseline.json`](../reports/metrics_baseline.json), [`reports/metrics_primary.json`](../reports/metrics_primary.json), [`reports/model_comparison.json`](../reports/model_comparison.json), [`reports/residuals_by_rul_bucket.json`](../reports/residuals_by_rul_bucket.json) and [`models/model_metadata.json`](../models/model_metadata.json); the document was generated from them in Milestone 7. The served model is `primary-20261108-f4c86df`.

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
- **Uncertainty:** since M8 a nominal 80 % quantile interval is served (§9). It is below nominal coverage out-of-fold and cannot cover true RUL > 125; health bands are illustrative thresholds.
- **Selection bias:** small (best of 12 configurations on CV; test-set agreement suggests it is minor).
- **Pickle tied to the scikit-learn pin:** `model.joblib` requires scikit-learn 1.9.1; a test fails on a version mismatch and retraining reproduces the artifact byte-identically.

## 8. Optional extensions — NOT IMPLEMENTED

None of the following is part of the MVP. Each would need its own change record and milestone (15).

| Extension (NOT IMPLEMENTED) | What it would need |
|---|---|
| FD002–FD004 (multiple conditions / fault modes) | Operating-condition normalisation (per-regime sensor scaling), regime-aware features, re-validated data contract and checksums, larger evaluation |
| ~~Anomaly detection~~ — **IMPLEMENTED in M9** (§10, ADR-035); trend target not met | Remaining: engine-specific baselines, a detection-timing target agreed in advance, evaluation on data with labelled faults |
| Sequence models (LSTM/CNN/Transformer) | A deep-learning dependency (ADR), GPU-optional training, sequence windows in the shared feature path, longer training |
| ~~Prediction intervals~~ — **IMPLEMENTED in M8** (§9, ADR-034) | Remaining: calibration toward nominal coverage (e.g. conformal adjustment on out-of-fold residuals) |
| Bias correction for the 21–60 range | Residual analysis on CV folds, asymmetric loss or post-hoc calibration, then re-selection |
| Drift monitoring / retraining | Logging of inputs, distribution checks against `training_ranges`, a retraining trigger and model registry |
| Cloud deployment | Container registry, managed runtime, auth/TLS, secrets management, cost controls |

## 9. Prediction intervals (Milestone 8, ADR-034)

Source: [`reports/metrics_intervals.json`](../reports/metrics_intervals.json) and `models/model_intervals_metadata.json`, trained 2026-11-08T18:55:47Z for served model `primary-20261108-f4c86df` (code: commit `8d893ff` plus the uncommitted M8 working tree, `git_dirty: true`).

**Method.** Two `HistGradientBoostingRegressor(loss="quantile")` models at q = 0.1 and q = 0.9 give a nominal **80%** interval. They use the served model's features, train-row filter, seed (42) and hyperparameters `{'learning_rate': 0.05, 'max_iter': 200, 'max_leaf_nodes': 15, 'l2_regularization': 0.0}`; there is no search, and the point model is not retrained. Post-processing: clamp to [0, rul_cap]; lower=min(lower, point); upper=max(upper, point). Coverage counts a true RUL inside [lower, upper], inclusive. Training cost: CV 14.3 s (3 models × 5 folds), final fit 1.8 s (2 models).

### 9a. Out-of-fold (5-fold GroupKFold by engine, every eligible training row)

| Scope | n | Coverage (unclipped truth) | Coverage (clipped truth) | Mean width | Median width |
|---|---|---|---|---|---|
| **All** | 17,731 | **0.509** | **0.737** | 37.2 | 36.7 |
| true RUL 0-20 | 2,100 | 0.768 | 0.768 | 32.1 | 29.7 |
| true RUL 21-60 | 4,000 | 0.627 | 0.627 | 43.9 | 43.9 |
| true RUL 61-125 | 6,392 | 0.767 | 0.767 | 43.6 | 45.5 |
| true RUL >125 | 5,239 | 0.000 | 0.773 | 26.2 | 21.9 |

Reordered (bound moved to the point, incl. quantile crossing): **1,023** of 17,731 (5.8%).

### 9b. Official test set (last observed cycle, 100 engines)

| Scope | n | Coverage (unclipped truth) | Coverage (clipped truth) | Mean width | Median width |
|---|---|---|---|---|---|
| **All** | 100 | **0.740** | **0.820** | 42.1 | 42.0 |
| true RUL 0-20 | 16 | 0.938 | 0.938 | 27.7 | 20.4 |
| true RUL 21-60 | 23 | 0.826 | 0.826 | 49.4 | 43.9 |
| true RUL 61-125 | 50 | 0.800 | 0.800 | 44.4 | 49.9 |
| true RUL >125 | 11 | 0.000 | 0.727 | 37.9 | 39.7 |

Reordered: **8** of 100.

### 9c. Reading the results — nothing was tuned toward 80 %

- **The cap makes > 125 uncoverable against unclipped truth.** Both bounds are capped at 125 because the models learn capped labels. Coverage is therefore 0.000 (CV) and 0.000 (test) for engines with true RUL > 125, by construction. Against clipped truth, the target the model actually learns, the same bucket reaches 0.773 / 0.727. This is expected; it is reported, not fixed.
- **Overall unclipped CV coverage (0.509) is dominated by that effect:** 5,239 of 17,731 training rows (30%) have true RUL > 125. Excluding the cap effect (clipped truth), CV coverage is 0.737, below the nominal 0.80.
- **Per bucket, out-of-fold coverage is below nominal everywhere** (0–20: 0.768, 21–60: 0.627, 61–125: 0.767). The 21–60 range — where the point model is also optimistically biased (§6) — is the least covered. Quantile models fitted on all training data are mildly over-confident on unseen engines.
- **On the official test set** coverage is 0.740 (unclipped) / 0.820 (clipped); per bucket 0.938 / 0.826 / 0.800 / 0.000. With 11–50 engines per bucket these estimates are noisy (one engine = 2–9 percentage points); the CV numbers over 17,731 rows are the more reliable estimate.
- **Width:** mean 37.2 cycles (CV) / 42.1 (test). Near failure the interval is narrowest (test 0–20: mean 27.7), where it matters most.
- **Demo engines:** late-life test engine 34 predicts 6.3 in [4.84, 18.76] (true 7, covered). Early-life test engine 1 predicts 125.0 in [118.45, 125.0] (true 112, **not** covered): the cap effect on a single engine.
- Honest use: treat the interval as an approximate ~74 % (CV, clipped truth) band for engines not beyond the cap, not as a calibrated 80 % guarantee. Calibration (e.g. conformal adjustment on out-of-fold residuals) is a candidate next step, NOT IMPLEMENTED.

## 10. Anomaly detection (Milestone 9, ADR-035)

Numbers in this section are read from [`reports/metrics_anomaly.json`](../reports/metrics_anomaly.json) (detector `anomaly-20261108-a25df0e`, for served model `primary-20261108-f4c86df`).

**What is measured.** C-MAPSS has no anomaly labels. The "anomaly" score is the Mahalanobis distance (Ledoit-Wolf covariance, standardised features) of an engine's 14 trailing 30-cycle sensor means from a **healthy reference**: 5,329 training rows with a full window and true RUL ≥ 125, from 90 of 100 engines (an engine needs ≥ 155 cycles of life to contribute). The threshold is the 99% quantile of **out-of-fold** healthy-row scores (5-fold GroupKFold by engine): **5.4036**; the in-sample quantile would be 5.1510, i.e. lower and optimistic. `normalized_score = score / threshold`; `> 1` is flagged.

### 10a. Results

| # | Measure | Train (out-of-fold, 100 engines run to failure) | Test (100 truncated engines) |
|---|---|---|---|
| 1 | Per-engine Spearman(normalised score, cycle): median (IQR) | **0.680 (IQR 0.506–0.818)** | 0.213 (IQR -0.177–0.569) |
| 1 | Share of engines with Spearman > 0.7 | 0.460 (100 engines scored) | 0.141 (99 scored; 1 engine has too few full-window cycles) |
| 2 | False-alarm rate on healthy rows (score > threshold) | 0.0101 of 5,329 rows (expected ≈ 0.01 by construction) | — |
| 3 | First sustained detection (3 consecutive cycles > threshold): fraction of life, median (IQR) | 0.758 (IQR 0.638–0.872) | — (test engines do not reach failure) |
| 3 | True RUL at detection, median (IQR) | 49.0 (IQR 26.0–75.8) cycles | — |
| 3 | Engines never flagged | 0.020 | — |
| 4 | Engines flagged at their last observed cycle | — | 0.250 overall (by bucket below) |
| 5 | Spearman(normalised score, predicted RUL) at the last cycle | — | -0.613 |

| Test engines by true RUL at last cycle | n | Flagged fraction |
|---|---|---|
| true RUL 0-20 | 16 | 0.750 |
| true RUL 21-60 | 23 | 0.391 |
| true RUL 61-125 | 50 | 0.080 |
| true RUL >125 | 11 | 0.000 |

**Acceptance (FR-019):** median per-engine Spearman(score, cycle) on train OOF > 0.7 → value **0.680** → **NOT MET**. The IsolationForest fallback was not triggered (not needed: median train-OOF Spearman 0.680 >= 0.5). Nothing was tuned toward the target after seeing the result.

### 10b. Reading the results

- **The score rises with degradation, but not consistently enough to meet the target.** Half the engines have a train-OOF trend above 0.68; 46 % exceed 0.7. A whole-life Spearman is diluted by the long flat early phase, where the score is noise around the healthy level.
- **False alarms match the design rate.** 1.01 % of healthy rows exceed the threshold, consistent with the out-of-fold 99th-percentile threshold (i.e. the threshold generalises to unseen engines).
- **Detection is late but useful.** The first sustained flag comes at a median 76 % of life, when a median 49 cycles remain (IQR 26–76); 2 % of training engines are never flagged. Measured on true RUL, the median detection (49 cycles left) falls between the dashboard's warning (60) and critical (20) thresholds.
- **Test trend is low because most test engines are cut off early.** In-session check on the committed artifact (command in 14): median per-engine Spearman is 0.517 for test engines with true RUL ≤ 60 (n = 39), 0.040 for 61–125 (n = 49) and −0.320 for > 125 (n = 11); across engines, trend vs true RUL at cutoff has Spearman −0.525. Engines still in the healthy region have no degradation trend to find.
- **Flags concentrate near failure.** 75 % of test engines with true RUL ≤ 20 are flagged at their last cycle, 39 % for 21–60, 8 % for 61–125 and none above 125.
- **The score agrees with the RUL model** (Spearman −0.613 with the predicted RUL: higher anomaly score, lower predicted RUL), even though the two models share only the feature function.
- **Demo engines:** test engine 1 (31 cycles, true RUL 112) scores 0.6657 (normal); late-life engine 34 (203 cycles, true RUL 7) scores 1.1015 (anomalous).

### 10c. What this evaluation does and does not prove

- **Does show:** on simulated run-to-failure data, distance from a population healthy reference increases as engines degrade, the out-of-fold threshold keeps the false-alarm rate on unseen healthy engines at the designed ~1 %, and the first sustained flag typically falls in the last third of life (IQR 64–87 % of life).
- **Does not show:** detection of real faults or of anything other than FD001's single simulated degradation mode; calibrated alarm timing (no labels define when an alarm *should* fire); robustness to sensor faults, drift or other operating conditions. The trend target (median > 0.7) was not met, so the score should be read as a coarse health indicator alongside the RUL prediction, not as a validated anomaly detector.
