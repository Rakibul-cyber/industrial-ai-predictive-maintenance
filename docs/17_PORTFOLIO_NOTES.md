# 17 — Portfolio Notes

Talking points grounded in the implemented system. Numbers come from `reports/*.json`, [16](16_EVALUATION_REPORT.md) and the measurements in [14](14_PROJECT_STATUS.md).

## CV entry (3–4 lines)

**Industrial AI — Predictive Maintenance Platform** (Python, scikit-learn, FastAPI, Streamlit, Docker, GitHub Actions).
End-to-end remaining-useful-life prediction on NASA C-MAPSS FD001: checksummed data pipeline, leakage-safe rolling features, engine-grouped CV, and a HistGradientBoosting model selected on CV over a Ridge baseline (test RMSE 16.61 vs 19.14 cycles), plus quantile-regression prediction intervals and an unsupervised anomaly score, both evaluated and reported honestly, including where they fall short.
Served via a validated FastAPI service (p95 13.6 ms) and a Streamlit dashboard; Docker Compose with readiness healthchecks; 305 tests, 97 % coverage, CI with a Docker smoke test on amd64.

## 60-second summary

"I built a predictive-maintenance system that estimates how many cycles a turbofan engine has left, using NASA's C-MAPSS simulation data. The data is downloaded and checksum-verified by script. Features are trailing 30-cycle statistics of 14 informative sensors, computed by one function that both training and the API use, so there's no train/serve skew. I validated with GroupKFold by engine, so the model is always scored on engines it hasn't seen, and I chose between a Ridge baseline and a gradient-boosted model on cross-validation, keeping the official test set purely for reporting. Boosting cut test RMSE from 19.1 to 16.6 cycles. The model is served by FastAPI with strict input validation and a single error format, and it degrades gracefully if the model is missing. A Streamlit dashboard calls the API to show predictions over an engine's life. Everything runs with one `docker compose up` from a fresh clone, and CI runs lint, 264 tests with a coverage gate and a Docker smoke test. Later I added prediction intervals, calibrated them conformally on held-out engines against a target I fixed in advance (met, narrowly), and a simple anomaly score that missed its target. One pre-registered fix (smoothing) also failed and made detection worse, so I kept serving the raw score and reported both."

## Likely interview questions

**1. Why GroupKFold instead of normal K-fold?**
Each engine contributes up to ~360 highly correlated rows. Random K-fold would put neighbouring cycles of the same engine in both train and validation, which inflates scores. GroupKFold by `unit` keeps every engine on one side, so CV estimates performance on unseen engines, which is the deployment question. CV and test agree closely (primary 16.56 vs 16.61 RMSE), which supports that.

**2. Why cap RUL at 125?**
Early in life the sensors are flat, so a linear "cycles to failure" label asks the model to predict something the data cannot show. A cap is the common piecewise-linear target for FD001. 125 is below the shortest training life (128 cycles), so every engine still has an uncapped degradation segment. The cost is visible: engines with true RUL > 125 are all under-predicted, and those 11 engines contribute 29 % of the test squared error.

**3. Why select on CV rather than the test set?**
Choosing with the test set turns it into a validation set and biases the reported number. The rule is pre-registered (ADR-024): lowest CV RMSE, tie-break MAE. Test metrics are reported for both models, and a disagreement between the rankings would have been flagged (it didn't occur).

**4. How did you prevent leakage?**
- Grouped folds.
- Trailing windows only: each window is computed independently, and a test proves that adding future cycles doesn't change a row.
- Sensor selection on train only.
- Scaling inside the sklearn `Pipeline`, so it is fit per fold.
- HistGradientBoosting's internal early stopping turned off, because its validation split is row-wise.
- No test-set involvement in tuning or selection.

**5. Why not LightGBM or MLflow?**
On the dev machine LightGBM installed but failed to import (missing `libomp`), and scikit-learn's HistGradientBoosting gives comparable trees with no extra dependency. MLflow would have added 40+ packages to track one model; a metadata JSON with data hashes, git SHA, library versions and metrics covers lineage here. Both are reasonable additions at larger scale.

**6. How do training and serving stay consistent?**
- One `features.build` function serves both.
- The pipeline's first step selects exactly the 43 training columns.
- The same clamp to [0, 125] is used in evaluation and in the API.
- A contract test asserts that the API prediction equals the offline prediction.
- A real-data test recomputes the served model's test RMSE through the API's inference function.

**7. What happens if the model file is missing or incompatible?**
The API starts anyway. `/health` returns 200 with `model_loaded: false`, while `/predict` and `/model/info` return 503 in the standard error envelope. The Docker healthcheck reads `model_loaded`, so the container is unhealthy and the dashboard waits. scikit-learn is pinned, `/model/info` reports a version match, and a test fails if the installed version differs from the one used for training.

**8. How good is the model, honestly?**
Test RMSE 16.6 and MAE 12.6 cycles. That's solid for a simple tabular approach, but I would not claim state of the art. Published deep-learning results on FD001 are generally reported lower, but they often score against clipped truth, so they aren't directly comparable; I did not verify specific numbers in this project. It is most accurate near failure (MAE 3.4 for true RUL ≤ 20). For engines with 21–60 cycles left it is on average ~5 cycles late, which is the risky direction for maintenance. Since M8 it gives an 80 % interval, which under-covers (question 11).

**9. What would you do with more time?**
- Calibrate or bias-correct the 21–60 range.
- Calibrate the intervals per RUL range (M10's single conformal correction leaves 21–60 at 0.687).
- Give the anomaly score engine-specific baselines (question 12).
- Extend to FD002–FD004 with per-regime normalisation.
- Try a sequence model under the same CV protocol.
- Add input-drift monitoring against the stored training ranges.

**10. Why commit a binary model to git?**
It is 451 KB and makes a fresh clone runnable with `docker compose up` alone. The risks (pickle portability, review) are mitigated by the scikit-learn pin, a version-check test, a test that the committed model reproduces the recorded demo predictions, and a retraining procedure that reproduces it byte-for-byte.

**11. Why quantile regression for the intervals, and what does the RUL cap do to coverage?** (M8)
Two extra HistGradientBoosting models with quantile loss (q = 0.1 and 0.9) reuse the exact pipeline and hyperparameters of the served model. That means no new dependency and no search, and the widths depend on the input. The point model is untouched, and the API falls back to null bounds if the artifact doesn't match it. The cap is the honest catch: the quantile models learn capped labels and the bounds are clamped at 125, so an engine whose true RUL is above 125 can never be covered. Against unclipped truth, out-of-fold coverage is 0.509 overall and 0 in the > 125 bucket. Against the capped target it's 0.737, still below the nominal 0.80. I reported that rather than tuning toward 80 %, and calibrated it conformally in M10 (question 13).

**12. How do you evaluate an anomaly detector when there are no anomaly labels?** (M9)
C-MAPSS has run-to-failure trajectories but no labelled faults, so I treat "anomaly" as distance from a healthy state and ask whether the score behaves like a health indicator. The healthy reference is training rows with true RUL ≥ 125, where the RUL target is flat. The score is a Mahalanobis distance on the 14 rolling-mean sensors with a Ledoit-Wolf covariance. The threshold is the 99th percentile of **out-of-fold** healthy scores: in-sample it would be 5.15 instead of 5.40, which is optimistic. I fixed five checks before looking at results:
- trend: per-engine Spearman(score, cycle), target median > 0.7;
- false alarms on healthy rows;
- detection timing (3 consecutive cycles above threshold);
- test-engine flags by true-RUL bucket;
- agreement with the RUL model.

The result: the false-alarm rate is 1.01 % (as designed), the first sustained detection comes at a median 76 % of life with 49 cycles left, and the score correlates −0.61 with the predicted RUL. But the median trend is **0.680, so the target was missed**. I reported it, didn't tune the method or the target afterwards, and kept the project title unchanged. What it does not prove: detection of real faults, or calibrated alarm timing.

**13. What does conformalized quantile regression do, and why does engine-level correlation matter?** (M10)
CQR keeps the quantile models and learns one correction from data they didn't train on. For each out-of-fold row I take E = max(lo − y, y − hi): how far the true (capped) RUL lies outside the interval, negative if inside. The correction Q is the ⌈(n+1)·0.8⌉-th smallest E, and every interval becomes [lo − Q, hi + Q]. If calibration and new rows are exchangeable, that gives at least 80 % coverage. No model is retrained.

The catch is exchangeability. Rows from one engine are strongly correlated, so 8,800 rows from 50 engines behave more like 50 samples than 8,800. That's why I split by **engine**, computed Q on 50 engines, and measured coverage on the other 50, against a band I wrote down first (0.80 ± 0.03). Result: 0.734 → 0.777, met at the lower edge. The Q from half the engines (1.32) was smaller than the Q from all of them (1.74), and that variability is exactly the engine-level effect. The guarantee is also marginal: the 21–60 range is still at 0.687. I'd fix that with RUL-range-aware calibration rather than by tuning α.

**14. The anomaly score missed its target. Why did you test only one pre-registered remedy, and what did you do when it failed?** (M11)
The raw score's median trend was 0.680 against a target of 0.7 that was on record. If I'd swept half-lives until one crossed 0.7, I'd have been fitting the parameter to the acceptance metric. With 100 engines and one metric, a sweep almost guarantees a winner by chance, and the "pass" would say nothing about new engines. So I fixed one remedy before computing anything: an EWM with a 10-cycle half-life. I wrote down the criterion (trend > 0.7 and false alarms ≤ 2 %) and what I expected: the false-alarm half passes by construction, and smoothing adds lag.

It failed: 0.693. The side-by-side evaluation then showed the real cost. 16 % of training engines were never flagged, against 2 % raw; detection came 5 cycles later; and fewer near-failure test engines were flagged. So I didn't ship it. The API serves the raw score again, and the smoothing code, threshold and evaluation stay in the repo behind `smoothing_enabled: false` as a documented negative result. The general point: a remedy that nudges the headline metric while making the system worse at its job is not an improvement, and you only see that if you measure more than the headline metric.

**15. Which sensors matter, and how do you know?** (M12)
I measured how much the served model relies on each input with permutation importance: shuffle one column, re-score, and see how much the RMSE grows, 10 times per feature, without refitting. I did it on two populations because each has a weakness:
- **Training rows (17,731):** in-sample, so the error level is optimistic, but the ranking is stable.
- **The 100 test engines:** out-of-sample, but noisy, so I report the spread across repeats.

Both put `s_3` and `s_14` on top. Rolling means and stds matter more than raw readings, and on the test set `cycle` (engine age) is the single biggest feature. That fits the windowed design but is also a transfer risk. Below the top handful, the test-set ranking is within noise, and I say so rather than ranking 14 sensors precisely.

As a cross-check I compared against the Ridge baseline's standardized coefficients. They barely agree (Spearman 0.23 / 0.06): `s_3` is first for the trees and last for Ridge. That's a useful lesson. Coefficients on correlated, standardized features are not importances, and permutation importance is about the model's reliance, not causation. Everything is in a model card whose tables are generated from committed files and checked by a test.

## Strongest points

- An honest evaluation protocol: grouped CV, test used only for reporting, residuals analysed by RUL range, limitations stated.
- No train/serve skew by construction, backed by contract tests.
- Reproducibility: checksummed data, pinned dependencies, deterministic training, byte-identical retrain from a fresh clone.
- Production habits at small scale: one error envelope, degraded mode, readiness healthchecks, non-root digest-pinned image, CI with a container smoke test.
- A full decision trail: 39 ADRs and 13 change records, plus a model card (doc 18) whose numbers trace to committed files.

## Weakest points

- Modest model accuracy; the 80 % intervals (M8) are under-covering and cannot cover true RUL > 125; optimistic bias in the 21–60 cycle range.
- The anomaly score missed its trend target (M9, 0.680). The one pre-registered remedy failed (M11, 0.693) and was rejected. The score has no labels to be validated against.
- Single subset (FD001) on simulated data.
- The cap hides long-RUL engines (the engine 1 demo sits at 125).
- One 1.1 GB image serves both roles; the API carries Streamlit's dependencies.
- The dashboard has no browser-level automated test (AppTest only), and no screenshots were captured in-session.
- The Linux bind-mount write path in containers is not asserted by the smoke test.
