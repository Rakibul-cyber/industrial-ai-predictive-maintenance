# 17 — Portfolio Notes

Talking points grounded in the implemented system. Numbers come from `reports/*.json`, [16](16_EVALUATION_REPORT.md) and the measurements in [14](14_PROJECT_STATUS.md).

## CV entry (3–4 lines)

**Industrial AI — Predictive Maintenance Platform** (Python, scikit-learn, FastAPI, Streamlit, Docker, GitHub Actions).
End-to-end remaining-useful-life prediction on NASA C-MAPSS FD001: checksummed data pipeline, leakage-safe rolling features, engine-grouped CV, and a HistGradientBoosting model selected on CV over a Ridge baseline (test RMSE 16.61 vs 19.14 cycles).
Served via a validated FastAPI service (p95 13.6 ms) and a Streamlit dashboard; Docker Compose with readiness healthchecks; 187 tests, 95 % coverage, CI with a Docker smoke test on amd64.

## 60-second summary

"I built a predictive-maintenance system that estimates how many cycles a turbofan engine has left, using NASA's C-MAPSS simulation data. The data is downloaded and checksum-verified by script. Features are trailing 30-cycle statistics of 14 informative sensors, computed by one function that both training and the API use, so there's no train/serve skew. I validated with GroupKFold by engine, so the model is always scored on engines it hasn't seen, and I chose between a Ridge baseline and a gradient-boosted model on cross-validation, keeping the official test set purely for reporting. Boosting cut test RMSE from 19.1 to 16.6 cycles. The model is served by FastAPI with strict input validation and a single error format, and it degrades gracefully if the model is missing. A Streamlit dashboard calls the API to show predictions over an engine's life. Everything runs with one `docker compose up` from a fresh clone, and CI runs lint, 187 tests with a coverage gate and a Docker smoke test."

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
Test RMSE 16.6 and MAE 12.6 cycles. That's solid for a simple tabular approach, but I would not claim state of the art. Published deep-learning results on FD001 are generally reported lower, but they often score against clipped truth, so they aren't directly comparable; I did not verify specific numbers in this project. It is most accurate near failure (MAE 3.4 for true RUL ≤ 20). For engines with 21–60 cycles left it is on average ~5 cycles late, which is the risky direction for maintenance. It gives no uncertainty estimate.

**9. What would you do with more time?**
- Calibrate or bias-correct the 21–60 range.
- Add prediction intervals (quantile boosting or conformal prediction).
- Extend to FD002–FD004 with per-regime normalisation.
- Try a sequence model under the same CV protocol.
- Add input-drift monitoring against the stored training ranges.

**10. Why commit a binary model to git?**
It is 451 KB and makes a fresh clone runnable with `docker compose up` alone. The risks (pickle portability, review) are mitigated by the scikit-learn pin, a version-check test, a test that the committed model reproduces the recorded demo predictions, and a retraining procedure that reproduces it byte-for-byte.

## Strongest points

- An honest evaluation protocol: grouped CV, test used only for reporting, residuals analysed by RUL range, limitations stated.
- No train/serve skew by construction, backed by contract tests.
- Reproducibility: checksummed data, pinned dependencies, deterministic training, byte-identical retrain from a fresh clone.
- Production habits at small scale: one error envelope, degraded mode, readiness healthchecks, non-root digest-pinned image, CI with a container smoke test.
- A full decision trail: 32 ADRs and 6 change records.

## Weakest points

- Modest model accuracy and no uncertainty quantification; optimistic bias in the 21–60 cycle range.
- Single subset (FD001) on simulated data.
- The cap hides long-RUL engines (the engine 1 demo sits at 125).
- One 1.1 GB image serves both roles; the API carries Streamlit's dependencies.
- The dashboard has no browser-level automated test (AppTest only), and no screenshots were captured in-session.
- The Linux bind-mount write path in containers is not asserted by the smoke test.
