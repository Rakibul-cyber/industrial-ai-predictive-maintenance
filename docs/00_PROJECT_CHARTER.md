# 00 — Project Charter

**Project:** Industrial AI — Predictive Maintenance Platform (renamed by CR-007 / ADR-033)
**Owner:** MD Rakibul Hassan · **Charter date:** 2026-11-08 · **Phase:** Milestone 0 (planning)

## 1. Problem statement

Unplanned failures of rotating equipment (here: turbofan engines) are expensive; time-based maintenance replaces parts too early or too late. Given a history of per-cycle sensor readings for an engine, the platform must estimate its **Remaining Useful Life (RUL)** in operating cycles, so maintenance can be scheduled before failure.

The project uses the public NASA C-MAPSS run-to-failure simulation dataset (subset **FD001**: one operating condition, one fault mode — HPC degradation; verified from the dataset `readme.txt`, see [04](04_DATASET_SPECIFICATION.md)).

## 2. Objectives

1. Reproducible data acquisition with checksum verification.
2. A leakage-free RUL regression pipeline: one interpretable baseline and one stronger primary model, both evaluated on the official FD001 test set.
3. A versioned model artifact with metadata that ties it to data hash, code version, and metrics.
4. A REST API serving real predictions with strict input validation.
5. A dashboard that obtains every prediction from the API (no hard-coded results).
6. Automated tests, CI (lint + tests), and a Docker Compose deployment that starts both services locally.

## 3. Target users

| User | Need |
|---|---|
| Maintenance planner (persona) | See predicted RUL and a health band per engine to prioritise work. |
| ML / MLOps reviewer (portfolio audience) | Inspect a clean, reproducible, tested end-to-end ML system. |
| Integrating developer | Call a documented, validated HTTP API. |

## 4. Value

- **Business (illustrative):** earlier, data-driven maintenance scheduling; health bands translate a regression output into an action category.
- **Technical:** demonstrates correct handling of grouped time-series data (engine-level CV, causal features, truncated test trajectories), artifact versioning, and a tested serving path.
- **Portfolio relevance:** covers the full lifecycle — data contract → modelling → evaluation → serving → UI → tests → containers → documentation — with honest, reproducible metrics.

## 5. MVP definition

The MVP is complete when, on a clean machine with Python 3.13 and Docker:
`download data → train (baseline + primary) → evaluate on official test set → save artifact → docker compose up → dashboard shows API predictions for FD001 test engines`
works end to end, and all acceptance criteria in [11](11_ACCEPTANCE_CRITERIA.md) are met.

## 6. Out of scope (MVP)

- Subsets FD002–FD004 (multiple operating conditions / fault modes).
- Anomaly-detection models beyond the minimal M9 detector (ADR-035), e.g. autoencoders or supervised fault classifiers. The MVP delivered RUL prediction only; Milestone 9 adds a minimal unsupervised detector (see the post-MVP note in §7).
- Deep learning / sequence models (LSTM, CNN, Transformers).
- Hyperparameter search beyond the single bounded 12-configuration search of the primary model (ADR-026).
- Cloud deployment, Kubernetes, authentication, multi-tenancy.
- Drift monitoring, online retraining, experiment-tracking servers (MLflow).
- Real (non-simulated) sensor data ingestion or streaming.

These are candidate extensions only after Milestone 7 and only via [15](15_CHANGE_CONTROL.md).

## 7. Deliverables

| # | Deliverable | Milestone |
|---|---|---|
| D1 | Data download + validation module and CLI | M1 |
| D2 | RUL labelling + feature engineering module | M2 |
| D3 | Training/evaluation pipeline, model artifact + `model_metadata.json` | M3 |
| D4 | FastAPI service (`/health`, `/predict`, `/model/info`) | M4 |
| D5 | Streamlit dashboard calling the API | M5 |
| D6 | Dockerfile + docker-compose; CI workflow (CI delivered in M1) | M6 / M1 |
| D7 | Final README and docs matching code | M7 |

**Post-MVP extension (Milestone 8, CR-008 / ADR-034):** prediction intervals (nominal 80 %, quantile gradient boosting) on `/predict`, in the dashboard and in the evaluation report. This moves out of the extensions list into implemented scope; the other extensions remain NOT IMPLEMENTED.

**Post-MVP extension (Milestone 9, CR-010 / ADR-035):** minimal unsupervised anomaly detection: Mahalanobis distance of rolling-mean sensors to a healthy reference (true RUL ≥ 125), out-of-fold threshold, `POST /anomaly`, dashboard view, and evaluation without labels (trend, false alarms, detection timing). Outcome: implemented and verified, but its trend target (median per-engine Spearman > 0.7 on train OOF) was **not met** (0.680; 16 §10), so the project title is unchanged. Remaining extensions stay NOT IMPLEMENTED.

## 8. Success criteria

- Official FD001 test-set MAE and RMSE are **measured and reported from actual runs** for both models (no target value is promised in advance; see [12](12_RISK_REGISTER.md) R4).
- The served model is the candidate with the lower 5-fold GroupKFold CV RMSE (tie-break CV MAE); official test metrics are reported for both but do not drive the choice (ADR-024).
- All tests pass locally and in CI; Docker Compose starts both services and the smoke test passes.
- Every statement in README/docs is consistent with the code (verified in M7).
