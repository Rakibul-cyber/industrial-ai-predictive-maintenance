# 13 — Decision Log (ADRs)

ADR-001…021 dated **2026-11-08** (Milestone 0); architecture freeze approved by the user on 2026-11-08, which accepted ADR-001…021 and added ADR-022. Status values: *Proposed*, *Accepted*, *Superseded by ADR-xxx*. Changes follow [15](15_CHANGE_CONTROL.md); accepted ADRs are never edited silently — a new ADR supersedes them.

| ID | Title | Status |
|---|---|---|
| ADR-001 | MVP scope: FD001 RUL regression only | Accepted (2026-11-08) |
| ADR-002 | Python 3.13 | Accepted (2026-11-08) |
| ADR-003 | Data source, checksums, no redistribution | Accepted (2026-11-08) |
| ADR-004 | RUL target with clip 125 | Accepted (2026-11-08) |
| ADR-005 | Feature strategy | Accepted (2026-11-08) |
| ADR-006 | Baseline: StandardScaler + Ridge | Accepted (2026-11-08) |
| ADR-007 | Primary: HistGradientBoostingRegressor (LightGBM rejected) | Accepted (2026-11-08) |
| ADR-008 | Validation protocol and metrics | Accepted (2026-11-08) |
| ADR-009 | NASA score coefficient interpretation | Accepted (2026-11-08) |
| ADR-010 | Model selection rule | Accepted (2026-11-08) |
| ADR-011 | Artifacts: joblib + metadata JSON; no MLflow | Accepted (2026-11-08) |
| ADR-012 | Backend: FastAPI + Pydantic v2 + uvicorn | Accepted (2026-11-08) |
| ADR-013 | httpx2 instead of httpx for TestClient | Accepted (2026-11-08) |
| ADR-014 | Frontend: Streamlit + requests over HTTP | Accepted (2026-11-08) |
| ADR-015 | Packaging: one image, two compose services, mounted artifacts | Accepted (2026-11-08) |
| ADR-016 | Tooling: ruff, exact pins, pytest, CI from M1 | Accepted (2026-11-08) |
| ADR-017 | Health band thresholds | Accepted (2026-11-08) |
| ADR-018 | Clamp predictions at 0 | Accepted (2026-11-08) |
| ADR-019 | Modular monolith, package `predmaint` | Accepted (2026-11-08) |
| ADR-020 | Generic sensor names | Accepted (2026-11-08) |
| ADR-021 | Synthetic fixtures for CI; real-data tests gated | Accepted (2026-11-08) |
| ADR-022 | 503 without artifact; demo = train command + `docker compose up` | Accepted (2026-11-08) |

---

### ADR-001 — MVP scope: FD001 RUL regression only

- **Context:** C-MAPSS has four subsets; FD002/FD004 have six operating conditions, FD003/FD004 two fault modes (readme, verified). Project name mentions anomaly detection.
- **Decision:** MVP = RUL regression on FD001. Anomaly detection, other subsets, deep learning, cloud, drift monitoring are out of scope.
- **Alternatives:** all four subsets (needs condition normalisation); add an anomaly model (second model family, second API surface).
- **Rationale:** smallest scope that demonstrates the full lifecycle correctly.
- **Consequences:** model valid only for FD001-like conditions (05 §11); name reflects platform direction, documented in 00.

### ADR-002 — Python 3.13

- **Context:** local `python3` = 3.13.3; `python:3.13-slim` = 3.13.18; all pins install and import on 3.13; cp313 Linux wheels exist for x86_64 and aarch64.
- **Decision:** `requires-python = ">=3.13,<3.14"`; base image `python:3.13-slim` (digest pinned in M6).
- **Alternatives:** 3.12 (would need a second interpreter locally, none installed).
- **Rationale:** matches the verified machine; no evidence of incompatibility.
- **Consequences:** `python` is not on PATH locally — commands use `python3` or the venv.

### ADR-003 — Data source, checksums, no redistribution

- **Context:** S3 URL linked from the NASA PCoE page works (12,429,152 bytes); two other NASA URLs fail (404 / 403); licence "not specified" on data.nasa.gov.
- **Decision:** download from the S3 URL; verify SHA-256 of outer zip, inner zip, and FD001 files (04 §3); never commit data.
- **Alternatives:** commit data (licence unclear); Kaggle mirror (not verified, extra licence question).
- **Rationale:** reproducible and integrity-checked without redistribution.
- **Consequences:** first run needs network access; R1 fallback defined.

### ADR-004 — RUL target with clip 125

- **Context:** linear RUL labels imply degradation is visible from cycle 1, but early-life sensor readings are mostly flat; a capped ("piecewise-linear") target is common practice for C-MAPSS.
- **Decision:** `RUL = max_cycle − cycle`, clipped at `rul_cap = 125` (config; `None` disables).
- **Alternatives:** no clip; other ceilings (e.g. 120, 130).
- **Rationale:** 125 is below the shortest training life (128 cycles, verified), so every training unit has an unclipped degradation segment; it is a widely used value for FD001.
- **Consequences:** predictions cap near 125 while true test RUL reaches 145 → systematic under-prediction for long-RUL units (05 §11). Test metrics are reported against unclipped truth (primary) and clipped truth (secondary).

### ADR-005 — Feature strategy

- **Context:** single operating condition; shortest test unit = 31 cycles (verified); no sequence models allowed.
- **Decision:** exclude `setting_1..3`; drop sensors with ≤ 2 unique training values (confirmed on real data in M2-T1); features = `cycle` + per kept sensor raw, trailing rolling mean and std over `window_size = 30` rows; only rows with `cycle ≥ 30` are used for training and inference.
- **Alternatives:** expanding/partial windows (`min_periods=1`) — gives early rows different statistical meaning than served rows; window 10/20 — less smoothing; include settings — near-noise for one condition.
- **Rationale:** identical feature semantics in training and serving; all 100 test units evaluable at full window; minimal, explainable features.
- **Consequences:** ~2,900 early training rows dropped (mostly clipped labels); API requires ≥ 30 cycles.

### ADR-006 — Baseline: StandardScaler + Ridge

- **Decision:** `Pipeline([StandardScaler(), Ridge(alpha=1.0)])`.
- **Alternatives:** plain LinearRegression (unstable with correlated rolling features), constant/mean predictor (too weak to be informative).
- **Rationale:** interpretable, regularised linear bar.
- **Consequences:** can predict negative RUL → ADR-018.

### ADR-007 — Primary: HistGradientBoostingRegressor (LightGBM rejected)

- **Context:** `lightgbm 4.7.0` installs but `import` fails: missing `libomp.dylib` (verified). HGB ships with scikit-learn 1.9.1.
- **Decision:** `HistGradientBoostingRegressor(early_stopping=False, random_state=42)`, other params default.
- **Alternatives:** LightGBM (system dependency on macOS), XGBoost (extra dependency, not evaluated), RandomForest (slower, larger artifacts).
- **Rationale:** zero additional dependencies; strong tabular performance; deterministic with fixed seed.
- **Consequences:** no tuning in MVP; `early_stopping=False` avoids a row-wise internal validation split.

### ADR-008 — Validation protocol and metrics

- **Decision:** 5-fold GroupKFold by `unit` on train; final evaluation at the last cycle of each official test unit vs `RUL_FD001`; MAE and RMSE primary; NASA score secondary; no classification metrics.
- **Alternatives:** random KFold (leaks cycles of the same engine); a single hold-out split (higher variance).
- **Rationale:** matches the deployment question (unseen engines) and the official benchmark protocol.
- **Consequences:** CV labels are clipped, test truth is not → CV and test numbers are not directly comparable (documented in 05 §7).

### ADR-009 — NASA score coefficient interpretation

- **Context (verified in the bundled paper, eq. 11, §VII):** text says late predictions are penalised more and Figure 9 rises faster for positive error, but the printed equation pairs `a1 = 10` with `d < 0` and `a2 = 13` with `d ≥ 0`.
- **Decision:** `d < 0 → exp(−d/13) − 1`; `d ≥ 0 → exp(d/10) − 1` (late penalised more), `d = predicted − true`.
- **Alternatives:** literal reading of the printed equation.
- **Rationale:** matches the stated intent and figure.
- **Consequences:** secondary metric only; R5 tracks the ambiguity.

### ADR-010 — Model selection rule

- **Decision:** ship the primary model only if both its test MAE and test RMSE are lower than the baseline's; otherwise ship the baseline.
- **Alternatives:** select on CV only; select on NASA score.
- **Rationale:** simple, pre-registered, two-metric agreement.
- **Consequences:** mild selection bias (two pre-specified candidates on test); CV reported to expose it.

### ADR-011 — Artifacts: joblib + metadata JSON; no MLflow

- **Context:** MLflow 3.18.0 would add 40+ transitive packages (verified dry-run).
- **Decision:** `models/model.joblib` (sklearn Pipeline) + `models/model_metadata.json` (schema 05 §9).
- **Alternatives:** MLflow local file store; ONNX.
- **Rationale:** one model, one subset — metadata JSON gives lineage (data hash, git SHA, versions, metrics) at zero dependency cost.
- **Consequences:** no experiment UI; sklearn-version coupling (R6).

### ADR-012 — Backend: FastAPI + Pydantic v2 + uvicorn

- **Decision:** endpoints `/health`, `/predict`, `/model/info`; unified error envelope; no auth (local MVP).
- **Alternatives:** Flask (manual validation), BentoML (extra framework).
- **Rationale:** typed validation and OpenAPI out of the box.
- **Consequences:** contract in 06 is binding for tests and dashboard.

### ADR-013 — httpx2 instead of httpx for TestClient (deviation from default)

- **Context (verified):** with fastapi 0.162.2 / starlette 1.10.0, importing `fastapi.testclient` with `httpx` installed emits `StarletteDeprecationWarning: Using httpx with starlette.testclient is deprecated; install httpx2 instead.` With `httpx2==2.16.0` installed, TestClient loaded `httpx2` (not `httpx`) and worked under `-W error`, also after uninstalling `httpx`.
- **Decision:** pin `httpx2==2.16.0` as the dev test dependency; do not install `httpx`.
- **Alternatives:** keep `httpx` and suppress the warning.
- **Rationale:** avoids building tests on a deprecated path.
- **Consequences:** differs from the original plan's "httpx TestClient" wording; behaviour for tests is the same.

### ADR-014 — Frontend: Streamlit + requests over HTTP

- **Decision:** Streamlit app calling the API via a `requests`-based client; `API_BASE_URL` from env; no model imports in the dashboard.
- **Alternatives:** httpx2 in the dashboard; React frontend.
- **Rationale:** `requests` is already a streamlit 1.67.0 dependency (verified) → no new runtime dependency.
- **Consequences:** dashboard tests use a fake session object.

### ADR-015 — Packaging: one image, two compose services, mounted artifacts

- **Decision:** single `Dockerfile` (runtime deps only); compose services `api` and `dashboard` differ only by command; `./models` mounted read-only into `api`, `./data` read-only into `dashboard`.
- **Alternatives:** two slim Dockerfiles (smaller API image without streamlit/pyarrow); baking the model into the image.
- **Rationale:** one build, one dependency set, simplest to maintain; mounting keeps data-derived content out of the image and avoids rebuilds on retrain.
- **Consequences:** API image carries Streamlit's dependencies (larger image, acceptable locally); artifacts must exist on the host before `compose up`.

### ADR-016 — Tooling: ruff, exact pins, pytest, CI from M1 (deviation: CI moved earlier)

- **Decision:** ruff for lint + format; exact `==` pins in `requirements*.txt`; `pyproject.toml` for package/tool config; GitHub Actions CI created in **M1** (not at the end).
- **Rationale:** every subsequent milestone is gated by lint + tests from the start.
- **Consequences:** CI results depend on user-authorized pushes (R11).

### ADR-017 — Health band thresholds

- **Decision:** `CRITICAL` < 30 cycles, `WARNING` 30–< 80, `HEALTHY` ≥ 80; stored in config and metadata.
- **Rationale:** illustrative maintenance-planning bands within the 0–125 prediction range; not derived from engineering limits (stated in 05 §11).
- **Consequences:** configurable without retraining logic changes.

### ADR-018 — Clamp predictions at 0

- **Decision:** `predicted_rul = max(prediction, 0)` in `predict_rul`, used by both evaluation and API.
- **Rationale:** negative RUL is meaningless; one shared clamp keeps offline and online numbers identical.

### ADR-019 — Modular monolith, package `predmaint`

- **Decision:** single `src/predmaint` package with `data`, `features`, `models`, `api`, `dashboard` subpackages; thin `scripts/`.
- **Rationale:** simplest structure satisfying separation of concerns; installable for tests and Docker.

### ADR-020 — Generic sensor names

- **Context:** the paper lists 21 physical sensors, but no source states the column order matches (UNVERIFIED).
- **Decision:** name columns `s_1..s_21`, `setting_1..3`.
- **Consequences:** dashboard labels are generic; no physical claims.

### ADR-021 — Synthetic fixtures for CI; real-data tests gated

- **Decision:** CI runs on synthetic C-MAPSS-shaped data; real-data (`data`), live download (`network`), and Docker (`docker`) tests run locally and are required for milestone sign-off.
- **Rationale:** no data redistribution; fast, deterministic CI.
- **Consequences:** real-data evidence comes from local runs recorded in 14. M1 confirmed: CI does **not** download the dataset (fixture tests only), keeping CI independent of the external S3 URL.

### ADR-022 — 503 without artifact; demo reproducibility path

- **Context:** architecture-freeze amendment by the user (2026-11-08).
- **Decision:** when no model artifact is loaded, `/health`, `/predict` and `/model/info` return HTTP 503 with the standard error body (`code = "MODEL_UNAVAILABLE"`, 06 §6). Demo reproducibility = one documented train command (`python scripts/train.py`, after `python scripts/download_data.py`) followed by `docker compose up`.
- **Alternatives:** refuse to start the API without an artifact; bake the artifact into the image.
- **Rationale:** the service fails visibly and uniformly instead of crashing; one short documented path from clone to running demo.
- **Consequences:** M6 decides (task M6-T4) whether the FD001 artifact is small enough to commit so the demo can skip training; until then it is regenerated by the train command.
