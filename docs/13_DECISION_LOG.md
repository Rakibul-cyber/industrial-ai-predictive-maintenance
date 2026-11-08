# 13 — Decision Log (ADRs)

ADR-001…021 dated **2026-11-08** (Milestone 0); architecture freeze approved by the user on 2026-11-08, which accepted ADR-001…021 and added ADR-022. Status values: *Proposed*, *Accepted*, *Superseded by ADR-xxx*. Changes follow [15](15_CHANGE_CONTROL.md); accepted ADRs are never edited silently — a new ADR supersedes them.

| ID | Title | Status |
|---|---|---|
| ADR-001 | MVP scope: FD001 RUL regression only | Accepted (2026-11-08); naming consequence superseded by ADR-033 |
| ADR-002 | Python 3.13 | Accepted (2026-11-08) |
| ADR-003 | Data source, checksums, no redistribution | Accepted (2026-11-08) |
| ADR-004 | RUL target with clip 125 | Accepted (2026-11-08) |
| ADR-005 | Feature strategy | Accepted (2026-11-08) |
| ADR-006 | Baseline: StandardScaler + Ridge | Accepted (2026-11-08) |
| ADR-007 | Primary: HistGradientBoostingRegressor (LightGBM rejected) | Accepted (2026-11-08); "no tuning" amended by ADR-026 |
| ADR-008 | Validation protocol and metrics | Accepted (2026-11-08) |
| ADR-009 | NASA score coefficient interpretation | Accepted (2026-11-08) |
| ADR-010 | Model selection rule | Superseded by ADR-024 |
| ADR-011 | Artifacts: joblib + metadata JSON; no MLflow | Accepted (2026-11-08) |
| ADR-012 | Backend: FastAPI + Pydantic v2 + uvicorn | Accepted (2026-11-08) |
| ADR-013 | httpx2 instead of httpx for TestClient | Accepted (2026-11-08) |
| ADR-014 | Frontend: Streamlit + requests over HTTP | Accepted (2026-11-08) |
| ADR-015 | Packaging: one image, two compose services, mounted artifacts | Accepted (2026-11-08); artifact mount superseded by ADR-030/032 |
| ADR-016 | Tooling: ruff, exact pins, pytest, CI from M1 | Accepted (2026-11-08) |
| ADR-017 | Health band thresholds | Superseded by ADR-025 |
| ADR-018 | Clamp predictions at 0 | Superseded by ADR-027 |
| ADR-019 | Modular monolith, package `predmaint` | Accepted (2026-11-08) |
| ADR-020 | Generic sensor names | Accepted (2026-11-08) |
| ADR-021 | Synthetic fixtures for CI; real-data tests gated | Accepted (2026-11-08) |
| ADR-022 | 503 without artifact; demo = train command + `docker compose up` | Accepted (2026-11-08); `/health` part amended by ADR-028 |
| ADR-023 | M2/M3 re-scope; per-candidate artifacts; feature list inside the pipeline | Accepted (2026-11-08, user-directed) |
| ADR-024 | Served model selected by CV RMSE (tie-break CV MAE) | Accepted (2026-11-08, user-directed) |
| ADR-025 | Health status thresholds 20 / 60 | Accepted (2026-11-08, user-directed) |
| ADR-026 | One bounded 12-configuration search for the primary model | Accepted (2026-11-08, user-directed) |
| ADR-027 | Clamp predictions to [0, RUL_CAP] | Accepted (2026-11-08, user-directed) |
| ADR-028 | `/health` always 200; readiness via `model_loaded` (amends ADR-022) | Accepted (2026-11-08, user-directed) |
| ADR-029 | Commit one FD001 test engine as demo payloads (exception to ADR-003 policy) | Accepted (2026-11-08, user-directed) |
| ADR-030 | Commit the served model artifact | Accepted (2026-11-08, user-directed) |
| ADR-031 | Dataset in containers: `./data` bind mount, dashboard downloads on first start | Accepted (2026-11-08, user-directed) |
| ADR-032 | One image (single stage, digest-pinned, model baked in), two compose services | Accepted (2026-11-08, user-directed) |
| ADR-033 | Project renamed "Industrial AI — Predictive Maintenance Platform"; anomaly detection stays an unauthorized extension | Accepted (2026-11-08, user-directed, CR-007) |
| ADR-034 | Prediction intervals: two quantile HGB models (q0.1/q0.9), separate artifact, graceful fallback | Accepted (2026-11-08, user-directed, CR-008) |
| ADR-035 | Minimal anomaly detection: Mahalanobis distance to a healthy reference, OOF threshold | Accepted (2026-11-08, user-directed, CR-010) |
| ADR-036 | *Unused.* Reserved for restoring the full project title if FR-019 (M9) or FR-021 (M11) were met; neither was, so no ADR-036 exists. | — |
| ADR-037 | Conformalized quantile regression (CQR) for the 80 % interval; one additive correction Q from OOF predictions | Accepted (2026-11-08, user-directed, CR-011) |
| ADR-038 | Smoothed anomaly score: per-engine EWM (half-life 10, fixed in advance) of the Mahalanobis distance, OOF smoothed threshold | Accepted (2026-11-08, user-directed, CR-012); **serving clause superseded by ADR-039** |
| ADR-039 | Serve the raw anomaly score; keep smoothing as a disabled, documented experiment (`smoothing_enabled: false`) | Accepted (2026-11-09, user-directed, CR-012 close-out) |

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
- **M2 confirmation (2026-11-08, evidence in 04 §5a):** the ≤ 2-unique-values rule drops `s_1, s_5, s_6, s_10, s_16, s_18, s_19` and keeps 14 sensors (smallest kept unique count 13). Exactly 2,900 rows were dropped (17,731 of 20,631 remain). Decision unchanged.

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

### ADR-010 — Model selection rule (Superseded by ADR-024)

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

### ADR-017 — Health band thresholds (Superseded by ADR-025)

- **Decision:** `CRITICAL` < 30 cycles, `WARNING` 30–< 80, `HEALTHY` ≥ 80; stored in config and metadata.
- **Rationale:** illustrative maintenance-planning bands within the 0–125 prediction range; not derived from engineering limits (stated in 05 §11).
- **Consequences:** configurable without retraining logic changes.

### ADR-018 — Clamp predictions at 0 (Superseded by ADR-027)

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

### ADR-023 — M2/M3 re-scope; per-candidate artifacts; feature list inside the pipeline

- **Context:** the Milestone 2 authorization (2026-11-08) moved baseline training and evaluation into M2 and required `models/baseline.joblib`, `models/baseline_metadata.json`, a committed `reports/metrics_baseline.json`, and that the kept-sensor list be stored inside the saved preprocessing artifact so inference never recomputes it. Doc 05 §9 described only the served `model.joblib` / `model_metadata.json`.
- **Decision:** (1) M2 = labels, features, baseline CV + test evaluation + artifacts; M3 = primary model, ADR-010 selection, served artifact, inference contract (CR-002). (2) Each candidate is saved as `models/<name>.joblib` + `models/<name>_metadata.json` (gitignored) plus a committed `reports/metrics_<name>.json`; M3 writes the served `models/model.joblib` + `models/model_metadata.json` exactly as 05 §9 and the API contract specify. (3) Every pipeline starts with `ColumnTransformer([("features", "passthrough", feature_names)])`, so the artifact itself carries the training feature list (and thus the kept sensors) and rejects inputs missing those columns; the list is also in metadata (`required_sensors`, `feature_names`).
- **Alternatives:** keep the original milestone split; pickle a custom wrapper object holding the sensor list (adds a project class to the pickle contract).
- **Rationale:** follows the user's explicit instruction; built-in sklearn components only; the served-model contract (05 §9, 06) is unchanged.
- **Consequences:** `reports/` directory added (07); `models/*_metadata.json` ignored except `model_metadata.json`; 01/08/09 traceability updated.

### ADR-024 — Served model selected by CV RMSE (supersedes ADR-010)

- **Context:** ADR-010 selected on official test MAE and RMSE, which uses the held-out test set for a model decision. The M3 authorization (2026-11-08) replaced it (CR-003).
- **Decision:** the served model is the candidate with the lowest 5-fold GroupKFold CV RMSE mean. Tie-breaks: CV MAE mean, then the simpler model. Official test metrics (unclipped truth primary, clipped secondary) and the NASA score are reported for every candidate but do not drive the choice. A test ranking that disagrees with the CV ranking is stated explicitly, and the CV choice stands. Candidates must share data hashes, config, feature names and kept sensors.
- **Alternatives:** ADR-010 (test-based); NASA score (secondary metric only, ADR-009).
- **Rationale:** keeps the official test set a clean held-out report.
- **Consequences:** `scripts/train.py --select`; promoted metadata carries `selected_by: "cv_rmse"` and a `comparison` block; `reports/model_comparison.{json,md}`. M3 outcome: primary selected, rankings agree (05 §7c).

### ADR-025 — Health status thresholds (supersedes ADR-017)

- **Context:** user-specified bands for the API/dashboard (2026-11-08).
- **Decision:** `HEALTH_BANDS = {"critical_below": 20, "warning_below": 60}` in `config.py`. Predicted RUL < 20 → `"critical"`, 20 ≤ RUL < 60 → `"warning"`, ≥ 60 → `"healthy"`. The response field is `health_status` (lowercase values).
- **Alternatives:** ADR-017's 30 / 80 uppercase bands. No documented reason was found to deviate from the user's numbers.
- **Consequences:** 05 §10, 06 §3.4 updated; illustrative thresholds, not engineering limits.

### ADR-026 — One bounded search for the primary model (amends ADR-007)

- **Context:** ADR-007 fixed HGB hyperparameters with no tuning. The M3 authorization allowed one bounded search of at most ~12 configurations scored by GroupKFold CV RMSE, with no second round (CR-003).
- **Decision:** exactly 12 configurations — (learning_rate, max_iter) ∈ {(0.1, 100), (0.05, 200), (0.03, 400)} × max_leaf_nodes ∈ {15, 31} × l2_regularization ∈ {0.0, 1.0}. `early_stopping=False` and `random_state=42` throughout; no scaler. Winner = lowest CV RMSE mean (tie-break CV MAE). The grid, every result and the winner go into metadata and doc 05 §7b.
- **Alternatives:** varying early stopping — rejected because its internal validation split is row-wise and leaks across engines (ADR-007 rationale).
- **Rationale:** a small, pre-registered search; no test-set involvement.
- **Consequences:** primary training ≈ 80 s wall (≈ 9.7 CPU-minutes multi-threaded) on the dev machine; winner `lr=0.05, max_iter=200, max_leaf_nodes=15, l2=0.0`. No further tuning is permitted without a new ADR.

### ADR-027 — Clamp predictions to [0, RUL_CAP] (supersedes ADR-018)

- **Context:** the M3 inference spec requires `predicted_rul` to be non-negative and capped at `RUL_CAP`. ADR-018 clamped at 0 only. Offline evaluation and inference must share one post-processing step.
- **Decision:** `predict_from_features(pipeline, features, rul_cap)` clips to `[0, rul_cap]`; it is used by CV, test evaluation and `predict_rul`.
- **Rationale:** training labels never exceed `rul_cap`, so predictions above it are extrapolation; one shared clamp keeps offline and online numbers identical.
- **Consequences:** baseline numbers changed slightly versus M2 (test RMSE 19.205 → 19.144; one test prediction was 130.58), disclosed in 05 §7a.

### ADR-028 — `/health` always 200; readiness via `model_loaded` (amends ADR-022)

- **Context:** ADR-022 / doc 06 said `/health` returns 503 without a model. The M4 authorization (2026-11-08) specifies `/health` → 200 with `model_loaded: false` when `MODEL_PATH` points nowhere (CR-004).
- **Decision:** `/health` always returns 200 with `status` (`ok`/`degraded`), `model_loaded`, `model_version` (null without a model) and `api_version`. `/predict` and `/model/info` keep returning 503 `MODEL_UNAVAILABLE` without a model (the rest of ADR-022 is unchanged).
- **Alternatives:** keep 503 on `/health` (readiness-style).
- **Rationale:** separates liveness (process up) from readiness (`model_loaded`), and the dashboard can show "API up, no model".
- **Consequences:** readiness checks must read `model_loaded`. The planned compose healthcheck (10 §4) is changed accordingly — a status-code-only check would report healthy without a model. 01 FR-010, 02 §8, 06, 10, 11 AC-21 updated.

### ADR-029 — Commit one FD001 test engine as demo payloads (exception to the ADR-003 policy)

- **Context:** ADR-003 / 04 §11: raw data never committed, test fixtures synthetic. The M4 authorization asks for `examples/predict_request.json` from real FD001 test engine 1 (all cycles) and `examples/predict_request_min.json` (30 cycles), committed and reused by README, tests and the dashboard. NASA states no licence for the dataset and requests citation (04 §11).
- **Decision:** commit exactly those two payloads (31 and 30 rows of engine 1, JSON) plus `examples/expected_predictions.json`. No other NASA rows are committed; `data/` stays ignored; the citation stays in README.
- **Alternatives:** synthetic demo payloads (not "real"); generate at runtime from the downloaded data (demo unusable without the dataset).
- **Rationale:** a small, attributed excerpt makes the demo checkable without downloading the dataset.
- **Consequences:** R1 (licence) now also covers this excerpt; if NASA terms ever forbid redistribution, remove `examples/*.json` and regenerate locally with `scripts/make_request_payload.py`.

### ADR-030 — Commit the served model artifact (decides M6-T4)

- **Context:** M6 authorization (2026-11-08). The served `models/model.joblib` is 451,385 bytes; `model_metadata.json` was already committed in `65db98d`. ADR-022 wanted demo reproducibility = one train command + `docker compose up`.
- **Decision:** commit `models/model.joblib` together with `models/model_metadata.json` (`.gitignore` exception `!models/model.joblib`), so a fresh clone runs `docker compose up` with no training step. Candidate files (`baseline*`, `primary*`) stay ignored. Retraining = the train/select commands, then commit both files together (10 §6).
- **Evidence:** before committing, `model_metadata.json` was confirmed to be the committed version, its `candidate_model_version` (`hgb-20261108T150441Z`) to equal the local primary candidate, and `model.joblib` to be a byte copy of `primary.joblib` (SHA-256 `39bfa6fd…28c1`).
- **Mitigations (pickle risk R6/R15):** scikit-learn pinned exactly (1.9.1); `/model/info.sklearn_version_match`; `tests/test_model_artifact.py` fails if the installed scikit-learn ≠ metadata version and checks that the committed model reproduces `examples/expected_predictions.json` (both run in CI without data); `tests/test_real_data.py` checks the official-test RMSE when the dataset is present.
- **Alternatives:** train inside the image build or at container start (slower, needs the dataset in the API); Git LFS (unnecessary at 451 KB).
- **Consequences:** binary changes appear in git history; reviewers must treat model-file diffs like code (R15).

### ADR-031 — Dataset in containers

- **Context:** the dashboard's explorer and trajectory read raw FD001 (M5); NASA data is not committed beyond the ADR-029 excerpt; the API does not need the dataset.
- **Decision:** `./data` is bind-mounted into both services (dashboard read-write, API read-only). The dashboard's compose command runs `scripts/download_data.py` (idempotent, SHA-256-verified, verified S3 URL) before `streamlit run`; a download failure only logs a warning. On Linux the dashboard runs as `APP_UID:APP_GID` so it can write the host-owned mount.
- **Evidence (M6, arm64):** first start → "Downloaded, verified and extracted the archive."; restart → "Dataset already present and checksums match; nothing downloaded."; host-side checksums equal 04 §3.
- **Alternatives:** named volume (no host reuse); download at image build (puts NASA data in the image); require a manual host download (not self-contained).
- **Consequences:** first start needs network; later starts are offline-capable; a host that already ran Option B reuses its `./data`.

### ADR-032 — One image, two compose services; model baked in (supersedes ADR-015's artifact mount)

- **Context:** ADR-015 planned one image + two services with `./models` mounted read-only. With the model committed (ADR-030), the image can carry it, so it runs anywhere without a mount.
- **Decision:** one `Dockerfile`, single stage, `FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90`; `pip install --only-binary=:all:` (verified: all wheels available; no compiler in the final image, so no multi-stage build needed); non-root `app` (1000); `src/`, `scripts/`, `examples/` and the two committed model files copied in; no `./models` mount. Compose services `api` (default CMD) and `dashboard` (Streamlit command) share `predmaint:latest`; healthchecks via `python -c` + urllib (`model_loaded` must be true for `api`; `/_stcore/health` for `dashboard`); `dashboard` `depends_on: api: service_healthy`; ports bound to 127.0.0.1; `restart: unless-stopped`.
- **Evidence:** image 1.1 GB on disk (`docker images`), 242,889,009 bytes content size; build 57 s cold on arm64.
- **Alternatives:** two slim images (smaller API image, two builds); keep the `./models` mount (retrain without rebuild, but the image is not self-contained).
- **Consequences:** retraining requires `docker compose up --build`; the API image carries Streamlit/pyarrow (size).

### ADR-033 — Project renamed; anomaly detection remains an unauthorized extension (supersedes ADR-001's naming consequence)

- **Context:** CR-007. The original title "Industrial AI — Predictive Maintenance & Anomaly Detection Platform" (doc 00) names a capability that was never implemented. ADR-001 scoped the MVP to RUL regression and noted that the "name reflects platform direction"; for a portfolio project, a title that overstates scope is a credibility defect.
- **Decision:** the project is named **"Industrial AI — Predictive Maintenance Platform"**. Anomaly detection remains an optional, **unauthorized** extension, mentioned only as planned / NOT IMPLEMENTED (doc 00 out-of-scope list, CLAUDE.md candidate milestones, 16 §8, README "Limitations and future work").
- **Alternatives:** implement anomaly detection (scope creep, R13); keep the old title with a disclaimer (still misleading at a glance).
- **Consequences:** naming only; no code, API, model or test change (the code titles never contained the old title). ADR-001's scope decision stands; only its naming consequence is superseded.

### ADR-034 — Prediction intervals via quantile gradient boosting (Milestone 8)

- **Context:** CR-008. The served point model gives no uncertainty estimate (16 §7).
- **Decision:**
  - **Method:** two extra `HistGradientBoostingRegressor(loss="quantile")` models at `quantile=0.1` and `0.9`, forming a nominal **80 %** interval. They use the identical feature pipeline (select step + 43 features), the same train-row filter and seed, and the served primary model's hyperparameters, read from `models/model_metadata.json`. There is no search. The point model is not retrained and its artifact does not change.
  - **Post-processing:** clamp both bounds to [0, RUL_CAP]; then `lower = min(lower, point)` and `upper = max(upper, point)`. The number of adjusted predictions (reorders) is counted and reported.
  - **Artifact:** `models/model_intervals.joblib` (dict of the two pipelines) + `models/model_intervals_metadata.json` (`for_model_version`, quantiles, hyperparameters, evaluation). Both are committed, copied into the image and loaded at startup next to `MODEL_PATH`. If the artifact is absent, unloadable, or its `for_model_version` ≠ the loaded point model, the API logs a warning and returns null interval fields; point predictions are unchanged in every case.
  - **API (additive):** `/predict` gains `rul_lower`, `rul_upper`, `interval_level`; `/health` gains `intervals_loaded`; `/model/info` gains `intervals` (or null); `api_version` 0.2.0.
  - **Evaluation:** coverage and width on 5-fold GroupKFold out-of-fold predictions and on the official test set (last cycle), overall and by true-RUL bucket, against unclipped and clipped truth. Nothing is tuned toward 80 %.
- **Alternatives:** conformal prediction on CV residuals (distribution-free marginal coverage, but a constant or residual-scaled width needs extra calibration data handling); bootstrap ensembles (several times the training cost); NGBoost/probabilistic models (new dependency).
- **Rationale:** reuses the existing model family and pipeline, needs no new dependency, and produces input-dependent widths.
- **Consequences:**
  - Quantile models are trained on capped labels and bounds are clamped at 125, so engines whose true RUL exceeds 125 cannot be covered against unclipped truth; this is expected and reported, not fixed.
  - Quantile crossing is possible and is handled by the ordering rule.
  - Retraining order: train/select the point model first, then `--model intervals`; commit the four model files together.

### ADR-035 — Minimal anomaly detection: Mahalanobis distance to a healthy reference (Milestone 9)

- **Context:** CR-010. C-MAPSS has no anomaly labels, so anomaly detection is framed as unsupervised health monitoring: an "anomaly" is a deviation from a healthy-state reference, and the method is judged by whether its score tracks degradation.
- **Decision:**
  - **Healthy reference:** training rows that pass the `WINDOW_SIZE` filter and have true (uncapped) RUL ≥ `RUL_CAP` (125). That is the part of each engine's life where the point model's target is flat at the cap. **Correction (M9 measurement, 2026-11-08):** an engine contributes only if its life is ≥ 155 cycles (cycle ≥ 30 and RUL ≥ 125); the measured reference is 5,329 rows from **90 of 100** engines, not every engine as first written.
  - **Features:** the 14 kept sensors' `*_roll_mean` columns from the shared feature function; no new features.
  - **Model:** `StandardScaler` → `sklearn.covariance.LedoitWolf`, both fitted on healthy rows; score = Mahalanobis distance.
  - **Threshold:** 99th percentile of healthy-row scores computed out-of-fold (5-fold GroupKFold by engine), so it is not fitted on the rows it is evaluated on. `normalized_score = score / threshold`; `is_anomalous = normalized_score > 1`.
  - **Fallback (at most once):** if the median per-engine Spearman(score, cycle) on train OOF is below 0.5, try `IsolationForest` (default parameters) on the same features and keep the better one on that metric, recording both.
  - **Artifact:** `models/model_anomaly.joblib` (scaler, covariance, feature list, threshold) + metadata (`for_model_version` of the served point model, healthy-row count, threshold, evaluation). Committed, baked into the image, loaded at startup; absent or mismatched → warning and 503 `ANOMALY_UNAVAILABLE` on `/anomaly` only.
  - **API (additive):** `POST /anomaly` (same request schema and validation as `/predict`, optional `?include_series=true`); `/health.anomaly_loaded`; `/model/info.anomaly`; `api_version` 0.3.0.
- **Alternatives:**
  - Healthy reference = first N cycles of each engine: calendar age, not degradation. Engines start with different initial wear, so a fixed N mixes healthy and early-degraded states unevenly; the RUL ≥ cap definition aligns the reference with where the RUL target itself is flat.
  - Autoencoders or deep models: excluded by scope.
  - One-class SVM: extra tuning, no clear benefit for 14 features.
- **Rationale:** a transparent, few-parameter, dependency-free baseline that reuses the shared feature path; Ledoit-Wolf shrinkage keeps the covariance well conditioned.
- **Evaluation / acceptance (FR-019):** trend (median per-engine Spearman > 0.7 on train OOF), false-alarm rate on healthy rows, detection timing (3 consecutive cycles above threshold), test flags by true-RUL bucket, agreement with the RUL model. Not tuned toward the target.
- **Consequences:** the score measures distance from a population-level healthy state, not engine-specific baselines; without labels the evaluation shows the score tracks simulated degradation, not that it detects real faults.
- **Outcome (M9, 2026-11-08):** fallback not triggered (median 0.680 ≥ 0.5); FR-019 trend target not met (0.680 ≤ 0.7). Per the M9 authorization the project title stays as is and no follow-up ADR (title restoration) was added.

### ADR-037 — Conformalized quantile regression for the 80 % interval (Milestone 10)

- **Context:** CR-011. The M8 interval (ADR-034) covers 0.737 of out-of-fold rows against the capped target, for a nominal 0.80.
- **Decision:**
  - **Method (standard split-conformal CQR, α = 0.2):** for calibration rows with target y and **raw** quantile-model bounds [lo, hi] (before clamping and ordering), nonconformity E = max(lo − y, y − hi). With n calibration rows and k = ⌈(n + 1)(1 − α)⌉, Q = the k-th smallest E (k > n → Q = +∞). Calibrated interval = [lo − Q, hi + Q], then the unchanged ADR-034 post-processing: clamp to [0, RUL_CAP], `lower = min(lower, point)`, `upper = max(upper, point)`. The correction is applied in that same function (`order_bounds`), so training evaluation and the API share it.
  - **Target:** the **capped** RUL (what the quantile models learn) for calibration and for the acceptance criterion. Coverage against unclipped truth is also reported; for true RUL > 125 it stays 0 by construction (bounds ≤ 125).
  - **Data:** the existing 5-fold GroupKFold out-of-fold quantile (and point) predictions on all eligible training rows; no extra model is trained.
  - **Honest evaluation:** training **engines** are split at random (`numpy.random.default_rng(random_seed=42)` permutation of the sorted unit ids; first half = calibration, rest = evaluation). Q_eval is computed on calibration-half rows; coverage and width are measured on evaluation-half rows before and after calibration → **the FR-020 number**. The shipped Q_final is computed on **all** out-of-fold rows; test-set coverage and width are reported with Q_final. Both Q values are recorded.
  - **Pre-registered acceptance (FR-020):** evaluation-half coverage of the calibrated interval against the capped target within **0.80 ± 0.03** (0.77 ≤ c ≤ 0.83). Test coverage is reported, not the criterion. A miss is reported, not tuned.
  - **Artifact:** `models/model_intervals_metadata.json` gains `conformal_q`, `alpha`, `calibration` (method, target, split, evaluation results); `model_intervals.joblib` is unchanged (checksum-verified). Metadata without `conformal_q` behaves exactly as v1.1 (M8) and reports `calibrated: false`.
  - **API (additive):** `/predict.interval_calibrated` (bool, `null` without intervals); `/model/info.intervals` gains `calibrated`, `conformal_q`, `alpha`, calibration coverage results and the caveat; `api_version` 0.4.0. Dashboard: trajectory band caption only.
- **Sign of Q:** standard CQR allows Q < 0, which would narrow an over-covering interval. The M8 interval under-covers, so a positive Q is expected; the sign is recorded, and widen-only behaviour is a property of Q ≥ 0, not an extra clamp.
- **Caveat (exchangeability):** the split-conformal guarantee assumes exchangeable rows. Rows of one engine are strongly correlated, and calibration rows come from 50 engines, so the guarantee is approximate. The engine-level split makes the evaluation honest (no engine is in both halves), but the effective sample size is closer to the number of engines than to the number of rows.
- **Alternatives:** a per-bucket or scaled correction (more parameters, more tuning surface); retraining quantiles at other levels (tuning toward the target); conformal on point residuals (discards the input-dependent widths).
- **Rationale:** one parameter, no new dependency, no retraining, reuses predictions the pipeline already makes, and is the documented fix from 16 §9.
- **Consequences:** intervals widen by Q everywhere (before clamping); the > 125 structural zero against unclipped truth is unchanged; retraining order unchanged (`--select`, then `--model intervals`, which now also calibrates).
- **Outcome (M10, 2026-11-08):** Q_eval = 1.320 (50 calibration engines) → evaluation-half coverage 0.734 → **0.777** (capped target): FR-020 met at the lower edge of the band. Q_final = 1.742 (all 17,731 OOF rows) ships. Test coverage 0.820 → 0.850 (capped), 0.740 → 0.750 (unclipped). Q is positive, so no interval narrowed. Q_eval < Q_final and the evaluation half still sits below 0.80, which is the engine-level (non-exchangeable) effect named in the caveat.

### ADR-038 — Smoothed anomaly score: EWM with a pre-registered half-life (Milestone 11)

- **Context:** CR-012. The raw M9 score missed FR-019 (median train-OOF trend 0.680 ≤ 0.7). The authorization tests one remedy for the score's cycle-to-cycle noise.
- **Decision:**
  - **Smoothing:** per engine, `pandas.Series.ewm(halflife=10, adjust=True).mean()` over the per-cycle Mahalanobis distances (raw, before normalisation), in cycle order, starting at the first full-window cycle. Half-life **10 cycles**, fixed in advance; no other value is tried.
  - **Threshold:** the 99th percentile of the **smoothed** healthy-reference scores from the same out-of-fold procedure as M9 (5-fold GroupKFold by engine; smoothing applied per engine to that engine's OOF scores). `normalized_score = smoothed / smoothed_threshold`; `is_anomalous = normalized_score > 1`. The raw threshold is kept and reported.
  - **Unchanged:** scaler, Ledoit-Wolf covariance, feature list, raw threshold. The detector object in `model_anomaly.joblib` is not modified; the smoothing config and smoothed threshold live in the metadata (`smoothing`, `smoothed`), like `conformal_q` in ADR-037. Fitted parameters are verified identical to the M9 artifact. Metadata without `smoothing` → M9 behaviour (`smoothed: false`).
  - **Evaluation:** all five M9 metrics for the smoothed score next to the raw score (train OOF and test, same buckets), plus the test-trend breakdown by true RUL at cutoff (≤ 60, 61–125, > 125) for both scores.
  - **Pre-registered acceptance (FR-021):** median per-engine Spearman(smoothed score, cycle) on train OOF **> 0.7** AND false-alarm rate on healthy OOF rows (smoothed threshold) **≤ 0.02**. Both must hold.
  - **API (additive):** `/anomaly` gains `raw_normalized_score`, `smoothed`, `cycles_in_smoothing`; series entries gain `raw_normalized_score`; `anomaly_score`/`threshold`/`normalized_score`/`is_anomalous` refer to the smoothed score when smoothing is configured. `/model/info.anomaly` gains the smoothing config and both evaluations. `api_version` 0.5.0.
- **Known in advance (stated before computing):**
  - The false-alarm condition is close to automatic: the threshold is the 99th percentile of the same OOF healthy rows, so the rate is ≈ 0.01 by construction (M9 raw: 0.0101). The trend condition is the binding one.
  - Smoothing removes cycle-to-cycle noise, so a per-engine rank correlation with cycle rises whenever a trend exists; it adds no information that the raw series lacks. Smoothing also adds lag, so detection can only come at or after the raw detection for the same threshold logic.
  - **History dependence:** the smoothed last-cycle score depends on how many cycles the client sends (EWM memory). Clients should send the full available history; `cycles_in_smoothing` reports how many full-window cycles were smoothed.
- **Alternatives:** other half-lives or a search over them (tuning against the acceptance metric, excluded); a rolling median; refitting the detector on smoothed features (changes the detector).
- **Rationale:** one parameter, fixed in advance, no new dependency, detector untouched.
- **Consequences:** if FR-021 is met, the title may be restored via ADR-036 (reserved); if not, the title stays and both attempts are reported.
- **Outcome (M11, 2026-11-08):** smoothed threshold 5.0906 (raw 5.4036). FR-021 **NOT MET**: trend median 0.693 ≤ 0.7, false-alarm 0.0101 ≤ 0.02. Smoothing also delayed detection (never flagged 0.160 vs 0.020). Title unchanged; ADR-036 stays unused. Per the authorization the served `is_anomalous` uses the smoothed score; removing `smoothing`/`smoothed` from the metadata restores the M9 behaviour.

### ADR-039 — Serve the raw anomaly score; smoothing retained as a disabled experiment (Milestone 11 close-out)

- **Context:** ADR-038's remedy did not meet FR-021 (smoothed trend median 0.693 ≤ 0.7), and it made detection worse: training engines never flagged rose from 2 % to 16 %, median RUL at first detection fell from 49 to 44 cycles, and fewer near-failure test engines were flagged (0–20: 0.750 → 0.688; 21–60: 0.391 → 0.348).
- **Decision:** supersedes ADR-038's **serving** clause.
  - `/anomaly` serves the **raw** normalised score for `is_anomalous`, as in M9.
  - The anomaly metadata carries an explicit `smoothing_enabled: false`, written by `scripts/train.py --model anomaly` (`SMOOTHING_ENABLED = False` in `models/anomaly.py`). It is a flag rather than deleting the keys, so the smoothing config, smoothed threshold and smoothed evaluation stay in the metadata, the report and `/model/info.anomaly` as a documented negative result.
  - The smoothing implementation (`ewm_smooth`, `smooth_per_engine`, side-by-side evaluation) stays in code and is tested through an explicitly enabled variant.
  - Response fields `smoothed` (false), `cycles_in_smoothing` (null) and `raw_normalized_score` (= `normalized_score`) remain present but inactive.
  - Dashboard: badge and caption use the served raw score. The trajectory chart shows the raw line only, because the series carries no separate smoothed values when smoothing is disabled, and its caption says so. The model-info table labels the smoothed column "smoothed: evaluated, not served".
- **Alternatives:** delete the smoothing keys (loses the negative result from the served metadata); keep serving the smoothed score (worse detection, no criterion met).
- **Rationale:** serve the score with the better measured detection behaviour; keep the evidence of the rejected remedy reproducible.
- **Consequences:** the history dependence of ADR-038 no longer applies to served responses. Metadata without the flag but with a `smoothing` config (M11 before this ADR) is still treated as enabled, so the flag must stay in the metadata. The detector and its artifact are unchanged.
