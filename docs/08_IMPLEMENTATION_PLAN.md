# 08 — Implementation Plan

**All milestones M0–M7 are COMPLETE (MVP, 2026-11-08); post-MVP Milestone 8 (prediction intervals) is COMPLETE.** Milestone 0 (planning) was approved with the architecture freeze on 2026-11-08. Milestones run strictly in order; each starts only after explicit user authorization (CLAUDE.md).

**Critical path:** M1 → M2 → M3 → M4 → M5 → M6 → M7 (fully sequential: features need data, training needs features, API needs the artifact, dashboard needs the API, compose needs both services, final docs need everything). Total estimate ≈ 22 h.

Common setup used by all verification commands (created in M1):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .
```

Test gating (09 §3): tests marked `data` skip when FD001 files are absent; `network` and `docker` tests skip unless `RUN_NETWORK_TESTS=1` / `RUN_DOCKER_TESTS=1`.

---

## M1 — Scaffold, tooling, CI, data acquisition & validation

- **Objective:** installable package, pinned deps, CI, and a verified, validated FD001 data load.
- **Dependencies:** M0 approval.
- **Requirements:** FR-001, FR-002, FR-003, NFR-004, NFR-006, NFR-008.
- **Files to create:** `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `.env.example`, `.github/workflows/ci.yml`, `models/.gitkeep`, `src/predmaint/__init__.py`, `src/predmaint/config.py`, `src/predmaint/data/__init__.py`, `src/predmaint/data/download.py`, `src/predmaint/data/load.py`, `src/predmaint/data/validate.py`, `scripts/download_data.py`, `tests/conftest.py`, `tests/test_download.py`, `tests/test_load.py`, `tests/test_validate.py`, `tests/test_real_data.py`.
- **Files to modify:** `docs/04_DATASET_SPECIFICATION.md` (§11 licence outcome), `docs/14_PROJECT_STATUS.md`, `docs/13_DECISION_LOG.md` (if any decision changes).
- **Tasks:**
  - [x] M1-T1 Write `pyproject.toml` (src layout, `requires-python = ">=3.13,<3.14"`, ruff config, pytest markers `data`/`network`/`docker`) and the two requirements files with the exact pins from [03](03_TECH_STACK.md).
  - [x] M1-T2 `config.py`: `DATA_DIR`, `MODEL_DIR`, `RUL_CAP=125`, `WINDOW_SIZE=30`, `RANDOM_SEED=42`, `CV_FOLDS=5`, band thresholds; env overrides for paths, `LOG_LEVEL`, `API_BASE_URL`, `API_TIMEOUT_SECONDS`.
  - [x] M1-T3 `download.py`: URL + SHA-256 constants from 04 §1/§3; skip download if valid files exist; nested extraction restricted to the 14 expected names.
  - [x] M1-T4 `load.py`: `sep=r"\s+"`, `header=None`, names per 04 §4; int casts for `unit`, `cycle`.
  - [x] M1-T5 `validate.py`: checks 1–6 of 04 §6, raising `DataValidationError` with a specific message.
  - [x] M1-T6 Tests: synthetic fixtures incl. trailing double spaces; every validation rule has a failing-case test; real-data shape test (marker `data`); live download test (marker `network`).
  - [x] M1-T7 (workflow file done; first run pending an authorized push) CI workflow (ubuntu-latest, `actions/setup-python` with `python-version: "3.13"`, install dev deps, `ruff check .`, `ruff format --check .`, `pytest`). **UNVERIFIED item:** runner provides Python 3.13 — confirmed on the first CI run, which requires a user-authorized push.
  - [x] M1-T8 **UNVERIFIED item — dataset licence:** read the NASA PCoE repository page and data.nasa.gov entry for usage/citation terms; record exact wording + URL in 04 §11; if no terms are found, record that and keep the no-redistribution policy.
  - [x] M1-T9 **Re-verify pins in project context:** fresh `.venv` from `requirements-dev.txt`, `pip check`, import smoke.
- **Verification commands:**

  ```bash
  python scripts/download_data.py && python scripts/download_data.py   # 2nd run: no download
  shasum -a 256 data/raw/CMAPSSData/{train,test,RUL}_FD001.txt
  python -m predmaint.data.validate                                     # full FD001 contract check
  pip check
  ruff check . && ruff format --check .
  pytest -v
  RUN_NETWORK_TESTS=1 pytest -v -m network
  ```

- **Acceptance criteria:** checksums match 04 §3; all tests pass (real-data test executed, not skipped, locally); ruff clean; `pip check` clean; licence outcome recorded; CI workflow file present (CI run result recorded in 14 once a push is authorized).
- **Estimated effort:** 4 h · **Status:** **COMPLETE** (2026-11-08; committed and pushed, CI green; evidence in 14).

## M2 — Labels, features, baseline training and evaluation (re-scoped by CR-002)

- **Objective:** leakage-free labels and features via one reusable code path; a trained, cross-validated, test-evaluated Ridge baseline with saved artifact and committed metrics report.
- **Dependencies:** M1.
- **Requirements:** FR-004, FR-005, FR-006, FR-008 (baseline), NFR-007.
- **Files to create:** `src/predmaint/data/labels.py`, `src/predmaint/features/__init__.py`, `src/predmaint/features/build.py`, `src/predmaint/models/__init__.py`, `registry.py`, `metrics.py`, `train.py`, `artifacts.py`, `predict.py` (clamp step only; under `src/predmaint/models/`), `scripts/train.py`, `reports/metrics_baseline.json` (generated, committed), `tests/test_labels.py`, `tests/test_features.py`, `tests/test_metrics.py`, `tests/test_train.py`.
- **Files to modify:** `src/predmaint/data/validate.py` (`expected=None` for synthetic runs), `tests/conftest.py` (synthetic training fixture), `tests/test_real_data.py` (sensor list), `.gitignore`, `README.md`, `CLAUDE.md` (warnings policy), docs 01, 04, 05, 07, 08, 09, 12, 13, 14, 15.
- **Tasks:**
  - [x] M2-T1 Near-constant sensors on real `train_FD001`: table recorded in 04 §5a; ADR-005 rule confirmed (7 dropped, 14 kept).
  - [x] M2-T2 `labels.py`: `add_rul(df, rul_cap)`.
  - [x] M2-T3 `build.py`: `sensor_stats`, `select_sensors`, `feature_names`, `build_features(df, sensors, window_size, keep)`, `last_rows`.
  - [x] M2-T4 `registry.py` (`make_baseline`, `make_model`), `metrics.py` (`mae`, `rmse`, `nasa_score`), `artifacts.py` (`save_artifacts`, `load_artifacts`), `predict.py` (`predict_from_features`, clamp ≥ 0), `train.py` (checksum gate, GroupKFold CV, final fit, test evaluation, metadata, report), `scripts/train.py --model baseline`.
  - [x] M2-T5 Tests: labels; features (values, row count, columns, isolation, truncation invariance, trailing-window-only invariance, order independence, selection rule); metrics (hand values, NASA asymmetry); training (disjoint folds, end-to-end save/reload/predict, metadata fields, reproducibility, checksum gate, artifact errors); real-data sensor list.
- **Verification commands:**

  ```bash
  ruff check . && ruff format --check .
  pytest -q
  python scripts/train.py --model baseline
  pytest -q --cov=predmaint
  ```

- **Acceptance criteria:** all tests pass (real-data tests executed locally); sensor table recorded in 04 from real data; feature count = 1 + 3 × len(kept sensors); baseline CV and official-test MAE/RMSE/NASA score from an actual run recorded in 05 §7a and 14; `reports/metrics_baseline.json` written.
- **Estimated effort:** 5 h · **Status:** **COMPLETE** (2026-11-08; committed and pushed, CI green; evidence in 14).

## M3 — Primary model, selection, served artifact, inference contract (CR-003)

- **Objective:** a tuned-once primary model, CV-based selection and promotion of the served artifact, and the inference contract the API will call.
- **Dependencies:** M2.
- **Requirements:** FR-007, FR-008 (primary), FR-009, NFR-001, NFR-002.
- **Files to create:** `src/predmaint/models/selection.py`, `reports/metrics_primary.json`, `reports/model_comparison.json`, `reports/model_comparison.md` (generated, committed), `tests/test_selection.py`, `tests/test_inference_contract.py`.
- **Files to modify:** `src/predmaint/config.py` (ADR-025 bands), `src/predmaint/models/registry.py` (`make_primary`, `PRIMARY_SEARCH_GRID`), `train.py` (search, `--select`), `predict.py` (inference contract, shared clamp), `scripts/train.py` (usage), `tests/conftest.py` (session-scoped served-model fixture), `tests/test_train.py`, `tests/test_real_data.py`, `.github/workflows/ci.yml` (v7 actions, `ubuntu-24.04`), `reports/metrics_baseline.json` (re-measured), README, CLAUDE.md, docs 00, 01, 02, 04, 05, 06, 07, 08, 09, 11, 12, 13, 14, 15.
- **Tasks:**
  - [x] M3-T1 Record CR-003 / ADR-024 (CV-based selection), ADR-025 (bands), ADR-026 (bounded search), ADR-027 (clamp to [0, RUL_CAP]).
  - [x] M3-T2 CI housekeeping: `actions/checkout@v7`, `actions/setup-python@v7` (both `node24`), `runs-on: ubuntu-24.04`; YAML parsed and validated locally.
  - [x] M3-T3 `registry.py`: `make_primary` + 12-configuration `PRIMARY_SEARCH_GRID`; `train.py`: `search_primary` (GroupKFold CV RMSE, tie-break MAE), grid/results/winner in metadata and report.
  - [x] M3-T4 `selection.py`: compatibility check, ADR-024 choice, disagreement flag, promotion to `model.joblib` + `model_metadata.json`, `reports/model_comparison.{json,md}`; `scripts/train.py --select`.
  - [x] M3-T5 `predict.py`: `load_model` (cached, `ModelNotAvailableError`), `predict_rul` (typed errors, last-window-only, clamp, ADR-025 status, range warnings), shared `predict_from_features`.
  - [x] M3-T6 Tests: primary train/save/reload/predict + reproducibility; selection cases (primary wins, baseline wins, disagreement, tie-breaks, promotion, incompatible/missing); inference contract; real-data promoted-model RMSE.
- **Verification commands:**

  ```bash
  ruff check . && ruff format --check .
  pytest -q
  python scripts/train.py --model baseline
  time python scripts/train.py --model primary
  python scripts/train.py --select
  pytest -q -m data -v                 # incl. promoted-model RMSE recomputation
  pytest -q --cov=predmaint
  ls -l models/*.joblib
  ```

- **Acceptance criteria:** both candidates have CV and official-test MAE/RMSE/NASA from actual runs (05 §7a, 14); the served model follows ADR-024; a repeat primary run gives identical metrics; primary training < 5 min wall; `model.joblib` + `model_metadata.json` include `model_name`, `selected_by`, `model_version`, `comparison`; all tests pass, including real-data tests locally.
- **Estimated effort:** 5 h · **Status:** **COMPLETE** (2026-11-08; committed and pushed, CI green; evidence in 14).

## M4 — FastAPI service (CR-004)

- **Objective:** serve real predictions per [06](06_API_SPECIFICATION.md) through the M3 inference contract.
- **Dependencies:** M3.
- **Requirements:** FR-010, FR-011, FR-012, FR-013, NFR-003, NFR-010.
- **Files to create:** `src/predmaint/api/__init__.py`, `main.py`, `schemas.py`, `service.py`, `errors.py` (under `src/predmaint/api/`), `scripts/make_request_payload.py`, `examples/predict_request.json`, `examples/predict_request_min.json`, `examples/expected_predictions.json`, `tests/test_api.py`.
- **Files to modify:** `src/predmaint/config.py` (`MODEL_PATH`, `API_HOST`, `API_PORT`), `src/predmaint/models/predict.py` (`load_model` accepts a `.joblib` path, default `$MODEL_PATH`), `tests/test_real_data.py`, `.env.example`, README, docs 01, 02, 04, 05, 06, 07, 08, 09, 10, 11, 13, 14, 15.
- **Tasks:**
  - [x] M4-T1 Schemas per 06 §3–4 (`extra="forbid"`, `allow_inf_nan=False`, ≤ 1000 records, OpenAPI examples).
  - [x] M4-T2 `create_app()` with a lifespan that loads the model once; degraded mode on `ModelNotAvailableError` (ADR-022/028).
  - [x] M4-T3 Routes + one error-handler layer for every code in 06 §6; per-request log line; 500 without traceback.
  - [x] M4-T4 `make_request_payload.py` + committed examples and sidecar (ADR-029).
  - [x] M4-T5 Tests (TestClient via httpx2, synthetic served model): every endpoint, every error code, validation envelope, 404/405, 503, `MODEL_PATH` env, logging, OpenAPI metadata; real-data example test.
  - [x] M4-T6 Housekeeping: CPU-seconds note next to NFR-002 in 05 §7a; CI result for `65db98d` recorded in 14.
- **Verification commands:**

  ```bash
  ruff check . && ruff format --check .
  pytest -q && pytest -q --cov=predmaint
  python -m predmaint.api.main &
  curl -s localhost:8000/health
  curl -s localhost:8000/model/info
  curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d @examples/predict_request.json
  curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d @examples/predict_request_min.json
  # deliberately bad: 10 cycles → 422 INSUFFICIENT_CYCLES; then restart with MODEL_PATH=/nonexistent → /health 200 degraded, /predict 503
  kill %1
  # NFR-003 latency (in-process, 100 sequential calls):
  python - <<'EOF'
  import json, statistics, time
  from fastapi.testclient import TestClient
  from predmaint.api.main import create_app
  payload = json.load(open("examples/predict_request_min.json"))
  with TestClient(create_app()) as client:
      times = []
      for _ in range(100):
          start = time.perf_counter()
          assert client.post("/predict", json=payload).status_code == 200
          times.append(time.perf_counter() - start)
  print("p95_ms", round(statistics.quantiles(times, n=20)[18] * 1000, 1))
  EOF
  ```

- **Acceptance criteria:** all 06 behaviours tested and passing; curl outputs recorded in 14 and copied into 06; p95 < 200 ms recorded; committed examples reproduce their sidecar through the API.
- **Estimated effort:** 3 h · **Status:** **COMPLETE** (2026-11-08; committed and pushed, CI green; evidence in 14).

## M5 — Streamlit dashboard (CR-005)

- **Objective:** a UI that shows API results only.
- **Dependencies:** M4.
- **Requirements:** FR-014, FR-015, FR-016.
- **Files to create:** `src/predmaint/dashboard/__init__.py`, `api_client.py`, `helpers.py`, `app.py` (under `src/predmaint/dashboard/`), `examples/predict_request_late.json`, `tests/test_dashboard_client.py`, `tests/test_dashboard_helpers.py`, `tests/test_dashboard_app.py`.
- **Files to modify:** `examples/expected_predictions.json` (late-life entry), `tests/test_real_data.py`, `.env.example`, README, docs 02, 06, 07, 08, 09, 10, 14, 15.
- **Tasks:**
  - [x] M5-T0 `examples/predict_request_late.json` (test engine 34, true RUL 7) via `scripts/make_request_payload.py`, plus its sidecar entry; shown with engine 1 in README.
  - [x] M5-T1 `api_client.py`: `ApiClient(base_url, timeout, session)` with `health()`, `model_info()`, `predict()` → `ApiResult` (body or envelope unchanged; `API_UNREACHABLE` on network failure; never raises).
  - [x] M5-T2 `helpers.py`: `true_rul`, `EngineHistory`, `engine_payload`, `trajectory_cycles`, `true_rul_series`, `compute_trajectory`, `sensor_chart_frame`.
  - [x] M5-T3 `app.py`: status bar; tabs Engine explorer, RUL trajectory (cached per dataset/unit/step/model), Upload, Model info; graceful API-down and dataset-absent behaviour.
  - [x] M5-T4 Tests: client, helpers, AppTest smoke tests, import boundary, real-data true-RUL vs training labels, late example through the API.
- **Verification commands:**

  ```bash
  ruff check . && ruff format --check .
  pytest -q && pytest -q --cov=predmaint
  python -m predmaint.api.main &
  streamlit run src/predmaint/dashboard/app.py --server.address localhost --server.port 8501 &
  # use the dashboard; confirm POST /predict lines in the API log; then stop both servers
  ```

- **Acceptance criteria:** tests pass; the dashboard produced real `/predict` calls visible in the API log; displayed values recorded in 14; API-down and dataset-absent behaviour verified.
- **Estimated effort:** 3 h · **Status:** **COMPLETE** (2026-11-08; committed and pushed, CI green; evidence in 14).

## M6 — Containerisation, test finalisation, CI docker job

- **Objective:** a fresh clone runs both services via Docker Compose with no training step; the test suite is gated on coverage; CI builds and smoke-tests the image.
- **Dependencies:** M5.
- **Requirements:** FR-017 (and NFR-005 gate, NFR-008).
- **Files to create:** `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `scripts/smoke_test.sh`, `tests/test_docker_smoke.py`, `tests/test_model_artifact.py`; commit `models/model.joblib`.
- **Files to modify:** `.gitignore` (`!models/model.joblib`), `pyproject.toml` (coverage gate in `addopts`, marker docs), `.github/workflows/ci.yml` (`docker` job), `.env.example` (`APP_UID`/`APP_GID`), README (badge, Quickstart Option A/B), CLAUDE.md, docs 01, 02, 03, 07, 08, 09, 10, 11, 12, 13, 14.
- **Tasks:**
  - [x] M6-T1 Dockerfile: single stage, `python:3.13-slim@sha256:e48f59cd…ed90`, wheels only (`--only-binary=:all:`), non-root `app`, model baked in (ADR-032). Digest recorded (R9 closed).
  - [x] M6-T2 Compose: `api` + `dashboard` from one image; `model_loaded` healthcheck; `/_stcore/health` healthcheck; `service_healthy` dependency; `./data` bind mount with first-start download (ADR-031); 127.0.0.1 ports; `unless-stopped`.
  - [x] M6-T3 Smoke test `scripts/smoke_test.sh` (+ `docker` marker wrapper); verified pass (exit 0) and forced failure (exit 1, teardown done).
  - [x] M6-T4 Commit the served artifact (ADR-030) + `test_model_artifact.py` (scikit-learn pin, reproduces the sidecar) — decided: commit.
  - [x] M6-T5 Coverage gate 85 % in pytest config; markers documented.
  - [x] M6-T6 CI `docker` job on `ubuntu-24.04` (amd64 outcome recorded after the user's push); README badge.
- **Verification commands:**

  ```bash
  ruff check . && ruff format --check .
  pytest -q                                     # shows "Required test coverage of 85% reached"
  docker build -t predmaint:latest . && docker images predmaint
  docker compose up -d --build && docker compose ps
  curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d @examples/predict_request.json
  curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d @examples/predict_request_late.json
  docker compose logs dashboard                 # first start: download; restart: "already present"
  docker compose down
  bash scripts/smoke_test.sh; echo "exit=$?"
  ```

- **Acceptance criteria:** image builds; both services healthy; containerised predictions == sidecar; dashboard downloaded FD001 on first start and skipped on restart; smoke test passes (exit 0) and fails non-zero on error; coverage gate passes; outputs recorded in 14.
- **Estimated effort:** 4 h · **Status:** **COMPLETE** (2026-11-08; committed and pushed, CI green; evidence in 14).

## M7 — Finalisation and release readiness

- **Objective:** documentation matches the implementation; the acceptance checklist is closed; portfolio-ready README, evaluation report and notes.
- **Dependencies:** M6 (committed `ccf68a7`, CI green incl. the `docker` job).
- **Requirements:** NFR-005, NFR-009 (and final status of every FR/NFR).
- **Files created:** `docs/16_EVALUATION_REPORT.md`, `docs/17_PORTFOLIO_NOTES.md`, `reports/residuals_by_rul_bucket.json`, `LICENSE`.
- **Files modified:** `README.md` (final), `CLAUDE.md`, docs 01, 02, 06, 07, 08, 10, 11, 12, 14, 15.
- **Tasks:**
  - [x] M7-T1 Close UNVERIFIED items from the CI run 42509206610 (amd64 build, compose v2 on the runner; the Linux bind-mount write path is recorded as not asserted).
  - [x] M7-T2 Fresh clone from GitHub, README Option A and Option B followed literally; one README defect found and fixed (CR-006), step re-run green.
  - [x] M7-T3 Docs-vs-implementation audit of 02, 06, 07, 10 (table in 14); final status for all FR/NFR in 01; no ADR left Proposed.
  - [x] M7-T4 Evaluation report generated from `reports/*.json`; residuals computed and saved.
  - [x] M7-T5 Final README, MIT LICENSE.
  - [x] M7-T6 Portfolio notes.
  - [x] M7-T7 Close-out: CLAUDE.md, 11, 12, 14, 08; final defect review.
- **Verification commands:**

  ```bash
  git clone https://github.com/Rakibul-cyber/industrial-ai-predictive-maintenance.git /tmp/clean && cd /tmp/clean   # then README Option A and B
  ruff check . && ruff format --check .
  pytest -q                                     # coverage gate 85 %
  grep -rnE "TBD|TODO|FIXME" README.md CLAUDE.md docs/ src/ | grep -v "grep -rnE" || echo "clean"
  ```

- **Acceptance criteria:** every item in [11](11_ACCEPTANCE_CRITERIA.md) ticked with evidence or explicitly marked Not met with a reason.
- **Estimated effort:** 2 h · **Status:** **COMPLETE** (2026-11-08; uncommitted, awaiting the user's final commit).

## M8 — Prediction intervals (post-MVP extension, CR-008 / ADR-034)

- **Objective:** a nominal 80 % prediction interval served next to every point prediction, evaluated honestly (coverage and width, overall and by RUL bucket), without changing the point model.
- **Dependencies:** M7 (MVP complete, CR-007 committed).
- **Requirements:** FR-018.
- **Files to create:** `src/predmaint/models/intervals.py`, `models/model_intervals.joblib`, `models/model_intervals_metadata.json`, `reports/metrics_intervals.json`, `tests/test_intervals.py`.
- **Files to modify:** `src/predmaint/__init__.py` (0.2.0), `models/predict.py`, `models/train.py` (CLI), `models/registry.py`, `api/{schemas,service,main}.py`, `dashboard/{helpers,app}.py`, `scripts/make_request_payload.py`, `scripts/smoke_test.sh`, `Dockerfile`, `.gitignore`, `examples/expected_predictions.json`, tests (`conftest`, `test_api`, `test_inference_contract`, `test_dashboard_*`, `test_model_artifact`, `test_real_data`), README, CLAUDE.md, docs 00, 01, 05, 06, 08, 09, 10, 11, 12, 13, 14, 15, 16, 17.
- **Tasks:**
  - [x] M8-T1 Change control: CR-008, ADR-034, FR-018, doc 00 scope, M8 plan.
  - [x] M8-T2 `registry.make_quantile_model`; `models/intervals.py` (`run_interval_training`, `summarize_intervals`); `scripts/train.py --model intervals` (refuses without the served model or with a served baseline).
  - [x] M8-T3 Inference: `IntervalModel`, `order_bounds`, `interval_mismatch`, `load_interval_model`; `predict_rul(..., intervals)`; point prediction byte-identical.
  - [x] M8-T4 API 0.2.0: `rul_lower`/`rul_upper`/`interval_level`, `/health.intervals_loaded`, `/model/info.intervals`; warning + null fields when absent or mismatched.
  - [x] M8-T5 Dashboard: interval caption, shaded trajectory band (Vega-Lite area), model-info intervals table; graceful nulls.
  - [x] M8-T6 Artifacts trained and committed-ready; sidecar regenerated (point values unchanged); Dockerfile copies them; smoke test asserts `intervals_loaded` and `lower ≤ point ≤ upper`.
  - [x] M8-T7 Tests (intervals, API, dashboard, artifact, real data); CR-009 smoke-test port fix.
- **Verification commands:**

  ```bash
  ruff check . && ruff format --check .
  pytest -q
  python scripts/train.py --model intervals
  python -m predmaint.api.main &   # curl /health and /predict (late + engine 1); stop
  bash scripts/smoke_test.sh; echo $?
  ls -l models/*.joblib
  ```

- **Acceptance criteria:** FR-018 met with evidence; point model and its artifact byte-identical; all tests pass with the 85 % gate; smoke test passes with the interval assertions.
- **Estimated effort:** 4 h · **Status:** **COMPLETE** (2026-11-08; uncommitted, awaiting review). Evidence in 14.

## Requirement → milestone traceability

| Milestone | Requirements |
|---|---|
| M1 | FR-001, FR-002, FR-003, NFR-004, NFR-006, NFR-008 |
| M2 | FR-004, FR-005, FR-006, FR-008 (baseline), NFR-007 |
| M3 | FR-007, FR-008 (primary), FR-009, NFR-001, NFR-002 |
| M4 | FR-010, FR-011, FR-012, FR-013, NFR-003, NFR-010 |
| M5 | FR-014, FR-015, FR-016 |
| M6 | FR-017 |
| M7 | NFR-005, NFR-009 |
| M8 | FR-018 |
