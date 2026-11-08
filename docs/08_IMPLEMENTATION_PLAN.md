# 08 — Implementation Plan

Milestone 0 (planning, this document set) is complete and approved (architecture frozen 2026-11-08). Milestones run strictly in order; each starts only after explicit user authorization (CLAUDE.md).

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
- **Estimated effort:** 4 h · **Status:** Complete 2026-11-08 (awaiting user review; uncommitted). Evidence in 14. CI run itself pending an authorized push (U5).

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
- **Estimated effort:** 5 h · **Status:** Complete 2026-11-08 (uncommitted, awaiting review). Evidence in 14.

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
- **Estimated effort:** 5 h · **Status:** Complete 2026-11-08 (uncommitted, awaiting review). Evidence in 14.

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
- **Estimated effort:** 3 h · **Status:** Complete 2026-11-08 (uncommitted, awaiting review). Evidence in 14.

## M5 — Streamlit dashboard

- **Objective:** UI that displays API results only.
- **Dependencies:** M4.
- **Requirements:** FR-014, FR-015, FR-016.
- **Files to create:** `src/predmaint/dashboard/__init__.py`, `src/predmaint/dashboard/client.py`, `src/predmaint/dashboard/app.py`, `tests/test_dashboard_client.py`.
- **Files to modify:** `docs/14_PROJECT_STATUS.md`.
- **Tasks:**
  - M5-T1 `client.py`: `ApiClient(base_url, timeout, session=None)` with `health()`, `model_info()`, `predict(unit_id, records)`; raises `ApiUnavailableError` / `ApiError(code, message)`.
  - M5-T2 `app.py`: status indicator, model card, unit selector (FD001 test file under `DATA_DIR` if present) or upload (raw FD format), prediction panel, sensor trend charts (`st.line_chart`).
  - M5-T3 Tests: client payload shape; error-envelope parsing; connection error → `ApiUnavailableError` (fake session via monkeypatch); static test that `dashboard/` does not import `predmaint.models`.
- **Verification commands:**

  ```bash
  pytest -v tests/test_dashboard_client.py
  python -m predmaint.api.main &  streamlit run src/predmaint/dashboard/app.py
  # manual: select unit 1 → prediction equals curl result for the same 30 cycles; stop API → error banner, no crash
  ```

- **Acceptance criteria:** tests pass; manual check results (incl. API-down behaviour) recorded in 14.
- **Estimated effort:** 3 h · **Status:** Not started.

## M6 — Containerisation

- **Objective:** both services run via Docker Compose with the mounted artifact.
- **Dependencies:** M5.
- **Requirements:** FR-017.
- **Files to create:** `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `tests/test_docker_smoke.py`.
- **Files to modify:** `docs/10_DEPLOYMENT_PLAN.md` (pinned image digest), `docs/14_PROJECT_STATUS.md`.
- **Tasks:**
  - M6-T1 Dockerfile per [10](10_DEPLOYMENT_PLAN.md) §4; pin `python:3.13-slim@sha256:<digest>` (record digest; R9).
  - M6-T2 Compose per 10 §4 (api healthcheck, read-only model mount, dashboard `API_BASE_URL=http://api:8000`, `DATA_DIR` mount read-only).
  - M6-T3 Smoke test (marker `docker`): `compose up -d --build`, poll `/health` ≤ 60 s, POST a payload, assert 200 + `predicted_rul ≥ 0`, `compose down`.
  - M6-T4 Decide whether the FD001 `model.joblib` is small enough to commit (ADR-022): measure its size with `ls -l models/`, record the decision and rationale in 13 and 10 §5; if committed, adjust `.gitignore` and the demo path (11 AC-22) accordingly.
- **Verification commands:**

  ```bash
  docker compose build
  docker compose up -d && docker compose ps
  curl -s localhost:8000/health && curl -sI localhost:8501 | head -1
  RUN_DOCKER_TESTS=1 pytest -v -m docker
  docker compose down
  ```

- **Acceptance criteria:** image builds; `api` healthy; dashboard returns HTTP 200; smoke test passes; outputs recorded in 14.
- **Estimated effort:** 3 h · **Status:** Not started.

## M7 — Finalisation and release readiness

- **Objective:** documentation matches code; full acceptance checklist passes.
- **Dependencies:** M6.
- **Requirements:** NFR-005, NFR-009 (and re-verification of all FR/NFR).
- **Files to modify:** `README.md` (final), `docs/*` as needed for fidelity, `models/model_metadata.json` (final training run, ready to commit), `docs/11_ACCEPTANCE_CRITERIA.md` (ticked with evidence), `docs/14_PROJECT_STATUS.md`.
- **Tasks:** clean-clone run of every README/08/10 command; coverage measurement; fill README results table from metadata; tick 11 with evidence; final consistency pass (same checks as M0 Step 4).
- **Verification commands:**

  ```bash
  git clone <repo> /tmp/clean && cd /tmp/clean   # then the full sequence in 10 §2
  pytest -v --cov=predmaint --cov-report=term-missing   # ≥ 80 % excl. dashboard/app.py
  ruff check . && ruff format --check .
  grep -rnE "TBD|TODO|FIXME" README.md CLAUDE.md docs/ src/ | grep -v "grep -rnE" || echo "clean"
  ```

- **Acceptance criteria:** every item in [11](11_ACCEPTANCE_CRITERIA.md) checked with evidence.
- **Estimated effort:** 2 h · **Status:** Not started.

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
