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
  - M1-T1 Write `pyproject.toml` (src layout, `requires-python = ">=3.13,<3.14"`, ruff config, pytest markers `data`/`network`/`docker`) and the two requirements files with the exact pins from [03](03_TECH_STACK.md).
  - M1-T2 `config.py`: `DATA_DIR`, `MODEL_DIR`, `RUL_CAP=125`, `WINDOW_SIZE=30`, `RANDOM_SEED=42`, `CV_FOLDS=5`, band thresholds; env overrides for paths, `LOG_LEVEL`, `API_BASE_URL`, `API_TIMEOUT_SECONDS`.
  - M1-T3 `download.py`: URL + SHA-256 constants from 04 §1/§3; skip download if valid files exist; nested extraction restricted to the 14 expected names.
  - M1-T4 `load.py`: `sep=r"\s+"`, `header=None`, names per 04 §4; int casts for `unit`, `cycle`.
  - M1-T5 `validate.py`: checks 1–6 of 04 §6, raising `DataValidationError` with a specific message.
  - M1-T6 Tests: synthetic fixtures incl. trailing double spaces; every validation rule has a failing-case test; real-data shape test (marker `data`); live download test (marker `network`).
  - M1-T7 CI workflow (ubuntu-latest, `actions/setup-python` with `python-version: "3.13"`, install dev deps, `ruff check .`, `ruff format --check .`, `pytest`). **UNVERIFIED item:** runner provides Python 3.13 — confirmed on the first CI run, which requires a user-authorized push.
  - M1-T8 **UNVERIFIED item — dataset licence:** read the NASA PCoE repository page and data.nasa.gov entry for usage/citation terms; record exact wording + URL in 04 §11; if no terms are found, record that and keep the no-redistribution policy.
  - M1-T9 **Re-verify pins in project context:** fresh `.venv` from `requirements-dev.txt`, `pip check`, import smoke.
- **Verification commands:**

  ```bash
  python scripts/download_data.py && python scripts/download_data.py   # 2nd run: no download
  shasum -a 256 data/raw/CMAPSSData/{train,test,RUL}_FD001.txt
  pip check
  ruff check . && ruff format --check .
  pytest -v
  RUN_NETWORK_TESTS=1 pytest -v -m network
  ```

- **Acceptance criteria:** checksums match 04 §3; all tests pass (real-data test executed, not skipped, locally); ruff clean; `pip check` clean; licence outcome recorded; CI workflow file present (CI run result recorded in 14 once a push is authorized).
- **Estimated effort:** 4 h · **Status:** In progress.

## M2 — Labels and feature engineering

- **Objective:** leakage-free labels and features via one reusable code path.
- **Dependencies:** M1.
- **Requirements:** FR-004, FR-005.
- **Files to create:** `src/predmaint/data/labels.py`, `src/predmaint/features/__init__.py`, `src/predmaint/features/build.py`, `tests/test_labels.py`, `tests/test_features.py`.
- **Files to modify:** `docs/04_DATASET_SPECIFICATION.md` (sensor table), `docs/13_DECISION_LOG.md` (ADR-005 confirmation), `docs/14_PROJECT_STATUS.md`.
- **Tasks:**
  - M2-T1 **UNVERIFIED item — near-constant sensors:** compute std and unique count of `s_1..s_21` on `train_FD001` with a one-off command; record the table and resulting kept/dropped lists in 04; confirm or amend ADR-005 via 15.
  - M2-T2 `labels.py`: `add_rul(df, rul_cap)`.
  - M2-T3 `build.py`: `select_sensors(train_df) -> list[str]`, `build_features(df, sensors, window_size) -> DataFrame` (adds `cycle` + raw/roll_mean/roll_std, drops rows with `cycle < window_size`), `last_rows(features_df) -> DataFrame`, `feature_names(sensors) -> list[str]`.
  - M2-T4 Tests: RUL formula/clip/None; rolling values vs hand-computed; causality (mutate future rows → past features unchanged); per-unit isolation (no window crosses units); eligibility filter; feature order; selection rule on synthetic constant/near-constant columns.
- **Verification commands:**

  ```bash
  pytest -v tests/test_labels.py tests/test_features.py
  pytest -v && ruff check . && ruff format --check .
  ```

- **Acceptance criteria:** all tests pass; sensor table recorded in 04 from real data; feature count = 1 + 3 × len(kept sensors).
- **Estimated effort:** 3 h · **Status:** Not started.

## M3 — Training, evaluation, artifacts, inference contract

- **Objective:** real CV and test metrics for both models; selected artifact + metadata; shared inference function.
- **Dependencies:** M2.
- **Requirements:** FR-006, FR-007, FR-008, FR-009, NFR-001, NFR-002, NFR-007.
- **Files to create:** `src/predmaint/models/__init__.py`, `registry.py`, `metrics.py`, `train.py`, `artifacts.py`, `predict.py` (all under `src/predmaint/models/`), `scripts/train.py`, `tests/test_metrics.py`, `tests/test_train.py`, `tests/test_inference_contract.py`.
- **Files to modify:** `tests/conftest.py` (tiny-model fixture), `docs/14_PROJECT_STATUS.md` (actual metrics, runtime), `docs/05_ML_DESIGN.md` only if a documented detail proves wrong (via 15).
- **Tasks:**
  - M3-T1 `registry.py`: `make_baseline()`, `make_primary(seed)` per 05 §2–3.
  - M3-T2 `metrics.py`: `mae`, `rmse`, `nasa_score` (05 §7, ADR-009).
  - M3-T3 `train.py`: checksum gate, CV (GroupKFold 5 by unit), final fit, test evaluation at last cycle, ADR-010 selection, metadata assembly.
  - M3-T4 `artifacts.py`: `save_artifacts`, `load_artifacts` with schema-version check.
  - M3-T5 `predict.py`: inference contract (05 §10) incl. typed errors, clamp, bands, range warnings.
  - M3-T6 `scripts/train.py`: thin CLI (`--model-dir`, `--data-dir`), prints metrics table.
  - M3-T7 Tests: metric known values (incl. NASA `d=10`, `d=-13`); GroupKFold folds have disjoint units; training smoke on synthetic data writes artifact with all metadata keys; contract: loaded model + required columns → float ≥ 0; each typed error raised; offline test-set prediction for a unit equals `predict_rul` on that unit's history.
- **Verification commands:**

  ```bash
  time python scripts/train.py
  python -c "import json;m=json.load(open('models/model_metadata.json'));print(m['selected'],m['metrics']['baseline']['test'],m['metrics']['primary']['test'])"
  cp models/model_metadata.json /tmp/meta1.json && python scripts/train.py && python -c "import json;a=json.load(open('/tmp/meta1.json'))['metrics'];b=json.load(open('models/model_metadata.json'))['metrics'];print('identical' if a==b else 'DIFFERENT')"
  pytest -v --cov=predmaint --cov-report=term-missing
  ```

- **Acceptance criteria:** both models have CV and test MAE/RMSE/NASA from an actual run, recorded in 14; second run prints `identical`; training time < 5 min recorded; artifact + metadata present with all 05 §9 fields; all tests pass.
- **Estimated effort:** 4 h · **Status:** Not started.

## M4 — FastAPI service

- **Objective:** serve real predictions per [06](06_API_SPECIFICATION.md).
- **Dependencies:** M3.
- **Requirements:** FR-010, FR-011, FR-012, FR-013, NFR-003, NFR-010.
- **Files to create:** `src/predmaint/api/__init__.py`, `main.py`, `schemas.py`, `service.py`, `errors.py` (under `src/predmaint/api/`), `scripts/make_request_payload.py`, `tests/test_api.py`.
- **Files to modify:** `docs/14_PROJECT_STATUS.md`.
- **Tasks:**
  - M4-T1 Schemas per 06 §3–4 (`extra="forbid"`, `allow_inf_nan=False`).
  - M4-T2 Lifespan loading from `MODEL_DIR`; degraded mode on failure (02 §8).
  - M4-T3 Routes + error envelope handlers for every code in 06 §6.
  - M4-T4 `make_request_payload.py`.
  - M4-T5 Tests (TestClient via httpx2): each endpoint happy path; each error code; NaN/Infinity rejected; extra field rejected; 503 when `MODEL_DIR` empty; API prediction == `predict_rul` for the same input.
- **Verification commands:**

  ```bash
  pytest -v tests/test_api.py
  uvicorn predmaint.api.main:app --port 8000 &   # then:
  curl -s localhost:8000/health
  curl -s localhost:8000/model/info
  python scripts/make_request_payload.py --unit 1 --last 30 > /tmp/payload.json
  curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d @/tmp/payload.json
  kill %1
  # NFR-003 latency (in-process, 100 sequential calls):
  python - <<'EOF'
  import json, statistics, time
  from fastapi.testclient import TestClient
  from predmaint.api.main import app
  payload = json.load(open("/tmp/payload.json"))
  with TestClient(app) as client:
      times = []
      for _ in range(100):
          start = time.perf_counter()
          assert client.post("/predict", json=payload).status_code == 200
          times.append(time.perf_counter() - start)
  print("p95_ms", round(statistics.quantiles(times, n=20)[18] * 1000, 1))
  EOF
  ```

- **Acceptance criteria:** all 06 behaviours tested and passing; curl outputs recorded in 14; p95 < 200 ms recorded.
- **Estimated effort:** 3 h · **Status:** Not started.

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
  uvicorn predmaint.api.main:app --port 8000 &  streamlit run src/predmaint/dashboard/app.py
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
| M2 | FR-004, FR-005 |
| M3 | FR-006, FR-007, FR-008, FR-009, NFR-001, NFR-002, NFR-007 |
| M4 | FR-010, FR-011, FR-012, FR-013, NFR-003, NFR-010 |
| M5 | FR-014, FR-015, FR-016 |
| M6 | FR-017 |
| M7 | NFR-005, NFR-009 |
