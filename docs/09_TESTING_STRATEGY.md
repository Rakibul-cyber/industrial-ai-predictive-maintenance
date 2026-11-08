# 09 — Testing Strategy

Framework: pytest 9.2.0 + pytest-cov 7.1.0; API tests via `fastapi.testclient.TestClient` backed by **httpx2** (ADR-013). Test file names per milestone are fixed in [07](07_REPOSITORY_STRUCTURE.md).

## 1. Test types

| Type | Purpose | Data used |
|---|---|---|
| Unit | Pure functions: parsing, validation rules, labels, features, metrics, health bands | Synthetic fixtures (`conftest.py`) |
| Data validation | Data contract on real FD001 (shapes, nulls, monotonic cycles, RUL count) | Real files (marker `data`) |
| Leakage | GroupKFold unit disjointness; feature causality; per-unit window isolation | Synthetic |
| Training smoke | End-to-end train on tiny synthetic data → artifact + metadata keys | Synthetic |
| Inference contract | `load_model` + `predict_rul` behaviour and typed errors; offline == API output | Tiny trained model fixture |
| API | Every endpoint, status code, and error code in [06](06_API_SPECIFICATION.md) | Tiny trained model fixture written to `tmp_path`, `MODEL_DIR` monkeypatched |
| Dashboard client | Payload building, error parsing, API-down handling; import-boundary check | Fake `requests` session |
| Docker smoke | Compose up → `/health` → `/predict` → down | Real artifact (marker `docker`) |
| Network | Live download + checksum | NASA S3 (marker `network`) |

## 2. Synthetic fixtures

`conftest.py` generates C-MAPSS-shaped frames (26 columns, 3–5 units, 40–80 cycles each, two constant sensors, one 2-valued sensor, trending sensors correlated with RUL) and writes them in the raw format **including the two trailing spaces** observed in the real files. No NASA data is copied into the repo (04 §11).

## 3. Gating and markers

- `data`: auto-skip if `DATA_DIR/raw/CMAPSSData/train_FD001.txt` is absent. Must **run (not skip)** locally for M1, M3, M7 sign-off.
- `network`: skip unless `RUN_NETWORK_TESTS=1`.
- `docker`: skip unless `RUN_DOCKER_TESTS=1`.
- Warnings: `filterwarnings = ["error::DeprecationWarning"]` (all modules) in `pyproject.toml`. The M1 suite is clean under it (48 passed, no warnings). If a future third-party deprecation forces a narrower filter, record why via 15.
- Download tests use a local stdlib HTTP server (no internet); only the `network` test touches NASA S3. CI does **not** download the dataset (ADR-021).

## 4. Required passing tests per milestone (cumulative)

| Milestone | Must pass before "complete" |
|---|---|
| M1 | `test_download` (checksum logic, zip-slip guard, skip-if-valid), `test_load` (26 cols with trailing spaces), `test_validate` (one failing case per rule in 04 §6), `test_real_data` (run locally), `network` test once locally |
| M2 | + `test_labels`, `test_features` (values, causality, isolation, eligibility, order, selection rule) |
| M3 | + `test_metrics` (incl. NASA known values), `test_train` (disjoint folds, smoke artifact, metadata keys), `test_inference_contract` (typed errors, clamp, bands, offline == contract) |
| M4 | + `test_api` (200 paths; 422 `VALIDATION_ERROR`/`INSUFFICIENT_CYCLES`/`MISSING_SENSORS`/`NON_CONSECUTIVE_CYCLES`; NaN/Infinity rejected; extra field rejected; 503 `MODEL_UNAVAILABLE` on `/health`, `/predict`, `/model/info`; 404/405 envelopes; 500 `INTERNAL_ERROR` via a monkeypatched failing predictor with no trace in the body; API output == `predict_rul`) |
| M5 | + `test_dashboard_client` (payload, error parsing, `ApiUnavailableError`, no `predmaint.models` import in `dashboard/`) |
| M6 | + `test_docker_smoke` with `RUN_DOCKER_TESTS=1` |
| M7 | Full suite incl. `data`, `docker` locally; coverage threshold met |

A milestone is not complete if any previously passing test fails.

## 5. CI definition (`.github/workflows/ci.yml`, created in M1)

- Triggers: `push` and `pull_request` on `main`.
- Job `lint-test` on `ubuntu-latest`: checkout → `actions/setup-python` (`python-version: "3.13"`, pip cache) → `pip install -r requirements-dev.txt && pip install -e .` → `ruff check .` → `ruff format --check .` → `pytest -v --cov=predmaint --cov-report=term-missing`.
- CI has no dataset and no Docker smoke: `data` tests skip, `network`/`docker` are gated off. Real-data and Docker results are recorded from local runs in [14](14_PROJECT_STATUS.md).
- The first run happens only after the user authorizes a push (CLAUDE.md Git rules).

## 6. Coverage expectation

- **≥ 80 % line coverage** of `src/predmaint`, with `src/predmaint/dashboard/app.py` omitted (Streamlit UI is verified manually in M5/M6). Enforced at M7; reported from M3 onwards.
- Coverage is a floor, not a goal; the leakage and contract tests above are mandatory regardless of percentage.
