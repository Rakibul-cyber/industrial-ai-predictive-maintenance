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
- Warnings policy (also in CLAUDE.md): `DeprecationWarning` is an error (`filterwarnings = ["error::DeprecationWarning"]` in `pyproject.toml`). A warning from project code is fixed at its cause. A warning raised inside a third-party library gets a narrowly scoped `filterwarnings` entry (module + message pattern) with a comment naming the library, the warning, and why it is out of our control. Warnings are never disabled globally or per test without that comment. Status after M2: no filters needed (83 passed, no warnings reported).
- Download tests use a local stdlib HTTP server (no internet); only the `network` test touches NASA S3. CI does **not** download the dataset (ADR-021).

## 4. Required passing tests per milestone (cumulative)

| Milestone | Must pass before "complete" |
|---|---|
| M1 | `test_download` (checksum logic, zip-slip guard, skip-if-valid), `test_load` (26 cols with trailing spaces), `test_validate` (one failing case per rule in 04 §6), `test_real_data` (run locally), `network` test once locally |
| M2 | + `test_labels` (formula, cap, None), `test_features` (values, row count, columns, per-engine isolation, truncation invariance, trailing-window-only invariance, selection rule), `test_metrics` (MAE/RMSE hand values, NASA known values + asymmetry), `test_train` (disjoint GroupKFold units, baseline end-to-end train → save → reload → predict, metadata fields, reproducibility, checksum gate, artifact errors), `test_real_data` sensor list (run locally) |
| M3 | + primary model in `test_train` (search, save/reload/predict, reproducibility), `test_selection` (ADR-024: primary wins, baseline wins, CV/test disagreement flagged, tie-breaks, promotion files, incompatible/missing candidates), `test_inference_contract` (load/caching/missing/corrupt model, schema and bounds, determinism, last-window-only, offline == contract, each typed error, range warnings, ADR-025 band edges, clamp), `test_real_data` promoted-model RMSE recomputation |
| M4 | + `test_api` (synthetic served-model fixture, never `models/`; `/health` 200 with and without model incl. `MODEL_PATH` pointing nowhere; 200 paths; 422 `VALIDATION_ERROR`/`INSUFFICIENT_CYCLES`/`MISSING_SENSORS`/`NON_CONSECUTIVE_CYCLES`; NaN/Infinity rejected; extra field rejected; 503 `MODEL_UNAVAILABLE` on `/health`, `/predict`, `/model/info`; 404/405 envelopes; 500 `INTERNAL_ERROR` via a monkeypatched failing predictor with no trace in the body; API output == `predict_rul`); > 1000 records, empty list, wrong type, malformed JSON → 422 envelope; one log line per request), `test_real_data` (POST both committed examples → matches `examples/expected_predictions.json`) |
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
