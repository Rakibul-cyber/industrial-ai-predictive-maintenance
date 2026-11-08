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
- `docker`: skip unless `RUN_DOCKER_TESTS=1`; runs `scripts/smoke_test.sh` (`RUN_DOCKER_TESTS=1 pytest -m docker --no-cov`).
- All three markers are registered in `pyproject.toml` (`[tool.pytest.ini_options].markers`).
- Warnings policy (also in CLAUDE.md): `DeprecationWarning` is an error (`filterwarnings = ["error::DeprecationWarning"]` in `pyproject.toml`). A warning from project code is fixed at its cause. A warning raised inside a third-party library gets a narrowly scoped `filterwarnings` entry (module + message pattern) with a comment naming the library, the warning, and why it is out of our control. Warnings are never disabled globally or per test without that comment. Status after M2: no filters needed (83 passed, no warnings reported).
- Download tests use a local stdlib HTTP server (no internet); only the `network` test touches NASA S3. CI does **not** download the dataset (ADR-021).

## 4. Required passing tests per milestone (cumulative)

| Milestone | Must pass before "complete" |
|---|---|
| M1 | `test_download` (checksum logic, zip-slip guard, skip-if-valid), `test_load` (26 cols with trailing spaces), `test_validate` (one failing case per rule in 04 §6), `test_real_data` (run locally), `network` test once locally |
| M2 | + `test_labels` (formula, cap, None), `test_features` (values, row count, columns, per-engine isolation, truncation invariance, trailing-window-only invariance, selection rule), `test_metrics` (MAE/RMSE hand values, NASA known values + asymmetry), `test_train` (disjoint GroupKFold units, baseline end-to-end train → save → reload → predict, metadata fields, reproducibility, checksum gate, artifact errors), `test_real_data` sensor list (run locally) |
| M3 | + primary model in `test_train` (search, save/reload/predict, reproducibility), `test_selection` (ADR-024: primary wins, baseline wins, CV/test disagreement flagged, tie-breaks, promotion files, incompatible/missing candidates), `test_inference_contract` (load/caching/missing/corrupt model, schema and bounds, determinism, last-window-only, offline == contract, each typed error, range warnings, ADR-025 band edges, clamp), `test_real_data` promoted-model RMSE recomputation |
| M4 | + `test_api` (synthetic served-model fixture, never `models/`; `/health` 200 with and without model incl. `MODEL_PATH` pointing nowhere; 200 paths; 422 `VALIDATION_ERROR`/`INSUFFICIENT_CYCLES`/`MISSING_SENSORS`/`NON_CONSECUTIVE_CYCLES`; NaN/Infinity rejected; extra field rejected; 503 `MODEL_UNAVAILABLE` on `/health`, `/predict`, `/model/info`; 404/405 envelopes; 500 `INTERNAL_ERROR` via a monkeypatched failing predictor with no trace in the body; API output == `predict_rul`); > 1000 records, empty list, wrong type, malformed JSON → 422 envelope; one log line per request), `test_real_data` (POST both committed examples → matches `examples/expected_predictions.json`) |
| M5 | + `test_dashboard_client` (success passthrough, every error envelope intact, timeout/connection/generic network error → clean `API_UNREACHABLE` result, non-JSON responses, import boundary), `test_dashboard_helpers` (true RUL for train/test vs hand values, payloads, trajectory sampling and computation, error propagation, chart data), `test_dashboard_app` (Streamlit `AppTest` with mocked client: status + prediction + slider + trajectory; API down; dataset absent), `test_real_data` (dashboard capped true-RUL series == `add_rul` labels for train units 1/39/69; late-life example through the API) |
| M6 | + `test_model_artifact` (committed model present, installed scikit-learn == metadata version, loads, reproduces the sidecar predictions — runs in CI without data), `scripts/smoke_test.sh` / `test_docker_smoke` (build, compose up, both healthchecks, `/health` model_loaded, late example → `critical` == sidecar, dashboard `/_stcore/health`, teardown, non-zero exit on failure), 85 % coverage gate |
| M7 | Full suite incl. `data`, `docker` locally; coverage threshold met |
| M8 | + `test_intervals` (artifact for the served model, point model untouched + deterministic retrain, refusals without served model / with served baseline, ordering/clamping invariants on random inputs incl. reorders, coverage summary hand values, inference with/without/mismatched intervals with byte-identical point, last-window-only), interval cases in `test_api` (health/info additions, absent and mismatched artifact → null fields + warning), `test_dashboard_helpers`/`test_dashboard_app` (band data, null handling, AppTest with interval fields), `test_model_artifact` (committed intervals match the served model; sidecar bounds), `test_real_data` (committed artifact reproduces the recorded test coverage; examples incl. bounds); smoke test asserts `intervals_loaded` and `lower ≤ point ≤ upper` |
| M9 | + `test_anomaly` (synthetic drift engine: score rises after onset, threshold exceeded, first detection after onset; OOF threshold > in-sample threshold, also recorded in the fixture's metadata; IsolationForest branch; `first_detection`, per-engine trend, detection timing, bucket flags and constant-input Spearman hand cases; deterministic training that leaves the point/interval files untouched; fallback tried at most once; refusals without served model / healthy rows / with other data; series length = `cycles_received − 30 + 1`, last-window-only, mismatch → unavailable, validation errors identical to `predict_rul`, loader errors), `/anomaly` cases in `test_api` (body == `score_anomaly`, series, identical 422 bodies to `/predict`, bad query param, 503 `ANOMALY_UNAVAILABLE` without model, health/info fields, absent/mismatched detector → warning + 503 while `/predict` is unchanged), `test_dashboard_client` (`anomaly` paths, 503 envelope passthrough), `test_dashboard_helpers` (badge, series frame, nulls), `test_dashboard_app` (AppTest with and without anomaly fields; one series call), `test_model_artifact` (committed detector matches the served model; sidecar anomaly block), `test_real_data` (retraining reproduces the committed threshold and evaluation; examples through `/anomaly`); smoke test asserts `anomaly_loaded`, late example anomalous, engine 1 < engine 34 |
| M10 | + calibration cases in `test_intervals` (nonconformity and conformal-quantile hand values incl. the exact ⌈(n+1)(1−α)⌉ index and the k > n → ∞ case; deterministic, disjoint, complete engine split; Q ≥ 0 never narrows and keeps clamp/order invariants on random inputs, `None` = v1.1, negative Q narrows as standard CQR; synthetic i.i.d. data where CQR lands within 0.03 of 0.80 from ~0.31; fixture metadata records Q, α, before == v1.1; metadata without Q → v1.1 bounds and `interval_calibrated: false`), `test_api` (`/model/info` calibration fields; uncalibrated metadata → `interval_calibrated: false`; exact `/predict` body gains the field), `test_dashboard_helpers`/`test_dashboard_app` (band caption names Q), `test_model_artifact` (sidecar calibrated bounds), `test_real_data` (committed Q reproduces the recorded calibrated test coverage; re-running the interval step reproduces every calibration number and identical quantile-model predictions); smoke test asserts `interval_calibrated: true` |

A milestone is not complete if any previously passing test fails.

## 5. CI definition (`.github/workflows/ci.yml`, created in M1)

- Triggers: `push` and `pull_request` on `main`.
- Job `lint-test` on `ubuntu-24.04`: `actions/checkout@v7` → `actions/setup-python@v7` (`python-version: "3.13"`, pip cache) → `pip install -r requirements-dev.txt && pip install -e .` → `ruff check .` → `ruff format --check .` → `pytest -v` (coverage gate from `pyproject.toml`). No dataset: `data` tests skip; `network`/`docker` are gated off.
- Job `docker` on `ubuntu-24.04` (amd64), `needs: lint-test`: `actions/checkout@v7` → `bash scripts/smoke_test.sh`. It builds the image, starts compose (the dashboard downloads FD001 into the bind mount), waits for both healthchecks, checks `/health` and the late-life `/predict` against the sidecar, checks `interval_calibrated` (M10) and `/anomaly` for both examples (M9), checks the dashboard health and tears down; a non-zero exit fails the job.
- Real-data tests are recorded from local runs in [14](14_PROJECT_STATUS.md).

## 6. Coverage expectation

- **Gate: 85 % line coverage** of `src/predmaint` (stricter than NFR-005's 80 % floor), with `src/predmaint/dashboard/app.py` omitted (Streamlit UI, covered by AppTest smoke tests). Enforced since M6 by `addopts = "... --cov=predmaint --cov-fail-under=85"` in `pyproject.toml`, so plain `pytest` and CI fail below it. Use `--no-cov` when running a subset of tests. Measured at M6: 94.98 % locally, and 95 % under CI conditions (no dataset).
- Coverage is a floor, not a goal; the leakage and contract tests above are mandatory regardless of percentage.
