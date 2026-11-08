# 02 — System Architecture

Style: **modular monolith** — one Python package `predmaint` (under `src/`), CPU-only, two runtime processes (API, dashboard) built from one Docker image. Rationale: ADR-015, ADR-019.

## 1. Components and responsibilities

| Component | Module | Responsibility | Does NOT |
|---|---|---|---|
| Config | `predmaint/config.py` | Single source of defaults (`rul_cap`, `window_size`, seeds, band thresholds, paths) and env-var overrides. | Hold secrets. |
| Data | `predmaint/data/` | Download + checksum (`download.py`), parse raw files (`load.py`), validate data contract (`validate.py`), compute labels (`labels.py`). | Engineer features. |
| Features | `predmaint/features/build.py` | Sensor selection, causal rolling features, "last row per unit" extraction. **Single code path used by training, evaluation and the API.** | Fit models. |
| Models | `predmaint/models/` | Model definitions (`registry.py`), CV + training (`train.py`), metrics (`metrics.py`), artifact save/load + metadata (`artifacts.py`), inference function (`predict.py`). | Know about HTTP. |
| API | `predmaint/api/` | FastAPI app (`main.py`), Pydantic schemas (`schemas.py`), model-holder service (`service.py`), error envelope (`errors.py`). | Train models or read raw data files. |
| Dashboard | `predmaint/dashboard/` | Streamlit UI (`app.py`), a thin HTTP client returning typed results (`api_client.py`), and display helpers (`helpers.py`: true RUL, request bodies, trajectory sampling). | Import model, feature, inference or API-server code, or compute predictions. |
| CLI scripts | `scripts/` | Thin entrypoints (`download_data.py`, `train.py`, `make_request_payload.py`) that call package functions. | Contain logic. |

## 2. Diagram

```mermaid
flowchart LR
  subgraph Offline["Offline (developer machine)"]
    NASA[(NASA PCoE S3<br/>CMAPSS zip)] -->|download_data.py<br/>SHA-256 check| RAW[data/raw/CMAPSSData]
    RAW --> LOAD[data.load + data.validate]
    LOAD --> LAB[data.labels<br/>RUL, clip 125]
    LAB --> FEAT[features.build<br/>sensor select + rolling W=30]
    FEAT --> CV[models.train<br/>GroupKFold(5) by unit]
    CV --> FIT[fit baseline + primary on full train]
    FIT --> EVAL[evaluate on test last cycle<br/>vs RUL_FD001]
    EVAL --> SEL{ADR-024<br/>selection by CV RMSE}
    SEL --> ART[(models/model.joblib<br/>models/model_metadata.json)]
  end
  subgraph Runtime["Runtime (docker compose)"]
    ART -->|read-only volume| API[api: FastAPI :8000]
    API -->|features.build + models.predict| API
    DASH[dashboard: Streamlit :8501] -->|HTTP JSON<br/>API_BASE_URL| API
    USER((User)) --> DASH
    CLIENT((curl / client)) --> API
  end
```

## 3. Data flow

1. `download_data.py` → `data/raw/cmapss_outer.zip` → nested `CMAPSSData.zip` → `data/raw/CMAPSSData/*.txt` (only expected member names are extracted; checksums verified).
2. `load.py` parses whitespace-delimited, header-less files into named DataFrames; `validate.py` enforces the contract in [04](04_DATASET_SPECIFICATION.md) §6.
3. No processed data is persisted for the MVP (FD001 parses in well under a second at this size); `data/processed/` is reserved and gitignored.

## 4. Training flow (`scripts/train.py`)

1. Verify checksums (NFR-007) → load → validate.
2. Label train rows (FR-004); determine kept sensors from **train only** (FR-005).
3. Build features; keep training rows with `cycle ≥ window_size` (ADR-005).
4. For each candidate (baseline, primary): 5-fold GroupKFold CV by `unit` → fold metrics.
5. Fit each candidate on all training rows; predict the last row of each test unit; compute test MAE/RMSE/NASA score.
6. `scripts/train.py --select`: apply ADR-024 (CV RMSE); copy the winner to `model.joblib`, write `model_metadata.json` with the comparison of both candidates, and `reports/model_comparison.{json,md}`.

## 5. Inference flow (`POST /predict`)

1. Pydantic validates the request shape and types (422 on failure).
2. Service checks business rules against loaded metadata: required sensors present, ≥ `window_size` records, consecutive cycles (422 with specific codes).
3. Records → DataFrame → `features.build` (same function as training) → last row.
4. `models.predict.predict_rul` → clamp to [0, RUL_CAP] (ADR-027) → health status (ADR-025) → out-of-range warnings.
5. Response with `model_version` from metadata.

The exact contract is in [06](06_API_SPECIFICATION.md); the function-level contract in [05](05_ML_DESIGN.md) §10.

## 6. API ↔ dashboard interaction

Confirmed in M5 (the API server log shows the dashboard's real calls; 14 §M5):

- The dashboard reads `API_BASE_URL` (default `http://localhost:8000`; `http://api:8000` in compose), `API_TIMEOUT_SECONDS` (default 10) and `DATA_DIR`. All HTTP goes through `dashboard/api_client.py` (`requests`). It never raises: each call returns an `ApiResult` with the success body or the docs/06 error envelope unchanged; network failures become a client-side `API_UNREACHABLE` envelope.
- Every script run calls `GET /health` (status bar) and, if a model is loaded, `GET /model/info` (kept sensors for the chart, health bands, model-info table).
- **Engine explorer:** the selected engine's first N cycles (raw FD001 under `DATA_DIR`, read with `predmaint.data.load`) become a docs/06 request body; `POST /predict` returns the predicted RUL and status shown in the UI. True RUL is computed locally from the dataset (`helpers.true_rul`).
- **RUL trajectory:** `POST /predict` every 5 cycles (`TRAJECTORY_STEP`) from cycle 30 to the engine's last cycle, cached with `st.cache_data` keyed by (dataset, unit, step, model_version). A failed call raises, so partial results are never cached.
- **Upload:** a docs/06 JSON body from the user is posted as-is; the response or error envelope is shown verbatim. This works without the raw dataset.
- Import boundary: dashboard modules may import only `predmaint.config`, `predmaint.data.load` and `predmaint.dashboard.*` — enforced by `tests/test_dashboard_client.py`.

## 7. Artifact handling

- Produced only by `scripts/train.py` into `MODEL_DIR` (default `models/`).
- `model.joblib`: a fitted sklearn `Pipeline` (scaler where applicable + estimator). Feature construction parameters (`window_size`, kept sensors, feature order) live in `model_metadata.json` and are applied by `features.build`.
- The served `model.joblib` + `model_metadata.json` are committed (ADR-030) and copied into the image (ADR-032); candidate artifacts stay local. Retraining = train + select + commit both files + rebuild (10 §6).
- Loaded once at API startup (FastAPI lifespan). Metadata `library_versions.scikit-learn` is compared with the runtime version and exposed as `sklearn_version_match` in `/model/info`; `tests/test_model_artifact.py` fails on a mismatch.

## 8. Failure handling

| Failure | Behaviour |
|---|---|
| Artifact or metadata missing/corrupt at startup | API still starts; logs a WARNING; `/health` → 200 with `model_loaded: false` (ADR-028); `/predict`, `/model/info` → 503 `MODEL_UNAVAILABLE`. Compose healthcheck stays unhealthy. |
| Invalid request body | 422 `VALIDATION_ERROR` with field-level details. |
| Business-rule violation | 422 with `INSUFFICIENT_CYCLES`, `MISSING_SENSORS`, or `NON_CONSECUTIVE_CYCLES`. |
| Values outside training range | 200 with entries in `warnings` (not rejected). |
| Unexpected exception | 500 `INTERNAL_ERROR`; stack trace logged server-side only. |
| API down / timeout (dashboard) | Dashboard shows "API unreachable at `<API_BASE_URL>`" and disables prediction; no crash. |
| Dataset download fails / checksum mismatch | `download_data.py` exits non-zero; training refuses to start. |
