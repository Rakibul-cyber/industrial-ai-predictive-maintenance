# 02 — System Architecture

Style: **modular monolith** — one Python package `predmaint` (under `src/`), CPU-only, two runtime processes (API, dashboard) built from one Docker image. Rationale: ADR-015, ADR-019, ADR-032.

## 1. Components and responsibilities

| Component | Module | Responsibility | Does NOT |
|---|---|---|---|
| Config | `predmaint/config.py` | Single source of defaults (`rul_cap`, `window_size`, seeds, band thresholds, paths) and env-var overrides. | Hold secrets. |
| Data | `predmaint/data/` | Download + checksum (`download.py`), parse raw files (`load.py`), validate data contract (`validate.py`), compute labels (`labels.py`). | Engineer features. |
| Features | `predmaint/features/build.py` | Sensor selection, causal rolling features, "last row per unit" extraction. **Single code path used by training, evaluation and the API.** | Fit models. |
| Models | `predmaint/models/` | Model definitions + 12-config search grid (`registry.py`), CV + training + evaluation (`train.py`), metrics (`metrics.py`), artifact save/load + metadata (`artifacts.py`), ADR-024 selection and promotion (`selection.py`), prediction-interval training/evaluation (`intervals.py`, M8), anomaly-detector training/evaluation (`anomaly.py`, M9), permutation feature importance (`explain.py`, M12), inference contract incl. interval bounds and anomaly scoring (`predict.py`). | Know about HTTP. |
| API | `predmaint/api/` | FastAPI app (`main.py`), Pydantic schemas (`schemas.py`), model-holder service (`service.py`), error envelope (`errors.py`). Endpoints `/health`, `/predict`, `/anomaly` (M9), `/model/info`. | Train models or read raw data files. |
| Dashboard | `predmaint/dashboard/` | Streamlit UI (`app.py`), a thin HTTP client returning typed results (`api_client.py`), and display helpers (`helpers.py`: true RUL, request bodies, trajectory sampling). | Import model, feature, inference or API-server code, or compute predictions. |
| CLI scripts | `scripts/` | Thin entrypoints (`download_data.py`, `train.py`, `make_request_payload.py`) that call package functions, plus the Docker smoke test `smoke_test.sh`. | Contain logic. |

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
    ART --> IV[models.intervals<br/>q0.1 / q0.9 HGB, M8]
    IV --> IART[(models/model_intervals.*)]
    ART --> AN[models.anomaly<br/>healthy ref RUL ≥ 125,<br/>Mahalanobis + OOF q0.99, M9<br/>+ EWM half-life 10, M11]
    AN --> AART[(models/model_anomaly.*)]
  end
  subgraph Runtime["Runtime (docker compose, one image)"]
    ART -->|committed, baked into image| API[api: FastAPI :8000<br/>/predict, /anomaly]
    IART -->|for_model_version check| API
    AART -->|for_model_version check| API
    FIMP[(models/model_feature_importance.json<br/>scripts/train.py --explain, M12)] -->|for_model_version check| API
    API -->|features.build + models.predict| API
    DASH[dashboard: Streamlit :8501] -->|HTTP JSON<br/>API_BASE_URL| API
    NASA -->|first start: download_data.py<br/>into ./data bind mount| DASH
    USER((User)) --> DASH
    CLIENT((curl / client)) --> API
  end
```

## 3. Data flow

1. `download_data.py` → `data/raw/cmapss_outer.zip` → nested `CMAPSSData.zip` → `data/raw/CMAPSSData/*.txt` (only expected member names are extracted; checksums verified).
2. `load.py` parses whitespace-delimited, header-less files into named DataFrames; `validate.py` enforces the contract in [04](04_DATASET_SPECIFICATION.md) §6.
3. No processed data is persisted for the MVP (FD001 parses in well under a second at this size); `data/processed/` is reserved and gitignored.

## 4. Training flow (`scripts/train.py --model baseline`, `--model primary`, `--select`)

1. Verify checksums (NFR-007) → load → validate.
2. Label train rows (FR-004); determine kept sensors from **train only** (FR-005).
3. Build features; keep rows that have a full trailing window of `window_size` cycles (= `cycle ≥ window_size` for training histories; ADR-005).
4. Per candidate run: 5-fold GroupKFold CV by `unit` → fold metrics. The primary run first scores the 12 configurations of the bounded search by CV RMSE (ADR-026) and keeps the best one.
5. Fit each candidate on all training rows; predict the last row of each test unit; compute test MAE/RMSE/NASA score.
6. `scripts/train.py --select`: apply ADR-024 (CV RMSE); copy the winner to `model.joblib`, write `model_metadata.json` with the comparison of both candidates, and `reports/model_comparison.{json,md}`.

## 5. Inference flow (`POST /predict`)

1. Pydantic validates the request shape and types (422 on failure).
2. `ModelService` only converts the records and calls `models.predict.predict_rul`, which checks, in order: ≥ `window_size` records, required columns non-null, finite values / integer cycles, consecutive cycles (422 with specific codes; 06 §6).
3. Only the last `window_size` records → `features.build` (same function as training) → one feature row.
4. Pipeline predict → clamp to [0, RUL_CAP] (ADR-027) → health status (ADR-025) → out-of-range warnings → if a matching interval artifact is loaded, q0.1/q0.9 predictions on the same feature row, clamped and ordered around the point (ADR-034); otherwise null bounds.
5. Response with `model_version` from metadata.

**`POST /anomaly` (M9, ADR-035):** same request schema and the same validation function as `/predict` (identical 422 errors). Features are built over the whole request (`features.build`), the detector scores every full-window row, and the response carries the last row's raw and normalised score (`score / threshold`), `is_anomalous` (normalised > 1) and, with `?include_series=true`, the normalised score for each full-window cycle. The last-cycle score depends only on the last `window_size` records, as for `/predict`. Without a loaded, matching detector → 503 `ANOMALY_UNAVAILABLE`; `/predict` is unaffected. M11 added an optional EWM smoothing of the per-cycle distances (ADR-038). It did not meet FR-021, so it is **disabled for serving** (ADR-039, `smoothing_enabled: false`): responses carry `smoothed: false` and the raw score, which depends only on the last 30 records.

The exact contract is in [06](06_API_SPECIFICATION.md); the function-level contract in [05](05_ML_DESIGN.md) §10.

## 6. API ↔ dashboard interaction

Confirmed in M5 (the API server log shows the dashboard's real calls; 14 §M5):

- The dashboard reads `API_BASE_URL` (default `http://localhost:8000`; `http://api:8000` in compose), `API_TIMEOUT_SECONDS` (default 10) and `DATA_DIR`. All HTTP goes through `dashboard/api_client.py` (`requests`). It never raises: each call returns an `ApiResult` with the success body or the docs/06 error envelope unchanged; network failures become a client-side `API_UNREACHABLE` envelope.
- Every script run calls `GET /health` (status bar) and, if a model is loaded, `GET /model/info` (kept sensors for the chart, health bands, model-info table).
- **Engine explorer:** the selected engine's first N cycles (raw FD001 under `DATA_DIR`, read with `predmaint.data.load`) become a docs/06 request body; `POST /predict` returns the predicted RUL and status shown in the UI. True RUL is computed locally from the dataset (`helpers.true_rul`).
- **RUL trajectory:** `POST /predict` every 5 cycles (`TRAJECTORY_STEP`) from cycle 30 to the engine's last cycle, cached with `st.cache_data` keyed by (dataset, unit, step, model_version). A failed call raises, so partial results are never cached.
- **Anomaly (M9), only when `/health.anomaly_loaded` is true:** the explorer makes one `POST /anomaly` call per prediction and shows the normalised score with a NORMAL/ANOMALOUS badge. The trajectory view makes **one** `POST /anomaly?include_series=true` call on the engine's full history (cached like the trajectory) and charts the normalised score with the threshold at 1.0. Missing or failed anomaly responses hide these elements; the RUL views are unchanged. Since the M11 close-out (ADR-039) the badge and chart use the served raw score; the chart caption says smoothing is evaluated but not served, and the model-info anomaly table shows raw (served) and smoothed (evaluated, not served) evaluations side by side.
- **Upload:** a docs/06 JSON body from the user is posted as-is; the response or error envelope is shown verbatim. This works without the raw dataset.
- Import boundary: dashboard modules may import only `predmaint.config`, `predmaint.data.load` and `predmaint.dashboard.*` — enforced by `tests/test_dashboard_client.py`.

## 7. Artifact handling

- Produced only by `scripts/train.py` into `MODEL_DIR` (default `models/`).
- `model.joblib`: a fitted sklearn `Pipeline`: a `ColumnTransformer` that selects the 43 training features (so the kept sensors live in the artifact, ADR-023), an optional scaler (baseline only), then the estimator. Feature construction parameters (`window_size`, kept sensors, feature order) are also in `model_metadata.json` and are applied by `features.build`.
- The served `model.joblib` + `model_metadata.json` are committed (ADR-030) and copied into the image (ADR-032), together with `model_intervals.*` (M8) and `model_anomaly.*` (M9), which record the `for_model_version` they belong to; candidate artifacts stay local. Retraining = train + select + intervals + anomaly + commit all six files + rebuild (10 §6).
- Loaded once at API startup (FastAPI lifespan). Metadata `library_versions.scikit-learn` is compared with the runtime version and exposed as `sklearn_version_match` in `/model/info`; `tests/test_model_artifact.py` fails on a mismatch.

## 8. Failure handling

| Failure | Behaviour |
|---|---|
| Artifact or metadata missing/corrupt at startup | API still starts; logs a WARNING; `/health` → 200 with `model_loaded: false` (ADR-028); `/predict`, `/model/info` → 503 `MODEL_UNAVAILABLE`. Compose healthcheck stays unhealthy. |
| Feature-importance file missing, unreadable or for another model version (M12) | API logs a WARNING; `/model/info.feature_importance` is `null`; nothing else changes. |
| Interval or anomaly artifact missing, corrupt or for another model version | API logs a WARNING and serves without it: null interval fields (M8); `/anomaly` → 503 `ANOMALY_UNAVAILABLE` and `/health.anomaly_loaded: false` (M9). `/predict` and the healthcheck are unaffected. |
| Invalid request body | 422 `VALIDATION_ERROR` with field-level details. |
| Business-rule violation | 422 with `INSUFFICIENT_CYCLES`, `MISSING_SENSORS`, or `NON_CONSECUTIVE_CYCLES`. |
| Values outside training range | 200 with entries in `warnings` (not rejected). |
| Unexpected exception | 500 `INTERNAL_ERROR`; stack trace logged server-side only. |
| API down / timeout (dashboard) | Dashboard shows "API unreachable at `<API_BASE_URL>`" and disables prediction; no crash. |
| Dataset download fails / checksum mismatch | `download_data.py` exits non-zero; training refuses to start. In the dashboard container it only logs a warning; the explorer and trajectory show the download hint (ADR-031). |
