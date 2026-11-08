# 06 — API Specification

FastAPI app built by `predmaint.api.main:create_app()` (module-level `app` for uvicorn). Run it with `python -m predmaint.api.main`, which binds `$API_HOST:$API_PORT` (default `127.0.0.1:8000`), or with `uvicorn predmaint.api.main:app --host … --port …`. The served model is loaded once at startup from `$MODEL_PATH` (default `$MODEL_DIR/model.joblib`; env vars in [10](10_DEPLOYMENT_PLAN.md) §3). JSON only. No authentication, CORS or rate limiting (local MVP; ADR-012). Interactive docs at `/docs` (title, version from the package, description, and a real example request when the server runs from the repo root).

All bodies below were copied from `curl` / TestClient output against the served model `primary-20261108-f4c86df` on 2026-11-08 (Milestone 4; interval fields from Milestone 8; `/anomaly`, `anomaly_loaded`, `/model/info.anomaly` from Milestone 9; calibrated bounds, `interval_calibrated`, `/model/info.intervals` calibration fields from Milestone 10; smoothed anomaly fields (inactive since the M11 close-out, ADR-039) and `api_version` 0.5.0 from Milestone 11).

## 1. Endpoints

| Method | Path | Purpose | Success | Errors |
|---|---|---|---|---|
| GET | `/health` | Liveness + model readiness | 200 (also without a model, ADR-028) | — |
| POST | `/predict` | RUL prediction for one engine | 200 | 422 (4 codes), 503, 500 |
| POST | `/anomaly` | Anomaly score for one engine (M9, ADR-035); `?include_series=true` adds the per-cycle series | 200 | 422 (same codes as `/predict`), 503 `ANOMALY_UNAVAILABLE`, 500 |
| GET | `/model/info` | Served model metadata and metrics | 200 | 503 `MODEL_UNAVAILABLE` |

**Interval artifact (ADR-034):** loaded at startup from `model_intervals.joblib` next to `MODEL_PATH`. If it is absent, unloadable or built for a different `model_version`, the API logs a WARNING and serves point predictions with `rul_lower`/`rul_upper`/`interval_level` = `null`; nothing else changes.

**Anomaly artifact (ADR-035):** loaded at startup from `model_anomaly.joblib` next to `MODEL_PATH`. If it is absent, unloadable, built for a different `model_version` or uses features the served model does not produce, the API logs a WARNING, `/health.anomaly_loaded` is `false`, `/model/info.anomaly` is `null` and `/anomaly` returns 503 `ANOMALY_UNAVAILABLE`; `/predict` is unaffected.

**No-artifact rule (ADR-022 as amended by ADR-028):** if the model cannot be loaded at startup, the API still starts and logs a WARNING. `/health` returns **200** with `"model_loaded": false`; `/predict` and `/model/info` return **503** `MODEL_UNAVAILABLE` in the standard error body (§6).

## 2. `GET /health`

With a model (200):

```json
{"status":"ok","model_loaded":true,"model_version":"primary-20261108-f4c86df","api_version":"0.5.0","intervals_loaded":true,"anomaly_loaded":true}
```

Without a model, e.g. `MODEL_PATH=/nonexistent` (200):

```json
{"status":"degraded","model_loaded":false,"model_version":null,"api_version":"0.5.0","intervals_loaded":false,"anomaly_loaded":false}
```

| Field | Type | Rule |
|---|---|---|
| `status` | `"ok" \| "degraded"` | `degraded` when no model is loaded. |
| `model_loaded` | bool | Whether `/predict` can serve. Readiness probes (M6 compose healthcheck) must check this field, not only the status code. |
| `model_version` | string \| null | Served model version; `null` without a model. |
| `api_version` | string | `predmaint.__version__` (0.2.0 since M8, 0.3.0 since M9, 0.4.0 since M10, 0.5.0 since M11). |
| `intervals_loaded` | bool | Whether a prediction-interval artifact matching the served model is loaded (ADR-034). |
| `anomaly_loaded` | bool | Whether an anomaly detector matching the served model is loaded (ADR-035). Not part of the readiness check. |

## 3. `POST /predict`

### 3.1 Request

`examples/predict_request_min.json` holds FD001 test engine 1, cycles 2–31 (30 records). Its first record is shown here; the other 29 have the same fields. `examples/predict_request.json` holds all 31 cycles of the same engine.

```json
{"unit_id": "test-unit-1", "records": [{"cycle": 2, "setting_1": -0.0027, "setting_2": -0.0003, "setting_3": 100.0, "s_1": 518.67, "s_2": 641.71, "s_3": 1588.45, "s_4": 1395.42, "s_5": 14.62, "s_6": 21.61, "s_7": 554.85, "s_8": 2388.01, "s_9": 9054.42, "s_10": 1.3, "s_11": 47.5, "s_12": 522.16, "s_13": 2388.06, "s_14": 8139.62, "s_15": 8.3803, "s_16": 0.03, "s_17": 393.0, "s_18": 2388.0, "s_19": 100.0, "s_20": 39.02, "s_21": 23.3916}, "…29 more records…"]}
```

(`"…29 more records…"` marks the elision here only; it is not valid input.) Build payloads for any test engine with `python scripts/make_request_payload.py --unit N [--last K]`.

### 3.2 Request field rules (Pydantic v2, `extra="forbid"`, `allow_inf_nan=False`)

| Field | Type | Rule |
|---|---|---|
| `unit_id` | string \| null | Optional; 1–64 chars; echoed back. |
| `records` | array of `SensorRecord` | Required; **1 ≤ length ≤ 1000** (`MAX_RECORDS_PER_REQUEST`, bounds memory) → otherwise 422 `VALIDATION_ERROR`. Business rule: ≥ `window_size`. |
| `SensorRecord.cycle` | int | Required; ≥ 1; fractional or non-numeric values rejected. |
| `SensorRecord.s_1` … `s_21` | float \| null | Optional at schema level; finite (JSON `NaN`/`Infinity` rejected). The model's 14 `required_sensors` (`s_2, s_3, s_4, s_7, s_8, s_9, s_11, s_12, s_13, s_14, s_15, s_17, s_20, s_21`) must be non-null in every record. Others are ignored. |
| `SensorRecord.setting_1` … `setting_3` | float \| null | Optional; accepted so raw FD rows map 1:1; ignored by the FD001 model. |
| any other field | — | Rejected (`extra="forbid"`), at top level and in records. |

### 3.3 Business validation (performed only by `predict_rul`, 05 §10; the API duplicates none of it)

1. `len(records) ≥ window_size` (30) → else 422 `INSUFFICIENT_CYCLES`.
2. `cycle` and all `required_sensors` present and non-null in every record → else 422 `MISSING_SENSORS` (`details` lists each missing sensor).
3. Records in chronological order as sent: `cycle` strictly increasing by exactly 1. No sorting is applied, so out-of-order, duplicate or missing cycles → 422 `NON_CONSECUTIVE_CYCLES`.
4. Only the last `window_size` records are used; the feature row is the last cycle. Older records cannot change the result.
5. Required-sensor values in the used window outside the metadata `training_ranges` → still 200, with an `OUT_OF_TRAINING_RANGE: …` string per sensor in `warnings`.

### 3.4 Response 200

`curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d @examples/predict_request_min.json`:

```json
{"unit_id":"test-unit-1","predicted_rul":125.0,"health_status":"healthy","cycles_received":30,"cycles_used":30,"last_cycle":31,"model_version":"primary-20261108-f4c86df","warnings":[],"rul_lower":116.7,"rul_upper":125.0,"interval_level":0.8,"interval_calibrated":true}
```

Since M8 (CR-008) responses also carry the interval fields. Since M10 the bounds carry the conformal correction (Q = 1.742) and `interval_calibrated`. Real bodies from the M10 local API (M8 bounds were [4.84, 18.76] and [118.45, 125.0]):

```json
{"unit_id":"test-unit-34","predicted_rul":6.3,"health_status":"critical","cycles_received":203,"cycles_used":30,"last_cycle":203,"model_version":"primary-20261108-f4c86df","warnings":[],"rul_lower":3.1,"rul_upper":20.5,"interval_level":0.8,"interval_calibrated":true}
{"unit_id":"test-unit-1","predicted_rul":125.0,"health_status":"healthy","cycles_received":31,"cycles_used":30,"last_cycle":31,"model_version":"primary-20261108-f4c86df","warnings":[],"rul_lower":116.7,"rul_upper":125.0,"interval_level":0.8,"interval_calibrated":true}
```

The same call with `examples/predict_request.json` (31 records) returns the identical prediction with `"cycles_received":31`. The late-life example `examples/predict_request_late.json` (test engine 34, 203 records, true RUL 7) returns `"predicted_rul":6.3,"health_status":"critical"` with interval [3.1, 20.5] since M10 ([4.84, 18.76] before calibration; sidecar; reproduced through the API by `tests/test_real_data.py` and shown by the dashboard in M5). The expected values are recorded in `examples/expected_predictions.json`. Engine 1's true RUL at cycle 31 is 112; with only 31 cycles of early-life history, the model's prediction sits at the 125 cap.

| Field | Type | Rule |
|---|---|---|
| `unit_id` | string \| null | Echo of the request. |
| `predicted_rul` | float | In `[0, rul_cap]` (ADR-027), cycles, rounded to 2 decimals. |
| `health_status` | `"critical" \| "warning" \| "healthy"` | ADR-025 (`predmaint.config.HEALTH_BANDS`): `critical` if `predicted_rul` < 20; `warning` if 20 ≤ `predicted_rul` < 60; `healthy` if ≥ 60. |
| `cycles_received` | int | Number of records in the request. |
| `cycles_used` | int | = `window_size` (30). |
| `last_cycle` | int | `cycle` of the last record. |
| `model_version` | string | Served model version. |
| `warnings` | array of string | Out-of-training-range messages (§3.3 rule 5). |
| `rul_lower` / `rul_upper` | float \| null | Bounds of the prediction interval (ADR-034): clamped to [0, `rul_cap`], `rul_lower ≤ predicted_rul ≤ rul_upper`, rounded to 2 decimals. `null` when no matching interval artifact is loaded (point prediction unchanged). |
| `interval_level` | float \| null | Nominal level, 0.8 (q0.1–q0.9); `null` without intervals. Calibrated coverage on held-out engines is 0.777 against the capped target; 0 for true RUL > 125 against unclipped truth (16 §9). |
| `interval_calibrated` | bool \| null | `true` when the bounds include the CQR correction Q (ADR-037); `false` for interval metadata without `conformal_q` (v1.1 bounds); `null` without intervals. |

## 4. `GET /model/info`

### 4.1 Response 200

`curl -s localhost:8000/model/info` (pretty-printed; the 43 `feature_names` are elided here):

```json
{
 "model_name": "primary",
 "model_version": "primary-20261108-f4c86df",
 "trained_at": "2026-11-08T15:04:41Z",
 "git_sha": "f4c86df37c4cfc83af8b53a6c3026eecaa385341",
 "selected_by": "cv_rmse",
 "data_subset": "FD001",
 "train_sha256": "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8",
 "window_size": 30,
 "rul_cap": 125,
 "n_features": 43,
 "required_sensors": ["s_2", "s_3", "s_4", "s_7", "s_8", "s_9", "s_11", "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21"],
 "feature_names": ["cycle", "s_2", "s_2_roll_mean", "s_2_roll_std", "…"],
 "health_bands": {"critical_below": 20, "warning_below": 60},
 "metrics": {
  "cv": {"mae_mean": 11.842774355821547, "mae_std": 0.6763920356345671, "rmse_mean": 16.558936995946503, "rmse_std": 0.6954267044128535},
  "test": {"mae": 12.566732875707899, "rmse": 16.609852374016835, "nasa_score": 441.5350326191249},
  "test_clipped_truth": {"mae": 11.496732875707899, "rmse": 15.333382373407852}
 },
 "sklearn_version_match": true,
 "intervals": {"for_model_version": "primary-20261108-f4c86df", "method": "HistGradientBoostingRegressor(loss='quantile') at q=0.1 and q=0.9 (ADR-034)", "quantiles": [0.1, 0.9], "interval_level": 0.8, "trained_at": "2026-11-08T18:55:47Z",
   "cv_oof": {"n": 17731, "coverage_unclipped": 0.5087699509333935, "coverage_clipped": 0.7370706671930517, "mean_width": 37.196710409508896, "median_width": 36.658377404963076, "n_reordered": 1023},
   "test": {"n": 100, "coverage_unclipped": 0.74, "coverage_clipped": 0.82, "mean_width": 42.12697130358668, "median_width": 41.95713193422868, "n_reordered": 8},
   "calibrated": true, "conformal_q": 1.7418749999999932, "alpha": 0.2,
   "calibration": {"method": "split-conformal CQR: E = max(lo - y, y - hi) on raw out-of-fold quantile predictions; Q = ceil((n+1)(1-alpha))-th smallest E; served bounds = order_bounds(point, lo - Q, hi + Q) (clamp to [0, rul_cap], order around point)", "target": "capped RUL = min(true RUL, rul_cap), the quantile models' training target",
     "q_eval": 1.3203055725249726, "n_calibration_engines": 50, "n_evaluation_engines": 50,
     "evaluation_half_before": {"n": 8921, "coverage_unclipped": 0.5036430893397601, "coverage_clipped": 0.7338863356125995, "mean_width": 37.15275304533112, "median_width": 36.54240506808031, "n_reordered": 569},
     "evaluation_half_after": {"n": 8921, "coverage_unclipped": 0.5360385606994732, "coverage_clipped": 0.7768187422934648, "mean_width": 39.188196797485176, "median_width": 38.574088329678375, "n_reordered": 351},
     "test_before": {"n": 100, "coverage_unclipped": 0.74, "coverage_clipped": 0.82, "mean_width": 42.12697130358668, "median_width": 41.95713193422868, "n_reordered": 8},
     "test_after": {"n": 100, "coverage_unclipped": 0.75, "coverage_clipped": 0.85, "mean_width": 44.74103261775972, "median_width": 43.69900693422867, "n_reordered": 6},
     "acceptance_criterion": "evaluation-half coverage (capped target) within 0.80 ± 0.03", "acceptance_met": true,
     "caveat": "Split-conformal coverage assumes exchangeable rows; rows of one engine are correlated, so the guarantee is approximate: the effective sample size is closer to the number of engines than to the number of rows."}},
 "anomaly": {"model_version": "anomaly-20261108-4734b37", "for_model_version": "primary-20261108-f4c86df", "method": "mahalanobis_ledoitwolf", "threshold": 5.40363364967852, "threshold_quantile": 0.99, "healthy_rows": 5329, "trained_at": "2026-11-08T21:00:02Z",
   "trend_median_train_oof": 0.6798250694250569, "false_alarm_rate_healthy": 0.010133233252017265, "detection_fraction_of_life_median": 0.7581860706860707, "detection_true_rul_median": 49.0, "never_flagged_fraction": 0.02, "acceptance_met": false,
   "smoothing": {"method": "ewm", "halflife": 10, "adjust": true}, "smoothing_enabled": false, "smoothed_threshold": 5.090594528745487, "smoothed_acceptance_met": false,
   "evaluation_raw": {…}, "evaluation_smoothed": {…}}
}
```

### 4.2 Field source

All fields come from `model_metadata.json` (05 §9), except `n_features` (= `len(feature_names)`) and `sklearn_version_match` (metadata `library_versions["scikit-learn"]` compared with the running scikit-learn). `metrics` holds the served model's CV means/stds (folds omitted) and official-test metrics. The full candidate comparison stays in `model_metadata.json` and `reports/model_comparison.md`.

### 4.3 Required fields (FR-012)

`model_name, model_version, trained_at, git_sha, selected_by, data_subset, train_sha256, window_size, rul_cap, n_features, required_sensors, feature_names, health_bands, metrics, sklearn_version_match`, plus `intervals` (object or `null`, ADR-034; `cv_oof`/`test` are the uncalibrated v1.1 numbers, and since M10 `calibrated`, `conformal_q`, `alpha` and `calibration` (before/after coverage, FR-020 verdict, exchangeability caveat; `null` for uncalibrated metadata) are added per ADR-037) and `anomaly` (object or `null`, ADR-035; fields from `model_anomaly_metadata.json`: version, method, OOF threshold, healthy-row count and the headline **raw** train-OOF evaluation incl. `acceptance_met` (FR-019); since M11 also `smoothing`, `smoothing_enabled` (`false` since ADR-039), `smoothed_threshold`, `smoothed_acceptance_met` (FR-021) and the full `evaluation_raw` / `evaluation_smoothed` objects, identical to the `evaluation` and `smoothed.evaluation` sections of `reports/metrics_anomaly.json`, abbreviated `{…}` above).

## 4a. `POST /anomaly` (M9, ADR-035; smoothing evaluated in M11, not served, ADR-038/039)

**Request:** identical to `POST /predict` (§3.1–3.3): same Pydantic schema and the same business validation function, so every 422 body is identical to `/predict`'s for the same input (tested). Query parameter `include_series` (bool, default `false`; an unparseable value → 422 `VALIDATION_ERROR`).

**Served behaviour since the M11 close-out (ADR-039): the raw score.** The pre-registered smoothing remedy (ADR-038, EWM half-life 10) did not meet FR-021 and delayed detection, so the anomaly metadata sets `smoothing_enabled: false`. `/anomaly` serves the **raw** M9 score: `smoothed: false`, `cycles_in_smoothing: null`, `normalized_score == raw_normalized_score`, raw threshold 5.4036. The raw last-cycle score depends only on the last 30 records, as in M9. The smoothing fields are present but inactive. The smoothing config, smoothed threshold (5.0906) and smoothed evaluation stay in the metadata and in `/model/info.anomaly` (`smoothing_enabled: false`) as a documented negative result (16 §10d).

**Response 200** — real bodies from the M11 close-out local API (`examples/predict_request.json`, test engine 1; `examples/predict_request_late.json`, test engine 34). The values equal M9's; only the detector version string changed:

```json
{"unit_id":"test-unit-1","last_cycle":31,"anomaly_score":3.5973,"normalized_score":0.6657,"threshold":5.4036,"is_anomalous":false,"model_version":"anomaly-20261108-4734b37","raw_normalized_score":0.6657,"smoothed":false,"cycles_in_smoothing":null,"series":null}
{"unit_id":"test-unit-34","last_cycle":203,"anomaly_score":5.9519,"normalized_score":1.1015,"threshold":5.4036,"is_anomalous":true,"model_version":"anomaly-20261108-4734b37","raw_normalized_score":1.1015,"smoothed":false,"cycles_in_smoothing":null,"series":null}
```

With `?include_series=true` (engine 34; 174 = 203 − 30 + 1 entries, first and last three shown):

```json
{"unit_id":"test-unit-34","last_cycle":203,"anomaly_score":5.9519,"normalized_score":1.1015,"threshold":5.4036,"is_anomalous":true,"model_version":"anomaly-20261108-4734b37","raw_normalized_score":1.1015,"smoothed":false,"cycles_in_smoothing":null,
 "series":[{"cycle":30,"normalized_score":0.8445,"raw_normalized_score":0.8445},{"cycle":31,"normalized_score":0.8533,"raw_normalized_score":0.8533},{"cycle":32,"normalized_score":0.8337,"raw_normalized_score":0.8337}, …, {"cycle":201,"normalized_score":1.1271,"raw_normalized_score":1.1271},{"cycle":202,"normalized_score":1.1481,"raw_normalized_score":1.1481},{"cycle":203,"normalized_score":1.1015,"raw_normalized_score":1.1015}]}
```

**If smoothing were enabled** (`smoothing_enabled: true`, the M11 behaviour, kept in code and tested): the score becomes the EWM over **every full-window cycle in the request**. The last-cycle score then depends on how many cycles are sent, so clients would have to send the full history; `cycles_in_smoothing` = cycles sent − 29. The M11 bodies were, for example, engine 34: `normalized_score` 1.0472 (smoothed threshold 5.0906, `cycles_in_smoothing` 174); engine 1: 0.7277 (2 cycles). Sending 60 vs 203 cycles of engine 34 gives the same `raw_normalized_score` (tested).

| Field | Type | Rule |
|---|---|---|
| `unit_id` | string \| null | Echoed from the request. |
| `last_cycle` | int | Last cycle of the request. |
| `anomaly_score` | float | **Served (ADR-039):** the last row's Mahalanobis distance, which depends only on the last 30 records. Only if `smoothed` were true: EWM (half-life 10) over all full-window cycles sent. 4 decimals. |
| `normalized_score` | float | `anomaly_score / threshold` (4 decimals); > 1 means anomalous. |
| `threshold` | float | 99th percentile of out-of-fold healthy-row scores, raw 5.4036 as served; it would be the smoothed threshold 5.0906 only if `smoothed` were true. 4 decimals. |
| `is_anomalous` | bool | `normalized_score > 1` (computed before rounding). |
| `model_version` | string | Anomaly detector version (`anomaly-YYYYMMDD-<sha7>`; M11 run: `anomaly-20261108-4734b37`, same detector parameters as M9). |
| `raw_normalized_score` | float \| null | Unsmoothed last-row distance / raw threshold (the M9 score); depends only on the last 30 records. |
| `smoothed` | bool | Whether the EWM was applied (ADR-038). **`false` as served** (ADR-039, `smoothing_enabled: false`); present but inactive. |
| `cycles_in_smoothing` | int \| null | Full-window cycles the EWM covered (cycles sent − 29); `null` without smoothing, i.e. **`null` as served** (ADR-039). |
| `series` | list \| null | Only with `include_series=true`: `{cycle, normalized_score, raw_normalized_score}` for every cycle with a full trailing window (`cycles_received − 30 + 1` entries, last entry = the top-level values). |

The score is an unsupervised health indicator, not a validated fault detector; its trend target was not met by the raw score (FR-019). The one pre-registered remedy, smoothing (FR-021), was not met either and was rejected for serving (16 §10d).

## 5. Status codes

| Code | When |
|---|---|
| 200 | Success; also `/health` without a model (ADR-028) |
| 404 | Unknown path (`NOT_FOUND`) |
| 405 | Wrong method (`METHOD_NOT_ALLOWED`) |
| 422 | Schema or business validation failure |
| 503 | `/predict` or `/model/info` without a loaded model; `/anomaly` without a loaded, matching detector |
| 500 | Unhandled server error |

## 6. Error envelope

Every non-2xx response, produced by one handler layer (`predmaint.api.errors` plus the request middleware for 500s):

```json
{"error":{"code":"MISSING_SENSORS","message":"required columns missing or null: s_7","details":[{"sensor":"s_7"}]}}
```

Exception classes come from the inference contract (`predmaint.models.predict`, 05 §10).

| `code` | HTTP | Raised by | Real example (`message` / `details`) |
|---|---|---|---|
| `VALIDATION_ERROR` | 422 | Pydantic `RequestValidationError` (missing/extra field, wrong type, empty list, > 1000 records, NaN/Infinity, malformed JSON); also `InvalidValuesError` | `"Request validation failed"`, `[{"loc":["body","records",0,"cycle"],"msg":"Input should be a valid integer, unable to parse string as an integer","type":"int_parsing"}]` |
| `INSUFFICIENT_CYCLES` | 422 | `InsufficientHistoryError` | `"10 cycles received; at least 30 are required"`, `[]` |
| `MISSING_SENSORS` | 422 | `MissingColumnsError` | `"required columns missing or null: s_7"`, `[{"sensor":"s_7"}]` |
| `NON_CONSECUTIVE_CYCLES` | 422 | `NonConsecutiveCyclesError` | `"cycles must be strictly increasing by exactly 1 in the order given"`, `[]` |
| `MODEL_UNAVAILABLE` | 503 | `ModelNotAvailableError` (no model loaded) | `"No model loaded from /nonexistent: artifact not found: /nonexistent/model.joblib (run \`python scripts/train.py\`)"`, `[]` |
| `ANOMALY_UNAVAILABLE` | 503 | `AnomalyNotAvailableError` (`/anomaly` only: no point model, or no matching detector) | `"No anomaly detector loaded: no point model loaded"`, `[]`; with the detector files removed: `"No anomaly detector loaded: artifact not found: <dir>/model_anomaly.joblib (run \`python scripts/train.py\`)"` |
| `NOT_FOUND` / `METHOD_NOT_ALLOWED` | 404 / 405 | Starlette HTTP exceptions | `"Not Found"` / `"Method Not Allowed"`, `[]` |
| `HTTP_ERROR` | any other 4xx/5xx raised as a Starlette `HTTPException` | fallback in `predmaint.api.errors` (no route raises one today) | the exception's detail, `[]` |
| `INTERNAL_ERROR` | 500 | Any other exception | `"Internal server error"`, `[]`. No traceback in the body; the full traceback is logged. |

## 7. Logging

One line per request on logger `predmaint.api`: `METHOD PATH STATUS LATENCYms`, e.g. `POST /predict 200 11.8ms`. uvicorn's own access log is disabled by `python -m predmaint.api.main` to avoid duplicates. Model-load success is logged at INFO; degraded mode at WARNING; unhandled errors at ERROR with traceback. Level: `LOG_LEVEL`.

## 8. Example calls

```bash
python -m predmaint.api.main &                       # 127.0.0.1:8000
curl -s localhost:8000/health
curl -s localhost:8000/model/info
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d @examples/predict_request.json
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d @examples/predict_request_min.json
curl -s -X POST localhost:8000/anomaly -H "Content-Type: application/json" -d @examples/predict_request_late.json
curl -s -X POST "localhost:8000/anomaly?include_series=true" -H "Content-Type: application/json" -d @examples/predict_request_late.json
python scripts/make_request_payload.py --unit 1 --last 10 > /tmp/short.json   # → 422 INSUFFICIENT_CYCLES
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d @/tmp/short.json
```
