# 06 — API Specification

FastAPI app `predmaint.api.main:app`, served by uvicorn on port **8000**. JSON only. No authentication (local MVP; ADR-012). OpenAPI UI at `/docs` is FastAPI's default and is not part of the contract.

## 1. Endpoints

| Method | Path | Purpose | Success | Errors |
|---|---|---|---|---|
| GET | `/health` | Liveness + model readiness | 200 | 503 `MODEL_UNAVAILABLE` |
| POST | `/predict` | RUL prediction for one engine | 200 | 422 (4 codes), 503, 500 |
| GET | `/model/info` | Loaded model metadata | 200 | 503 `MODEL_UNAVAILABLE` |

## 2. `GET /health`

200:

```json
{"status": "ok", "model_loaded": true, "model_version": "primary-20261108-f4c86df"}
```

503 (artifact missing/corrupt): error envelope (§6) with `code = "MODEL_UNAVAILABLE"`.

**No-artifact rule (ADR-022):** whenever no model artifact is loaded, `/health`, `/predict` and `/model/info` all return HTTP 503 with the standard error body (§6), `code = "MODEL_UNAVAILABLE"`. The API process still starts.

## 3. `POST /predict`

### 3.1 Request

```json
{
  "unit_id": "test-unit-1",
  "records": [
    {"cycle": 2, "s_1": 518.67, "s_2": 641.71, "s_3": 1588.45, "s_4": 1395.42, "...": "..."},
    {"cycle": 3, "s_1": 518.67, "s_2": 642.46, "s_3": 1586.94, "s_4": 1401.34, "...": "..."}
  ]
}
```

(Illustrative and truncated — `"...": "..."` stands for the remaining fields and would be rejected by `extra="forbid"`. A valid request has ≥ `window_size` = 30 records. Generate real payloads with `python scripts/make_request_payload.py --unit 1 --last 30`.)

### 3.2 Request field rules (Pydantic v2, `extra="forbid"`, `allow_inf_nan=False`)

| Field | Type | Rule |
|---|---|---|
| `unit_id` | string \| null | Optional; 1–64 chars; echoed back. |
| `records` | array of `SensorRecord` | Required; 1 ≤ length ≤ 1000 (schema). Business rule: ≥ `window_size` (from metadata). |
| `SensorRecord.cycle` | int | Required; ≥ 1. |
| `SensorRecord.s_1` … `s_21` | float \| null | Optional at schema level; finite. Every sensor in the model's `required_sensors` must be non-null in every record (business rule). Others are ignored. |
| `SensorRecord.setting_1` … `setting_3` | float \| null | Optional; accepted so raw FD rows map 1:1; ignored by the FD001 model. |

### 3.3 Business validation (after schema validation)

1. `len(records) ≥ window_size` → else 422 `INSUFFICIENT_CYCLES`. Confirmed: fewer than `WINDOW_SIZE` (30, from `predmaint.config`, recorded in metadata) cycles always yields HTTP 422 with the standard error body (§6).
2. All `required_sensors` present and non-null in all records → else 422 `MISSING_SENSORS` (details list the missing names).
3. Records must be in chronological order as sent: `cycle` strictly increasing by exactly 1 (no sorting is applied; out-of-order, duplicate or missing cycles) → else 422 `NON_CONSECUTIVE_CYCLES`.
4. Prediction uses the last `window_size` records; the feature row is the last cycle.
5. Values of required sensors outside metadata `training_ranges` → still 200, with a warning per (sensor) in `warnings`.

### 3.4 Response 200

```json
{
  "unit_id": "test-unit-1",
  "predicted_rul": 87.4,
  "health_status": "healthy",
  "cycles_received": 31,
  "cycles_used": 30,
  "last_cycle": 31,
  "model_version": "primary-20261108-f4c86df",
  "warnings": []
}
```

| Field | Type | Rule |
|---|---|---|
| `predicted_rul` | float | In `[0, rul_cap]` (ADR-027), cycles, rounded to 2 decimals. |
| `health_status` | `"critical" \| "warning" \| "healthy"` | ADR-025, constants in `predmaint.config.HEALTH_BANDS`: `critical` if `predicted_rul` < 20; `warning` if 20 ≤ `predicted_rul` < 60; `healthy` if ≥ 60. |
| `cycles_received` | int | Number of records in the request. |
| `last_cycle` | int | `cycle` of the last record. |
| `cycles_used` | int | = `window_size`. |
| `model_version` | string | From metadata. |
| `warnings` | array of string | e.g. `"OUT_OF_TRAINING_RANGE: s_7 value 560.1 outside [549.85, 556.06]"` (format illustrative of shape; ranges come from metadata). |

## 4. `GET /model/info`

### 4.1 Response 200

```json
{
  "model_version": "primary-20261108-f4c86df",
  "model_name": "primary",
  "trained_at": "2026-11-08T15:04:41Z",
  "git_sha": "unknown",
  "data_subset": "FD001",
  "train_sha256": "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8",
  "window_size": 30,
  "rul_cap": 125,
  "required_sensors": ["s_2", "..."],
  "feature_names": ["cycle", "s_2", "s_2_roll_mean", "s_2_roll_std", "..."],
  "health_bands": {"critical_below": 20, "warning_below": 60},
  "metrics": {"test": {"mae": 0.0, "rmse": 0.0, "nasa_score": 0.0}, "cv": {"mae_mean": 0.0, "rmse_mean": 0.0}},
  "selected_by": "cv_rmse",
  "sklearn_version_match": true
}
```

Values illustrate shape only; all fields are read from `model_metadata.json` (05 §9) except `sklearn_version_match` (runtime comparison).

### 4.2 Field source

`metrics` = the selected candidate's `test` and `cv` means. `selected_by` is copied from metadata (ADR-024). The full candidate comparison stays in `model_metadata.json` / `reports/model_comparison.md`.

### 4.3 Required fields (FR-012)

`model_version, model_name, trained_at, git_sha, data_subset, train_sha256, window_size, rul_cap, required_sensors, feature_names, health_bands, metrics, selected_by, sklearn_version_match`.

## 5. Status codes

| Code | When |
|---|---|
| 200 | Success |
| 404 | Unknown path (envelope, `NOT_FOUND`) |
| 405 | Wrong method (envelope, `METHOD_NOT_ALLOWED`) |
| 422 | Schema or business validation failure |
| 503 | Model not loaded |
| 500 | Unhandled server error |

## 6. Error envelope

All non-2xx responses:

```json
{"error": {"code": "MISSING_SENSORS", "message": "Required sensors missing or null: s_7, s_12", "details": [{"sensor": "s_7"}, {"sensor": "s_12"}]}}
```

Exception classes live in `predmaint.models.predict` (inference contract, 05 §10); all input errors subclass `InputContractError`.

| `code` | HTTP | Raised by |
|---|---|---|
| `VALIDATION_ERROR` | 422 | Pydantic `RequestValidationError` (`details` = Pydantic errors: `loc`, `msg`, `type`); also `InvalidValuesError` (non-finite values, non-integer cycles) should one pass schema validation. |
| `INSUFFICIENT_CYCLES` | 422 | `InsufficientHistoryError` |
| `MISSING_SENSORS` | 422 | `MissingColumnsError` (`cycle` or required sensor missing/null; `.columns` → `details`) |
| `NON_CONSECUTIVE_CYCLES` | 422 | `NonConsecutiveCyclesError` |
| `MODEL_UNAVAILABLE` | 503 | `ModelNotAvailableError` from `load_model` / model not loaded |
| `NOT_FOUND` / `METHOD_NOT_ALLOWED` | 404 / 405 | Starlette HTTP exceptions |
| `INTERNAL_ERROR` | 500 | Any other exception; message is generic, trace logged only. |

## 7. Example calls

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/model/info
python scripts/make_request_payload.py --unit 1 --last 30 > /tmp/payload.json
curl -s -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" -d @/tmp/payload.json
# Error example: too few cycles → 422 INSUFFICIENT_CYCLES
python scripts/make_request_payload.py --unit 1 --last 5 > /tmp/short.json
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @/tmp/short.json
```

`make_request_payload.py` reads `DATA_DIR/raw/CMAPSSData/test_FD001.txt` via `predmaint.data.load` and prints a valid request body (all 21 sensors + settings for the chosen unit's last N cycles).
