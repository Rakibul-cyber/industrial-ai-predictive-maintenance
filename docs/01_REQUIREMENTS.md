# 01 — Requirements

Each requirement has a measurable acceptance criterion and traces to the milestone in [08](08_IMPLEMENTATION_PLAN.md) that implements it. Verification commands live in 08; test design in [09](09_TESTING_STRATEGY.md).

## Functional requirements

| ID | Requirement | Acceptance criterion | Milestone |
|---|---|---|---|
| FR-001 | Download the C-MAPSS archive reproducibly. | `python scripts/download_data.py` fetches the archive from the URL in [04](04_DATASET_SPECIFICATION.md) §1, verifies the SHA-256 of the archive and of the FD001 files, and extracts to `data/raw/CMAPSSData/`. A checksum mismatch exits non-zero with a clear message. Re-running when files are present and valid performs no download. | M1 |
| FR-002 | Load FD001 files with the documented schema. | Loader returns DataFrames with exactly 26 named columns for train/test (`unit`, `cycle`, `setting_1..3`, `s_1..s_21`) and a 1-column RUL frame; real-data test asserts shapes 20631×26, 13096×26, 100×1. | M1 |
| FR-003 | Validate loaded data. | Validation fails (raises) on: wrong column count, any null, non-integer unit/cycle, cycles not starting at 1 or not strictly +1 per unit, RUL row count ≠ test unit count. Real FD001 passes. | M1 |
| FR-004 | Compute RUL training labels. | `RUL = max_cycle(unit) − cycle`, clipped at `rul_cap` (default 125, configurable). Unit tests: last cycle → 0; clip applied; unclipped when `rul_cap=None`. | M2 |
| FR-005 | Engineer features causally. | Sensors selected by the rule in [05](05_ML_DESIGN.md) §4; for each kept sensor: raw value, trailing rolling mean and std over `window_size` (default 30) rows of the same unit; plus `cycle`. Unit test proves changing rows after cycle *t* does not change features at *t*. | M2 |
| FR-006 | Train and cross-validate a baseline. | StandardScaler + Ridge evaluated with 5-fold GroupKFold by `unit`; per-fold and mean±std MAE/RMSE recorded in metadata. | M3 |
| FR-007 | Train and cross-validate a primary model. | HistGradientBoostingRegressor with fixed hyperparameters, same CV protocol and outputs as FR-006. | M3 |
| FR-008 | Evaluate on the official test set. | For each of 100 test units, predict at the last observed cycle; report MAE, RMSE (primary) and NASA score (secondary) against `RUL_FD001.txt`. | M3 |
| FR-009 | Select and persist the model. | Selection per ADR-010; `models/model.joblib` + `models/model_metadata.json` written with all fields of [05](05_ML_DESIGN.md) §9. | M3 |
| FR-010 | `GET /health`. | 200 with `model_loaded: true` when artifact loaded; 503 with error envelope otherwise. | M4 |
| FR-011 | `POST /predict`. | Valid request ([06](06_API_SPECIFICATION.md)) returns 200 with `predicted_rul`, `health_band`, `model_version`; value equals offline inference on the same input (contract test). | M4 |
| FR-012 | `GET /model/info`. | Returns metadata fields listed in 06 §4.3, read from `model_metadata.json`. | M4 |
| FR-013 | Structured errors. | Every 4xx/5xx response uses the envelope `{"error": {"code", "message", "details"}}`; each error code in 06 §6 has a test. | M4 |
| FR-014 | Dashboard shows model information. | Dashboard renders model version, type, and test metrics fetched from `GET /model/info`. | M5 |
| FR-015 | Dashboard requests and displays predictions. | User selects an FD001 test unit (from `DATA_DIR`) or uploads a file in the raw FD format; dashboard sends the history to `POST /predict` and shows predicted RUL, health band, and sensor trend charts. No prediction is computed or stored in the dashboard. | M5 |
| FR-016 | Dashboard handles API failure. | With the API stopped, the dashboard shows an error message naming `API_BASE_URL` and does not crash; API 4xx error messages are shown verbatim. | M5 |
| FR-017 | Containerised deployment. | `docker compose up --build` starts `api` (port 8000) and `dashboard` (port 8501); `api` healthcheck reports healthy when the model is mounted; smoke test passes. | M6 |

## Non-functional requirements

| ID | Requirement | Acceptance criterion | Milestone |
|---|---|---|---|
| NFR-001 | Reproducibility | Two consecutive `python scripts/train.py` runs on the same data produce identical test MAE/RMSE (to 1e-9). All seeds come from config (`random_seed=42`). | M3 |
| NFR-002 | CPU-only, laptop-scale | Full training (CV for both models + final fit + test evaluation) completes in < 5 minutes on the development machine (Apple Silicon, CPU); measured time recorded in [14](14_PROJECT_STATUS.md). | M3 |
| NFR-003 | Inference latency | p95 latency of `POST /predict` with 30 cycles < 200 ms over 100 sequential local calls (TestClient measurement command in 08 M4). | M4 |
| NFR-004 | Code quality | `ruff check .` and `ruff format --check .` pass; public functions have type hints and docstrings. Enforced in CI from M1. | M1 |
| NFR-005 | Test coverage | Line coverage ≥ 80 % for `src/predmaint`, excluding `dashboard/app.py` (Streamlit UI layer). | M7 (tracked from M3) |
| NFR-006 | Configuration & secrets | All runtime config via environment variables documented in `.env.example`; no secrets or absolute paths in code. | M1 |
| NFR-007 | Data integrity | Training refuses to run if FD001 checksums do not match [04](04_DATASET_SPECIFICATION.md) §3. | M3 |
| NFR-008 | Pinned dependencies | All runtime and dev dependencies pinned to the versions in [03](03_TECH_STACK.md); `pip check` passes in a fresh venv. | M1 |
| NFR-009 | Documentation fidelity | Every command in README and docs 08/10 executed successfully in M7; results recorded in 14. | M7 |
| NFR-010 | Artifact safety | API loads artifacts only from the configured local `MODEL_DIR` (joblib = pickle; never from user input or network). | M4 |
