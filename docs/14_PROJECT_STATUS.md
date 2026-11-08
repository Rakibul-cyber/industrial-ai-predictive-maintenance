# 14 — Project Status

| Field | Value |
|---|---|
| Current milestone | **MVP COMPLETE** (M0–M7) + M8 (prediction intervals, committed `a25df0e`) + **Milestone 9 (minimal anomaly detection, CR-010) COMPLETE** — uncommitted, awaiting review. FR-019 trend target **not met** (0.680 ≤ 0.7). |
| Last updated | 2026-11-08 |
| Git ref | `main` = `origin/main` @ `a25df0e` (M8, committed and pushed by the user; CI run 42514386686 green: lint-test + docker). M9 work is uncommitted. |
| Next approved action | User review and commit of M9 — commit the two new model files together with the four existing ones (10 §6); the push runs the CI `docker` job with the anomaly assertions. Rebuild any running demo stack with `--build`. Commit `docs/img/` with it (screenshots added, below). Further work goes through 15. |

## Milestone tracker

| Milestone | Status | Evidence |
|---|---|---|
| M0 Planning | **Approved** 2026-11-08 (architecture frozen; ADR-001…021 Accepted, ADR-022 added) | M0 section below |
| M1 Scaffold, CI, data | **Accepted** 2026-11-08; committed `4e8c04c`, pushed; CI green | M1 section + CI section below |
| M2 Labels, features, baseline (re-scoped, CR-002) | **Accepted** 2026-11-08; committed `f4c86df`, pushed; CI green | M2 section below |
| M3 Primary model, selection, inference contract (CR-003) | **Accepted** 2026-11-08; committed `65db98d`, pushed; CI green (v7 actions, ubuntu-24.04) | M3 section below |
| M4 API (CR-004) | **Accepted** 2026-11-08; committed `7d7de29`, pushed; CI green | M4 section below |
| M5 Dashboard (CR-005) | **Accepted** 2026-11-08; committed `6b4da81`, pushed; CI green | M5 section below |
| M6 Docker, tests, CI (ADR-030–032) | **Accepted** 2026-11-08; committed `ccf68a7`, pushed; CI green incl. `docker` job (amd64) | M6 section below |
| M7 Finalisation | **COMPLETE** 2026-11-08; committed `32fdae0` | M7 section below |
| M8 Prediction intervals (CR-008, ADR-034) | **COMPLETE** 2026-11-08; committed `a25df0e`, pushed; CI run 42514386686 green | M8 section below |
| M9 Minimal anomaly detection (CR-010, ADR-035) | **COMPLETE** 2026-11-08 (uncommitted, awaiting review); FR-019 trend target **not met** | M9 section below |

## M0 — what was verified (actual command results)

**Repository (start of session):** empty working tree except `.git`; `git status` → "No commits yet", branch `main`; no README/CLAUDE.md/docs/config existed, so nothing was overwritten.

**Environment:**

- macOS 27.0.1 (arm64); `python3 --version` → 3.13.3 (`python` not on PATH); pip 25.3; Docker 29.6.2 (daemon up, linux/aarch64), Compose v5.3.1; git 2.54.0.
- `python:3.13-slim` → Python 3.13.18, Debian 13.7, glibc 2.41.

**Packages (throwaway `.venv-verify/`, deleted afterwards):**

- Installed together + imported + `pip check` clean: pandas 3.1.1, numpy 2.5.4, scikit-learn 1.9.1, fastapi 0.162.2 (starlette 1.10.0), uvicorn 0.54.0, pydantic 2.14.3, streamlit 1.67.0, joblib 1.6.0, requests 2.34.2, pytest 9.2.0, pytest-cov 7.1.0, httpx2 2.16.0, ruff 0.17.4.
- Linux cp313 wheels downloaded for x86_64 and aarch64 (45 files each). `manylinux2014`-only tag query failed for pandas 3.1.1 (wheels are `manylinux_2_24/2_28`); the `manylinux_2_28` query succeeded.
- lightgbm 4.7.0: installs, **import fails** (`libomp.dylib` not found) → rejected.
- mlflow 3.18.0: dry-run shows 40+ added packages → not adopted.
- httpx 0.28.1 + TestClient → `StarletteDeprecationWarning` recommending httpx2; httpx2 verified as replacement.

**Dataset:**

- S3 archive URL → HTTP 200, 12,429,152 bytes; downloaded to `data/raw/` (gitignored); nested zip extracted to `data/raw/CMAPSSData/` (14 files). Checksums recorded in 04 §3.
- NASA PCoE page → 200, links the S3 archive. data.nasa.gov catalog → 200, "License not specified"; its `CMAPSSData.zip` link → 403; old Socrata link → 404.
- FD001: train 20631×26, test 13096×26, RUL 100×1; 0 nulls; 100 units each (ids 1–100); cycles start at 1 and step +1; test cycles/unit min 31; RUL range 7–145. Naive `sep=" "` parse gives 28 columns (two trailing spaces) — `\s+` gives 26.
- Paper (bundled PDF) read: NASA score eq. 11 coefficient pairing contradicts its text → ADR-009.

## Open items / blockers

| # | Item | Type | Tracked in |
|---|---|---|---|
| U1 | Dataset licence — **resolved in M1:** no licence stated on NASA PCoE or data.nasa.gov; PCoE requests acknowledgement + gives a dataset citation (04 §11). No redistribution. | Closed (accepted limitation) | 04 §11, 12 R1 |
| U2 | Third-party mirror availability — **UNVERIFIED** (only needed if S3 fails; S3 worked in every run incl. the M7 fresh clone) | Accepted limitation | 12 R1 |
| U3 | Near-constant sensor list — **resolved in M2**: ≤ 2-unique rule verified on real data (04 §5a) | Closed | 04 §5a, 12 R12 |
| U4 | NASA score official implementation — **UNVERIFIED** (interpretation fixed by ADR-009; secondary metric only) | Accepted limitation | 12 R5 |
| U5 | GitHub Actions runner resolves Python 3.13 — **resolved**: run 42491244396 on `4e8c04c` succeeded, `actions/setup-python@v5` step with `python-version: "3.13"` passed. Exact patch version not visible (logs need auth). | Closed | CI section, 12 R11 |
| U6 | Dockerfile/compose behaviour — **resolved in M6** on arm64 (build, healthy services, smoke test) | Closed | 08 M6, 12 R8 |
| U7 | amd64 image build — **resolved in M7:** CI `docker` job green on `ubuntu-24.04` (x64), run 42509206610 | Closed | 12 R10 |
| U8 | Column → physical sensor name mapping — **UNVERIFIED**, not used (ADR-020) | Accepted limitation | 12 R2 |

## M1 — results (2026-11-08, actual command output, trimmed)

**Doc amendments (architecture freeze):** ADR-001…021 → Accepted; ADR-022 added (503 without artifact; demo = train command + `docker compose up`); 06 (503 rule, < `WINDOW_SIZE` → 422), 10 §5, 11 (AC-21, AC-22), 05 (single config module for `WINDOW_SIZE`/`RUL_CAP`; NASA score never used for selection), 07, 08 (M6-T4 artifact-commit decision). `rul_clip` renamed `rul_cap` across docs. Committed as `71295ba`.

**Files created (08 M1 list, nothing else):** `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `.env.example`, `.github/workflows/ci.yml`, `models/.gitkeep`, `src/predmaint/{__init__,config}.py`, `src/predmaint/data/{__init__,download,load,validate}.py`, `scripts/download_data.py`, `tests/{conftest,test_download,test_load,test_validate,test_real_data}.py`. Modified: README (Quickstart), docs 03, 04, 08, 09, 12, 14, 15.

**M1-T9 fresh venv:** `pip install -r requirements-dev.txt && pip install -e .` → all 13 pins installed exactly as in 03; `pip check` → "No broken requirements found."

**Verification:**

```text
$ ruff check .                      → All checks passed!
$ ruff format --check .             → 30 files already formatted
$ rm -rf data/raw && python scripts/download_data.py
Downloaded, verified and extracted the archive.
  verified data/raw/CMAPSSData/train_FD001.txt  (+ test_FD001.txt, RUL_FD001.txt)
$ python scripts/download_data.py
Dataset already present and checksums match; nothing downloaded.
$ shasum -a 256 data/raw/CMAPSSData/{train,test,RUL}_FD001.txt
963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8  data/raw/CMAPSSData/train_FD001.txt
3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851  data/raw/CMAPSSData/test_FD001.txt
a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca  data/raw/CMAPSSData/RUL_FD001.txt
(all three match 04 §3)
$ python -m predmaint.data.validate
FD001 validation passed (data/raw/CMAPSSData)
  train: 20631 rows x 26 cols, 100 units
  test:  13096 rows x 26 cols, 100 units, cycles/unit min 31
  RUL:   100 values, range 7-145
$ pytest -q                          → 48 passed, 1 skipped (network) in 0.89s
$ RUN_NETWORK_TESTS=1 pytest -q -m network → 1 passed in 2.55s
$ DATA_DIR=/nonexistent pytest -q --cov=predmaint   (CI-equivalent, no dataset)
  → 45 passed, 4 skipped (3 data + 1 network); coverage 93 % total
$ pip check                          → No broken requirements found.
```

**Decisions / findings in M1:**

- **CR-001 (defect fix):** stdlib `urllib` failed TLS verification (`CERTIFICATE_VERIFY_FAILED`; python.org macOS build has no default CA file). Download switched to the already-pinned `requests` (certifi bundle). Details in 15.
- CI does **not** download the dataset; fixture tests only (ADR-021 consequence recorded in 13).
- `filterwarnings = error::DeprecationWarning` is enabled globally; the suite is clean under it.
- Real-data tests ran locally (not skipped) — required for M1 sign-off.
- Download tests use a local HTTP server; the server poll interval was lowered to 0.05 s (suite time 13 s → 0.7 s).

**Not done / pending:** CI has never executed (no push authorized) → U5 stays open.

## CI — first run (2026-11-08)

- Run [42491244396](https://github.com/Rakibul-cyber/industrial-ai-predictive-maintenance/actions/runs/42491244396), workflow `CI`, commit `4e8c04c`, event `push`: **conclusion `success`**. Job `lint-test` on `ubuntu-latest`, 14:44:03Z–14:44:44Z. All steps succeeded: checkout, setup-python (3.13), install dependencies, ruff lint, ruff format check, tests. Queried via the public GitHub REST API (`gh` CLI is not installed; raw logs return 403 without authentication).
- No `ci:` fix was needed, so no extra commit or push was made.
- Annotations (non-failing, follow-ups): (1) warning — "Node.js 20 is deprecated … actions/checkout@v4, actions/setup-python@v5 … forced to run on Node.js 24"; (2) notice — `ubuntu-latest` migrates to Ubuntu 26 from 2026-10-19. Bumping action versions needs a verified newer major version and an authorized commit.

## M2 — results (2026-11-08, actual command output, trimmed)

**Scope:** re-scoped by user request (CR-002 / ADR-023): labels, features, Ridge baseline training, CV, official-test evaluation, artifacts, and committed report. Warnings policy added to CLAUDE.md and 09.

**Files created:** `src/predmaint/data/labels.py`, `src/predmaint/features/{__init__,build}.py`, `src/predmaint/models/{__init__,registry,metrics,train,artifacts,predict}.py`, `scripts/train.py`, `reports/metrics_baseline.json`, `tests/{test_labels,test_features,test_metrics,test_train}.py`. Modified: `src/predmaint/data/validate.py` (`expected=None`), `tests/conftest.py`, `tests/test_real_data.py`, `.gitignore`, `README.md`, `CLAUDE.md`, docs 01, 04, 05, 07, 08, 09, 12, 13, 14, 15. Generated, gitignored: `models/baseline.joblib`, `models/baseline_metadata.json`.

```text
$ ruff check .            → All checks passed!
$ ruff format --check .   → 44 files already formatted
$ pytest -q               → 83 passed, 1 skipped (network) in 2.43s
$ python scripts/train.py --model baseline
Sensor selection on train (drop if <= 2 unique values): dropped ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
Kept 14 sensors: ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
Features: 43 (cycle + 3 x 14), window_size=30, rul_cap=125
Training rows: 17731 of 20631 (rows without a full 30-cycle window dropped), 100 units
  CV MAE  = 15.504 ± 1.052
  CV RMSE = 19.287 ± 1.181
Official test set (100 units, last observed cycle, unclipped truth): MAE=15.403 RMSE=19.205 NASA score=595.2
  (truth clipped at 125: MAE=14.333 RMSE=17.910)
Saved models/baseline.joblib, models/baseline_metadata.json, reports/metrics_baseline.json
$ second training run → metrics identical: True
$ pytest -q --cov=predmaint → 83 passed, 1 skipped; TOTAL 93 %
```

**Findings:** the test RMSE of 19.2 lies inside the 15–30 sanity band and CV ≈ test, so no leakage signal was found and nothing was tuned. The rolling windows are computed independently per window, so feature rows are bit-identical regardless of surrounding history (tests for truncation and trailing-window-only invariance). The baseline artifact is 5.3 KB, which matters for M6-T4.

## M3 — results (2026-11-08, actual command output, trimmed)

**Amendments recorded:** CR-003; ADR-024 (CV-RMSE selection, supersedes ADR-010), ADR-025 (bands 20/60, supersedes ADR-017), ADR-026 (one bounded 12-config search, amends ADR-007), ADR-027 (clamp to [0, RUL_CAP], supersedes ADR-018).

**CI housekeeping (no ADR):** `actions/checkout@v4 → @v7` and `actions/setup-python@v5 → @v7`. The GitHub API showed latest releases v7.0.1 (2026-07-20) and v7.1.0 (2026-10-22); both `action.yml` files declare `using: node24`, and `setup-python@v7` still accepts `python-version`, `cache` and `cache-dependency-path`. `runs-on: ubuntu-latest → ubuntu-24.04`. The YAML parsed with Ruby's `YAML.safe_load` (runs-on and both `uses:` confirmed). It is first exercised on the next push. CI on the M2 commit `f4c86df` (run 42492050196) was green.

**Files created:** `src/predmaint/models/selection.py`, `tests/test_selection.py`, `tests/test_inference_contract.py`, `reports/metrics_primary.json`, `reports/model_comparison.json`, `reports/model_comparison.md`. **Modified:** `config.py`, `models/{registry,train,predict}.py`, `scripts/train.py`, `tests/{conftest,test_train,test_real_data}.py`, `.github/workflows/ci.yml`, `reports/metrics_baseline.json` (re-measured), README, CLAUDE.md, docs 00, 01, 02, 04, 05, 06, 07, 08, 09, 11, 12, 13, 14, 15. Generated, gitignored: `models/{primary,model}.joblib`, `models/{baseline,primary}_metadata.json`, `models/model_metadata.json`.

```text
$ ruff check .            → All checks passed!
$ ruff format --check .   → 48 files already formatted
$ pytest -q               → 120 passed, 1 skipped (network) in 5.94s
$ time python scripts/train.py --model primary
Bounded search (ADR-026): 12 configurations, scored by 5-fold GroupKFold CV RMSE
  [ 5/12] {'learning_rate': 0.05, 'max_iter': 200, 'max_leaf_nodes': 15, 'l2_regularization': 0.0}  CV RMSE 16.559 ± 0.695  MAE 11.843
  (other 11 configurations: CV RMSE 16.629–16.828; full table in 05 §7b)
Selected configuration: {'learning_rate': 0.05, 'max_iter': 200, 'max_leaf_nodes': 15, 'l2_regularization': 0.0}
  CV MAE  = 11.843 ± 0.676
  CV RMSE = 16.559 ± 0.695
Official test set (100 units, last observed cycle, unclipped truth): MAE=12.567 RMSE=16.610 NASA score=441.5
  (truth clipped at 125: MAE=11.497 RMSE=15.333)
Elapsed: 78.8 s      [time: 434.19s user 148.24s system 726% cpu 1:20.17 total]
$ python scripts/train.py --select
| baseline | `ridge-20261108T150138Z` | 19.152 ± 1.182 | 15.112 ± 1.156 | 19.144 | 15.347 | 590.6 | 17.845 |
| primary (selected) | `hgb-20261108T150258Z` | 16.559 ± 0.695 | 11.843 ± 0.676 | 16.610 | 12.567 | 441.5 | 15.333 |
Selected primary because its 5-fold GroupKFold CV RMSE (16.559) is lower than baseline's (19.152) (ADR-024). … The official test ranking agrees with the CV ranking.
$ (repeat primary run) → metrics identical: True | search identical: True; primary.joblib byte-identical; then --select again
  → served model_version primary-20261108-f4c86df (candidate hgb-20261108T150441Z)
$ pytest -q -m data -v    → 5 passed (incl. promoted-model test RMSE recomputed via predict_rul == metadata)
$ pytest -q --cov=predmaint → 120 passed, 1 skipped; TOTAL 94 %
$ ls -l models/*.joblib
   5297  baseline.joblib
 451385  model.joblib      (byte copy of primary.joblib)
 451385  primary.joblib
```

**Findings:**

- The baseline was re-measured because the shared clamp gained the `RUL_CAP` upper bound (ADR-027). Test RMSE moved 19.205 → 19.144 and CV RMSE 19.287 → 19.152; one test prediction was 130.58. The M2 numbers are kept in 05 §7a for transparency.
- The primary beats the baseline on CV (16.559 vs 19.152) and on test (16.610 vs 19.144); the CV and test rankings agree. The 12 configurations differ by < 0.3 CV RMSE, so there is no sign that a further search would change the conclusion. No second round was run.
- **Runtime:** wall time is well under 5 minutes (≈ 80 s). Total CPU time (user + sys) was ≈ 582 s ≈ 9.7 CPU-minutes, because HGB uses all cores. NFR-002 is defined as wall time, so it is met; if "5 minutes CPU" was meant as CPU-seconds, it is exceeded (single-threaded wall time is unmeasured).
- **Artifact size for M6-T4:** the served `model.joblib` is 451,385 bytes (≈ 441 KiB) and the baseline 5,297 bytes. Committing the served model is size-viable (well under typical repo limits, no Git LFS needed). The remaining M6 considerations are sklearn-version coupling of the pickle and keeping the committed file in sync with `model_metadata.json`.
- The inference contract checks cycle order **as given** (out-of-order input raises rather than being sorted); doc 06 §3.3 is updated accordingly.

## CI — after the M3 push (2026-11-08)

Run [42504678999](https://github.com/Rakibul-cyber/industrial-ai-predictive-maintenance/actions/runs/42504678999) on `65db98d`: **success**. Job `lint-test` ran on label `ubuntu-24.04` from 17:22:53Z to 17:23:28Z. Steps `Run actions/checkout@v7` and `Run actions/setup-python@v7`, install, ruff lint, ruff format check and tests all succeeded. **0 annotations**, so the Node 20 deprecation warning seen on v4/v5 is gone. The CI housekeeping is verified.

## M4 — results (2026-11-08, actual command output, trimmed)

**Decisions:** CR-004; ADR-028 (`/health` always 200, readiness via `model_loaded`; amends ADR-022); ADR-029 (commit FD001 test engine 1 as demo payloads; exception to ADR-003). The request keeps 06's field names `unit_id`/`records` (06 is explicit, so `engine_id` was not adopted). New env vars `MODEL_PATH`, `API_HOST`, `API_PORT`. No new dependencies.

**Housekeeping:** 05 §7a now states the training cost next to NFR-002: 80 s wall, ≈ 582 CPU-s (≈ 9.7 CPU-min) on an 8-core Apple M1 (`sysctl hw.ncpu` = 8, 4P + 4E), with an explicitly unmeasured estimate for smaller machines.

**Files created:** `src/predmaint/api/{__init__,main,schemas,service,errors}.py`, `scripts/make_request_payload.py`, `examples/{predict_request,predict_request_min,expected_predictions}.json`, `tests/test_api.py`. **Modified:** `src/predmaint/config.py`, `src/predmaint/models/predict.py`, `tests/test_real_data.py`, `.env.example`, README, docs 01, 02, 04, 05, 06, 07, 08, 09, 10, 11, 13, 14, 15.

```text
$ ruff check .            → All checks passed!
$ ruff format --check .   → 55 files already formatted
$ pytest -q               → 152 passed, 1 skipped (network) in 7.41s
$ pytest -q --cov=predmaint → api/errors 97 %, api/main 91 %, api/schemas 100 %, api/service 96 %; TOTAL 94 %
$ DATA_DIR=/nonexistent MODEL_DIR=/nonexistent pytest -q → 145 passed, 8 skipped (CI-equivalent)
$ python -m predmaint.api.main     (log: "Loaded model primary-20261108-f4c86df from models/model.joblib")
$ curl -s localhost:8000/health
{"status":"ok","model_loaded":true,"model_version":"primary-20261108-f4c86df","api_version":"0.1.0"}
$ curl -s localhost:8000/model/info   → model_name primary, selected_by cv_rmse, n_features 43, window_size 30, rul_cap 125,
  test rmse 16.609852374016835, cv rmse_mean 16.558936995946503, sklearn_version_match true (full body in 06 §4.1)
$ curl -s -X POST localhost:8000/predict -d @examples/predict_request.json
{"unit_id":"test-unit-1","predicted_rul":125.0,"health_status":"healthy","cycles_received":31,"cycles_used":30,"last_cycle":31,"model_version":"primary-20261108-f4c86df","warnings":[]}
$ curl -s -X POST localhost:8000/predict -d @examples/predict_request_min.json
{"unit_id":"test-unit-1","predicted_rul":125.0,"health_status":"healthy","cycles_received":30,"cycles_used":30,"last_cycle":31,"model_version":"primary-20261108-f4c86df","warnings":[]}
$ (bad) first 10 cycles → {"error":{"code":"INSUFFICIENT_CYCLES","message":"10 cycles received; at least 30 are required","details":[]}}  HTTP 422
server log: "GET /health 200 2.2ms", "GET /model/info 200 0.8ms", "POST /predict 200 35.3ms", "POST /predict 200 11.8ms", "POST /predict 422 1.3ms"; server stopped (port 8000 free afterwards)
$ MODEL_PATH=/nonexistent python -m predmaint.api.main
  log: WARNING predmaint.api: Model not available, serving in degraded mode: artifact not found: /nonexistent/model.joblib …
$ curl -s localhost:8000/health → {"status":"degraded","model_loaded":false,"model_version":null,"api_version":"0.1.0"}  HTTP 200
$ curl -s -X POST localhost:8000/predict -d @examples/predict_request_min.json
{"error":{"code":"MODEL_UNAVAILABLE","message":"No model loaded from /nonexistent: artifact not found: /nonexistent/model.joblib (run `python scripts/train.py`)","details":[]}}  HTTP 503
$ curl -s localhost:8000/model/info → same MODEL_UNAVAILABLE envelope, HTTP 503; server stopped
$ NFR-003: /predict, 30 cycles, 100 in-process calls → p50 10.9 ms, p95 13.6 ms
```

**Findings:**

- Engine 1 (31 cycles, true RUL 112) predicts 125.0, the cap. It is early-life, so the demo is honest but shows the clamp. Both example files give the same prediction, as expected, because only the last 30 cycles are used.
- Fixed during M4: `/model/info` returned `health_bands` as floats (`20.0`) because of a `dict[str, float]` schema; changed to `int`.
- Found during M4: nothing reads `.env` (no python-dotenv). README/10 told users to `cp .env.example .env` as if it took effect. Now documented as `set -a; . ./.env; set +a`.
- The planned M6 compose healthcheck must read `model_loaded` (ADR-028); doc 10 §4 is updated.

## CI — after the M4 push (2026-11-08)

Run [42505948636](https://github.com/Rakibul-cyber/industrial-ai-predictive-maintenance/actions/runs/42505948636) on `7d7de29`: **success**. Job `lint-test` ran on `ubuntu-24.04` from 17:38:16Z to 17:39:04Z; all steps succeeded; 0 annotations.

## M5 — results (2026-11-08, actual command output, trimmed)

**Pre-work:** `examples/predict_request_late.json` = FD001 test engine **34**, the only test engine with the minimum true RUL (7), all 203 cycles, generated with `scripts/make_request_payload.py --unit 34 --out … --expected …`. Sidecar entry: `predicted_rul 6.3`, `critical`, true RUL 7. The existing sidecar entries are unchanged.

**Decisions:** CR-005 (requested `api_client.py` returning typed results instead of 08's planned raising `client.py`; display helpers in `helpers.py`). No new dependency: charts use Streamlit's built-in `st.vega_lite_chart` (cycle marker and band thresholds as rule layers), so no Altair or Plotly import. The documented run command adds `--server.address localhost`, because Streamlit's default listens on all interfaces (observed: the log listed a Network URL and an External URL).

**Files created:** `src/predmaint/dashboard/{__init__,api_client,helpers,app}.py`, `examples/predict_request_late.json`, `tests/test_dashboard_{client,helpers,app}.py`. **Modified:** `examples/expected_predictions.json`, `tests/test_real_data.py`, `.env.example`, README, docs 02, 06, 07, 08, 09, 10, 14, 15.

```text
$ ruff check .            → All checks passed!
$ ruff format --check .   → 62 files already formatted
$ pytest -q               → 181 passed, 1 skipped (network) in 8.18s   (DeprecationWarning still = error; Streamlit needed no filter)
$ pytest -q --cov=predmaint → dashboard/api_client 100 %, dashboard/helpers 100 % (app.py omitted per NFR-005); TOTAL 95 %
$ DATA_DIR=/nonexistent MODEL_DIR=/nonexistent pytest -q → 170 passed, 12 skipped (CI-equivalent)
$ python -m predmaint.api.main &   ;   streamlit run src/predmaint/dashboard/app.py --server.headless true --server.port 8501 &
  dashboard: "Uvicorn server started on :::8501"; GET /_stcore/health → ok; GET / → HTTP 200
```

**How the live check was driven — read this:** a headless Chrome load of the page (`--dump-dom`, with `--virtual-time-budget` and with `--timeout`) returned only the page shell; Streamlit's websocket session never ran, so the browser produced **no** API calls. Instead, the real `src/predmaint/dashboard/app.py` was driven with Streamlit's `AppTest` against the **live** API (no mocks: real `requests` calls to 127.0.0.1:8000), selecting engines, moving the slider and clicking "Compute RUL trajectory". No interactive browser session was performed (screenshots are an M7 task).

```text
AppTest session → status: API reachable yes · model loaded yes · primary-20261108-f4c86df
test 34 @ N=203:  predicted 6.3   | true 7
train 1 @ N=30:   predicted 118.2 | true 162 (capped target 125)
train 1 @ N=100:  predicted 93.0  | true 92
train 1 @ N=192:  predicted 0.9   | true 0
trajectory train 1: 34 rows (cycles 30..192)
 cycle predicted_rul health_status true_rul true_rul_capped
    30        118.25       healthy      162             125
    55        125.00       healthy      137             125
    80        104.56       healthy      112             112
   105         99.08       healthy       87              87
   130         76.36       healthy       62              62
   155         30.62       warning       37              37
   185          3.88      critical        7               7
   190          2.56      critical        2               2
   192          0.90      critical        0               0
API server log for that session: 44 POST /predict, 10 GET /health, 10 GET /model/info, e.g.
  18:46:12,007 GET /health 200 10.1ms · 18:46:12,010 GET /model/info 200 1.2ms · 18:46:12,180 POST /predict 200 103.6ms
  … trajectory burst … 18:46:13,046 POST /predict 200 12.9ms · 18:46:13,065 POST /predict 200 15.2ms · 18:46:13,103 POST /predict 200 14.2ms
Both servers stopped afterwards (health checks on :8000 and :8501 fail).
```

**Findings:**

- **Train engine 1 is training data**, so its close numbers (93.0 vs 92 at N=100; 0.9 vs 0 at the end) are in-sample and optimistic. Test engine 34 (6.3 vs 7) is the honest out-of-sample example. The trajectory shape is as expected: it starts near the cap, descends, and crosses warning (< 60) and critical (< 20) near the end.
- Health status: the API returned `healthy` at train-1 N = 30 and `critical` at N = 185, 190, 192 (trajectory rows above), and `critical` for test 34 (sidecar, reproduced through the API in `test_real_data`). For N = 100 the status was not read back; under ADR-025, 93.0 ≥ 60 means `healthy`.
- In AppTest, `st.badge` is not exposed as a typed element, so badge text was not read back; status values come from the API responses (trajectory table and sidecar).
- **M6 note:** the dashboard container needs FD001 under `$DATA_DIR/raw/CMAPSSData` (planned: read-only mount of `./data`) for the explorer and trajectory; status, Upload and Model info work without it (10 §4).

## CI — after the M5 push (2026-11-08)

Run [42506959482](https://github.com/Rakibul-cyber/industrial-ai-predictive-maintenance/actions/runs/42506959482) on `6b4da81`: **success** (lint-test job on `ubuntu-24.04`).

## M6 — results (2026-11-08, actual command output, trimmed; dev machine Apple M1, arm64)

**Decisions:** ADR-030 (commit the served model; decides M6-T4), ADR-031 (`./data` bind mount; the dashboard downloads FD001 on first start), ADR-032 (one single-stage image, digest-pinned, model baked in, two compose services; supersedes ADR-015's `./models` mount). Before committing, the local `model.joblib` was confirmed to be the artifact described by the already-committed `model_metadata.json` (candidate `hgb-20261108T150441Z`, byte copy of `primary.joblib`, SHA-256 `39bfa6fd…28c1`). No new Python dependency.

**Files created:** `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `scripts/smoke_test.sh`, `tests/test_model_artifact.py`, `tests/test_docker_smoke.py`. **Committed artifact (newly tracked):** `models/model.joblib`. **Modified:** `.gitignore`, `pyproject.toml`, `.github/workflows/ci.yml`, `.env.example`, README, CLAUDE.md, docs 01, 02, 03, 07, 08, 09, 10, 11, 12, 13, 14.

```text
$ ruff check .            → All checks passed!
$ ruff format --check .   → 64 files already formatted
$ pytest -q               → Required test coverage of 85% reached. Total coverage: 94.98%
                            187 passed, 2 skipped (network, docker) in 11.14s
$ pytest -q -m data --no-cov → 11 passed  ·  CI-equivalent (no data/model dirs) → 176 passed, 13 skipped, gate reached
$ docker build -t predmaint:latest .   → exit 0, 57 s cold (pip --only-binary=:all:; no compiler in image)
$ docker images predmaint → predmaint latest 736171c7b08a 1.1GB   (inspect .Size 242,889,009 bytes; arm64; user app)
$ mv data/raw → scratchpad backup (to show a first-start download); docker compose up -d
 Container predmaint-api-1 Healthy · Container predmaint-dashboard-1 Started
$ docker compose ps
SERVICE     STATUS                    PORTS
api         Up 12 seconds (healthy)   127.0.0.1:8000->8000/tcp
dashboard   Up 6 seconds (healthy)    127.0.0.1:8501->8501/tcp
$ docker compose logs dashboard   (first start)
Downloaded, verified and extracted the archive.
  verified /app/data/raw/CMAPSSData/train_FD001.txt (+ test_FD001.txt, RUL_FD001.txt)
host: data/raw/cmapss_outer.zip 12429152 bytes; train_FD001.txt SHA-256 963b5e22…de0f8 (== 04 §3)
$ curl -s -X POST localhost:8000/predict -d @examples/predict_request.json
{"unit_id":"test-unit-1","predicted_rul":125.0,"health_status":"healthy","cycles_received":31,"cycles_used":30,"last_cycle":31,"model_version":"primary-20261108-f4c86df","warnings":[]}
$ curl -s -X POST localhost:8000/predict -d @examples/predict_request_late.json
{"unit_id":"test-unit-34","predicted_rul":6.3,"health_status":"critical","cycles_received":203,"cycles_used":30,"last_cycle":203,"model_version":"primary-20261108-f4c86df","warnings":[]}
sidecar comparison: predict_request / _min / _late → MATCH, MATCH, MATCH
$ docker compose restart dashboard → logs: "Dataset already present and checksums match; nothing downloaded."
$ docker compose exec dashboard … ApiClient('http://api:8000').health() → 200 {... "model_loaded": True ...}; id → uid=1000(app)
$ docker compose down → containers + network removed
(new download's FD001 checksums == backup; backup deleted)
$ bash scripts/smoke_test.sh; echo $?
   api=healthy dashboard=starting … api=healthy dashboard=healthy
== GET /health → {"status":"ok","model_loaded":true,"model_version":"primary-20261108-f4c86df","api_version":"0.1.0"}
== POST /predict examples/predict_request_late.json → {"unit_id":"test-unit-34","predicted_rul":6.3,"health_status":"critical",…}
== GET dashboard /_stcore/health → ok
SMOKE TEST PASSED (arm64)
exit code: 0   (leftover containers: 0)
$ SMOKE_TIMEOUT_S=1 bash scripts/smoke_test.sh → "SMOKE TEST FAILED: timed out waiting for healthy services", exit code 1, leftover containers 0
$ RUN_DOCKER_TESTS=1 pytest -q -m docker --no-cov → 1 passed in 16.90s
$ docker run -e MODEL_PATH=/nonexistent/model.joblib … (AC-21 in a container)
  /health → {"status":"degraded","model_loaded":false,…} HTTP 200 · /predict → MODEL_UNAVAILABLE HTTP 503 · /model/info → HTTP 503
  compose healthcheck command → exit 1 (container would be unhealthy, as intended)
$ pip check → No broken requirements found.   $ git ls-files data/ → data/.gitkeep
```

**Findings:**

- **arm64: verified** end to end (build, healthy compose, containerised predictions, first-start download/skip, smoke test pass and fail paths). **amd64: not yet verified.** It is confirmed only by the CI `docker` job after the user's push; record the run URL and result here then. Whether the GitHub runner provides `docker compose` v2 is UNVERIFIED until that run.
- A first AC-21 attempt showed `HTTP 000`. This was my polling, not the app: Docker's port proxy accepted the TCP connection before uvicorn was listening, so curl got an empty reply and `--retry-connrefused` did not retry. Rerun with a proper wait loop: ready after ~3 s, behaviour as specified.
- Linux bind-mount permissions are handled by running the dashboard as `APP_UID:APP_GID` (the smoke test exports the host IDs) with `HOME=/tmp`. This was exercised on macOS only; the CI Linux job is the first real test of it.
- Coverage gate set at 85 % (stricter than NFR-005's 80 % floor); subsets need `--no-cov`.
- README now has the real CI badge (verified: badge URL → HTTP 200 `image/svg+xml`, title "CI - passing") and Quickstart Option A (Docker) / Option B (local Python). A stale README sentence ("data is not redistributed") was corrected to mention the ADR-029 excerpt.
- 11: 16 criteria met, 3 partly met (AC-11 screenshot, AC-12 CI with the docker job, AC-22 fresh-clone run), 3 open for M7 (AC-04 README results table, AC-17 docs-match-code, AC-18 final defect review).

## M7 — results (2026-11-08)

### Step 1 — UNVERIFIED items, from CI run 42509206610 on `ccf68a7`

Both jobs succeeded with 0 annotations: `lint-test` 18:17:00–18:17:50Z (incl. "Tests (coverage gate 85 %)"), `docker` 18:17:52–18:18:38Z ("Docker smoke test (build, compose up, healthchecks, /predict, teardown)").

| Item | Outcome |
|---|---|
| amd64 image build | **Confirmed.** The `docker` job built and smoke-tested the image on GitHub's `ubuntu-24.04` hosted runner (x64). The script's arch line is not readable: job logs return HTTP 403 without authentication. |
| `docker compose` v2 on the runner | **Confirmed.** `scripts/smoke_test.sh` calls `docker compose` under `set -euo pipefail` and the step succeeded. |
| Linux bind-mount write path | **Not confirmed.** By design a failed FD001 download only logs a warning and the dashboard still becomes healthy (ADR-031), so a green smoke test does not prove the write worked, and the log line that would show it is not readable without authentication. Listed as a known gap; a smoke-test assertion is a candidate next milestone (CLAUDE.md). |

### Step 2 — fresh clone from GitHub (temporary directory outside the repo, deleted afterwards)

```text
$ git clone https://github.com/Rakibul-cyber/industrial-ai-predictive-maintenance.git   → HEAD ccf68a7; models/ = model.joblib, model_metadata.json; data/ empty
Option A (README, literally):
$ export APP_UID=$(id -u) APP_GID=$(id -g); docker compose up -d --build → exit 0, "Image predmaint:latest Built", api Healthy
$ docker compose ps
api         Up 12 seconds (healthy)   127.0.0.1:8000->8000/tcp
dashboard   Up 6 seconds (healthy)    127.0.0.1:8501->8501/tcp
$ curl -s -X POST localhost:8000/predict … @examples/predict_request_late.json
{"unit_id":"test-unit-34","predicted_rul":6.3,"health_status":"critical","cycles_received":203,"cycles_used":30,"last_cycle":203,"model_version":"primary-20261108-f4c86df","warnings":[]}
dashboard log: "Downloaded, verified and extracted the archive." (clone's ./data owned by host UID 501)
$ docker compose down → network removed
Option B (README as it was, literally — incl. the three train.py commands):
… train --model baseline: test MAE=15.347 RMSE=19.144 · --model primary: selected {lr 0.05, max_iter 200, leaves 15, l2 0}, CV RMSE 16.559 ± 0.695, test MAE=12.567 RMSE=16.610 · --select: primary
retrained model.joblib is BYTE-IDENTICAL to the committed one
$ pytest -q → 3 failed, 184 passed   ← README defect (CR-006): model_version changed to primary-20261108-ccf68a7, sidecar still f4c86df
Fix: README Option B no longer retrains; optional "Retrain" block refreshes the sidecar (make_request_payload.py --expected ×3); doc 10 §6 likewise.
Re-run (clone reset): corrected Option B → ruff clean; pytest -q → "Required test coverage of 85% reached. Total coverage: 94.98%", 187 passed, 2 skipped
                      retrain + sidecar refresh → 187 passed, 2 skipped (only the 3 model_version strings changed)
$ python -m predmaint.api.main; curl -s -X POST localhost:8000/predict … @examples/predict_request_min.json
{"unit_id":"test-unit-1","predicted_rul":125.0,"health_status":"healthy","cycles_received":30,"cycles_used":30,"last_cycle":31,"model_version":"primary-20261108-f4c86df","warnings":[]}
API stopped; 0 predmaint containers running; clone deleted.
```

### Step 3 — documentation vs implementation

| Document | Compared against | Discrepancies found | Resolved |
|---|---|---|---|
| 02 Architecture | code modules, flows, compose | (1) diagram showed a read-only model volume (model is baked in since ADR-032); (2) §1 omitted `selection.py` and `smoke_test.sh`; (3) §4 described one training run, not three commands + 12-config search; (4) §5 said the service checks business rules and builds features from all records (it is `predict_rul`, last 30 records); (5) §7 omitted the pipeline's feature-select step; (6) §8 lacked the container download-failure path | All six fixed in the doc |
| 06 API | live `/openapi.json`, `errors.py` | Endpoints, methods, all request/response fields match. Undocumented fallback error code `HTTP_ERROR` | Added to 06 §6 |
| 07 Structure | `git ls-files` | Matches. `data/processed/` is listed but never created; M7 files missing | Marked "not created"; LICENSE, docs 16–17, `docs/img/`, residuals report added |
| 10 Deployment | `docker compose config`, `Dockerfile` | Ports, user, restart, depends_on, volumes (ro/rw), healthcheck timings and env vars all match. §6 retraining lacked the sidecar refresh (CR-006); §9 amd64 pending | §6 fixed; §9 records the CI result |
| README | fresh clone | Option B retrain caused failing tests (CR-006) | Fixed and re-verified |
| 01 Requirements | tests/commands | Final status column added: 27/27 have a status — 26 Met, FR-015 Met as amended (CR-005), NFR-004 Met with a noted docstring gap | — |
| 13 Decisions | — | 32 ADRs; none left Proposed (all Accepted or Superseded) | — |

### Step 4–6 — deliverables

`docs/16_EVALUATION_REPORT.md` (generated from `reports/*.json` and `models/model_metadata.json`), `reports/residuals_by_rul_bucket.json` (pandas one-off on the committed model; overall MAE 12.567 / RMSE 16.61 == metadata; buckets 0–20: MAE 3.43, signed +1.62; 21–60: 8.82, +5.38; 61–125: 14.60, +1.42; >125: 24.46, −24.46), final `README.md`, `LICENSE` (MIT, 2026, MD Rakibul Hassan), `docs/17_PORTFOLIO_NOTES.md`.

### Final verification

```text
$ ruff check .            → All checks passed!
$ ruff format --check .   → 66 files already formatted
$ pytest -q               → Required test coverage of 85% reached. Total coverage: 94.98%
                            187 passed, 2 skipped (network, docker opt-in) in 10.51s
$ pytest -q -m data --no-cov → 11 passed   ·   pip check → No broken requirements found   ·   placeholder grep → clean
```

### Final defect review — no critical defects

| # | Defect / rough edge | Severity | State |
|---|---|---|---|
| 1 | README Option B retrain broke artifact tests (CR-006) | Medium (docs) | Fixed in M7 |
| 2 | `HTTP_ERROR` fallback code undocumented | Low (docs) | Fixed in M7 |
| 3 | Smoke test does not assert that the FD001 download succeeded (Linux bind-mount write path unverified in CI) | Low | Open, candidate next milestone |
| 4 | Screenshots missing (`docs/img/` absent); README image links broken until added | Low (docs) | **Closed 2026-11-08** (post-M9): four screenshots supplied by the owner and placed in `docs/img/` (see "Screenshots" below) |
| 5 | 16 of 84 public functions lack docstrings (trivial properties, nested helpers, one-line wrappers) | Low | Accepted (no-refactor rule) |
| 6 | No browser-level dashboard test; AppTest cannot read `st.badge` | Low | Accepted |
| 7 | Optimistic bias (+5.4 cycles) for true RUL 21–60; cap saturation for RUL > 125 | Model limitation | Documented (16 §6–7) |
| 8 | Image 1.1 GB; API image carries Streamlit/pyarrow | Low | Accepted (ADR-032) |
| 9 | `.env` not auto-loaded in Option B | Low | Documented (10 §4) |
| 10 | Primary training CPU cost on small machines unmeasured | Low | Documented (05 §7a) |

**Blockers:** none.

## CR-007 — project renamed (2026-11-08, post-MVP change control)

Title changed from "Industrial AI — Predictive Maintenance & Anomaly Detection Platform" to **"Industrial AI — Predictive Maintenance Platform"** (ADR-033). Naming only: no code, API, model, test or dependency change. The FastAPI title "Predictive Maintenance — RUL API", the Streamlit titles and the `pyproject.toml` description never contained the old title or an anomaly claim, so they were left unchanged.

**Before:** 10 `anomaly` occurrences; one title (`docs/00:3`) and one justification of the overstated title (`docs/00:44`). **Edited:** doc 00 (title; line 44 rewritten as "planned extension, NOT IMPLEMENTED and not authorized"), CLAUDE.md (candidate list labelled NOT IMPLEMENTED), doc 13 (ADR-033; ADR-001 naming consequence marked superseded), doc 15 (CR-007), doc 14.

**Regression:** `ruff check .` → All checks passed! · `ruff format --check .` → 66 files already formatted · `pytest -q` → Required test coverage of 85% reached. Total coverage: 94.98% · 187 passed, 2 skipped.

**Final `git grep -n -i "anomaly"`, with each remaining occurrence classified:**

| Location | Classification |
|---|---|
| `CLAUDE.md:52` | Scope restriction: "No anomaly detection … unless approved" |
| `CLAUDE.md:76` | Candidate next milestone, under "NOT IMPLEMENTED, NOT authorized" |
| `README.md:149` | "Candidate next steps, **not implemented**" |
| `docs/00:44` | Out-of-scope list: "planned extension, NOT IMPLEMENTED and not authorized" |
| `docs/12:19` | R13 scope-creep risk example (implies not implemented) |
| `docs/13:39`, `:279–283` | ADR-033 table row and record of the rename |
| `docs/13:45–47` | Historical ADR-001 text (out of scope; ADR-001 naming consequence superseded by ADR-033) |
| `docs/15:112–117` | CR-007 change record quoting the old title as evidence |
| `docs/16:96` | Row in "Optional extensions — NOT IMPLEMENTED" |

No occurrence implies the feature exists. None appear in `src/`, `tests/` or `scripts/`.

## M8 — prediction intervals (2026-11-08)

**Start:** working tree clean; `main` = `origin/main` @ `8d893ff`; CI run [42511719150](https://github.com/Rakibul-cyber/industrial-ai-predictive-maintenance/actions/runs/42511719150) green. **Change control:** CR-008, ADR-034, FR-018, doc 00 scope, M8 plan; CR-009 (smoke-test port collision, below).

**Point model untouched:** SHA-256 of `models/model.joblib` (`39bfa6fd…28c1`) and `models/model_metadata.json` (`57f80600…dcfd`) recorded before M8 and re-checked after training, smoke test and final run: `OK` / `OK`.

```text
$ python scripts/train.py --model intervals
Intervals for served model primary-20261108-f4c86df: quantiles (0.1, 0.9) (nominal 80%), hyperparameters {'learning_rate': 0.05, 'max_iter': 200, 'max_leaf_nodes': 15, 'l2_regularization': 0.0}
5-fold GroupKFold out-of-fold: n=17731 coverage(unclipped)=0.509 coverage(clipped)=0.737 mean width=37.20 median width=36.66 reordered=1023
  bucket     n    cov_unclipped  cov_clipped  mean_width  median_width
  0-20      2100          0.768        0.768       32.13         29.73
  21-60     4000          0.627        0.627       43.91         43.94
  61-125    6392          0.767        0.767       43.63         45.54
  >125      5239          0.000        0.773       26.25         21.94
Official test set (last cycle): n=100 coverage(unclipped)=0.740 coverage(clipped)=0.820 mean width=42.13 median width=41.96 reordered=8
  0-20        16          0.938        0.938       27.72         20.45
  21-60       23          0.826        0.826       49.35         43.94
  61-125      50          0.800        0.800       44.35         49.85
  >125        11          0.000        0.727       37.85         39.73
Timings: CV 14.3 s (3 models x 5 folds), final fit 1.8 s (2 quantile models)
Elapsed: 16.4 s     [time: 83.19s user 32.64s system 630% cpu 18.364 total]
$ API_PORT=8001 python -m predmaint.api.main   (8000 held by the user's demo stack)
  log: Loaded model primary-20261108-f4c86df … / Loaded prediction intervals (80 %) from models/model_intervals.joblib
$ curl /health → {"status":"ok","model_loaded":true,"model_version":"primary-20261108-f4c86df","api_version":"0.2.0","intervals_loaded":true}
$ curl /predict @examples/predict_request_late.json → … "predicted_rul":6.3,"health_status":"critical", … "rul_lower":4.84,"rul_upper":18.76,"interval_level":0.8}
$ curl /predict @examples/predict_request.json (engine 1) → … "predicted_rul":125.0,"health_status":"healthy", … "rul_lower":118.45,"rul_upper":125.0,"interval_level":0.8}
$ bash scripts/smoke_test.sh; echo $?   (1st attempt)
  exit 1 — "Bind for 127.0.0.1:8000 failed: port is already allocated"
$ bash scripts/smoke_test.sh; echo $?   (after CR-009)
  /health → {… "api_version":"0.2.0","intervals_loaded":true}
  /predict late → {… "predicted_rul":6.3,"health_status":"critical", … "rul_lower":4.84,"rul_upper":18.76,"interval_level":0.8}
  SMOKE TEST PASSED (arm64) · exit code: 0
$ ls -l models/*.joblib → model.joblib 451385 · model_intervals.joblib 899705 · (candidates: primary 451385, baseline 5297, not committed)
$ docker images predmaint → 1.11GB (content 243,180,146 bytes); /app/models contains all four committed model files
$ ruff check . → All checks passed!  ·  ruff format --check . → 68 files already formatted
$ pytest -q → Required test coverage of 85% reached. Total coverage: 95.42% · 208 passed, 2 skipped
$ pytest -q -m data --no-cov → 12 passed · DATA_DIR=/nonexistent pytest -q → 199 passed, 11 skipped (gate reached)
```

**Findings:**

- Coverage is below the nominal 80 % out-of-fold in every bucket. As expected, it is 0 for true RUL > 125 against unclipped truth because of the cap. Reported in 16 §9, not tuned. Sidecar: late example 6.3 in [4.84, 18.76] (covers true 7); engine 1 125.0 in [118.45, 125.0] (does not cover true 112).
- In the synthetic test fixture Ridge wins CV, so a served baseline correctly makes interval training refuse (ADR-034 needs the served HGB hyperparameters). The fixture therefore promotes the primary explicitly (as `promote()` writes it), and a dedicated test covers the refusal.
- Two existing tests that assert exact response bodies (`/health` without a model; `/predict` == `predict_rul`) gained the new fields. That is a pure schema addition; no other existing assertion changed. `api_version` comes from `predmaint.__version__` (now 0.2.0), so no version literal changed in tests.
- **CR-009:** a demo stack started from this repo at 18:52:36Z (pre-M8 image, `api_version` 0.1.0) held ports 8000/8501, so the first smoke-test attempt failed. That stack was **not** stopped or modified. Fix: compose host ports via `API_HOST_PORT`/`DASHBOARD_HOST_PORT` (defaults unchanged), and the smoke test uses 18000/18501. The CI `docker` job will exercise the new interval assertions on the next push.
- Coverage of `src/predmaint`: 95.42 % (from 94.98 %).

## M9 — minimal anomaly detection (2026-11-08)

**Start:** M8 committed by the user as `a25df0e`; CI run 42514386686 green (lint-test + docker). (A first M9 attempt was stopped because M8 was still uncommitted.) **Change control:** CR-010, ADR-035, FR-019, doc 00 scope, M9 plan.

**Point and interval models untouched:** SHA-256 of `model.joblib` (`39bfa6fd…28c1`), `model_metadata.json` (`57f80600…dcfd`), `model_intervals.joblib` (`9c7cfc5e…6d12`) and `model_intervals_metadata.json` (`2dd9a1b3…c075`) recorded before M9 and re-checked after training and the smoke test: all `OK`. Re-running anomaly training reproduced `model_anomaly.joblib` byte-for-byte (`shasum -c` → `OK`).

**scipy:** `scipy.stats.spearmanr` resolves from scipy **1.18.1**, installed as a scikit-learn 1.9.1 requirement (`pip show scikit-learn` → `Requires: joblib, narwhals, numpy, scipy, threadpoolctl`) in both the venv and the image. Not pinned in `requirements.txt`; no new dependency.

```text
$ python scripts/train.py --model anomaly
Anomaly detector for served model primary-20261108-f4c86df: 14 rolling-mean features; healthy reference = true RUL >= 125: 5,329 rows from 90 engines (of 17,731 eligible rows)
Method: mahalanobis_ledoitwolf; OOF threshold (q0.99) = 5.4036; in-sample threshold on all healthy rows = 5.1510
Fallback: not needed: median train-OOF Spearman 0.680 >= 0.5
[1] Trend Spearman(score, cycle), train OOF: median 0.680 (IQR 0.506–0.818); share > 0.7 = 0.460 (100/100 engines scored)
[1] Trend Spearman(score, cycle), test: median 0.213 (IQR -0.177–0.569); share > 0.7 = 0.141 (99/100 engines scored)
[2] False-alarm rate on healthy rows (train OOF): 0.0101 of 5,329 rows
[3] Detection (score > threshold for 3 consecutive cycles), train OOF: fraction of life median 0.758 (IQR 0.638–0.872); true RUL at detection median 49.0 (IQR 26.0–75.8); never flagged 0.020
[4] Test engines flagged at last cycle: overall 0.250; by true RUL: 0-20: 0.750 (n=16), 21-60: 0.391 (n=23), 61-125: 0.080 (n=50), >125: 0.000 (n=11)
[5] Spearman(normalized score, RUL prediction) at test last cycle: -0.613
FR-019 acceptance (median per-engine Spearman(score, cycle) on train OOF > 0.7): value 0.680 → NOT MET
Timing: OOF evaluation 0.06 s
Saved models/model_anomaly.joblib, models/model_anomaly_metadata.json, reports/metrics_anomaly.json
Elapsed: 0.2 s     [time -p: real 1.82, user 1.85, sys 0.84]
$ API_PORT=8001 python -m predmaint.api.main   (local API; 8000 kept free for the demo stack)
$ curl /health → {"status":"ok","model_loaded":true,"model_version":"primary-20261108-f4c86df","api_version":"0.3.0","intervals_loaded":true,"anomaly_loaded":true}
$ curl -X POST /anomaly @examples/predict_request.json → {"unit_id":"test-unit-1","last_cycle":31,"anomaly_score":3.5973,"normalized_score":0.6657,"threshold":5.4036,"is_anomalous":false,"model_version":"anomaly-20261108-a25df0e","series":null}
$ curl -X POST /anomaly @examples/predict_request_late.json → {"unit_id":"test-unit-34","last_cycle":203,"anomaly_score":5.9519,"normalized_score":1.1015,"threshold":5.4036,"is_anomalous":true,"model_version":"anomaly-20261108-a25df0e","series":null}
$ curl -X POST "/anomaly?include_series=true" @examples/predict_request_late.json → series of 174 entries (= 203 − 30 + 1):
  first 3: {cycle 30: 0.8445}, {31: 0.8533}, {32: 0.8337} … last 3: {201: 1.1271}, {202: 1.1481}, {203: 1.1015}
$ curl -X POST /anomaly (10 records) → {"error":{"code":"INSUFFICIENT_CYCLES","message":"10 cycles received; at least 30 are required","details":[]}}
$ bash scripts/smoke_test.sh; echo $?   (ports 18000/18501)
  /health → {… "api_version":"0.3.0","intervals_loaded":true,"anomaly_loaded":true}
  /predict late → {… "predicted_rul":6.3,"health_status":"critical", … "rul_lower":4.84,"rul_upper":18.76,"interval_level":0.8}
  /anomaly engine 1 → normalized_score 0.6657, is_anomalous false · engine 34 → 1.1015, is_anomalous true
  SMOKE TEST PASSED (arm64) · exit code: 0
$ ls -l models/*.joblib → model.joblib 451385 · model_intervals.joblib 899705 · model_anomaly.joblib 4845 · (candidates: primary 451385, baseline 5297, not committed)
$ docker image inspect predmaint:latest → 243,192,895 bytes content (docker images: 1.11GB); /app/models holds all six committed model files
$ ruff check . → All checks passed!  ·  ruff format --check . → 70 files already formatted
$ pytest -q → Required test coverage of 85% reached. Total coverage: 95.92% · 251 passed, 2 skipped
$ pytest -q -m data --no-cov → 13 passed · DATA_DIR=/nonexistent pytest -q → 241 passed, 12 skipped; coverage 95.86% (gate reached)
```

**Test-trend check (in-session, committed artifact on real FD001; not stored in a report file):** per-engine Spearman(normalised score, cycle) on test engines, grouped by true RUL at cutoff: ≤ 60 → median 0.517 (n = 39); 61–125 → 0.040 (n = 49); > 125 → −0.320 (n = 11); Spearman(trend, true RUL) across engines = −0.525. This is the evidence for the explanation in 16 §10b.

**Findings:**

- **FR-019 acceptance NOT met:** median train-OOF trend 0.680 ≤ 0.7 (46 % of engines above 0.7). Reported, not tuned; the fallback threshold (0.5) was not reached, so IsolationForest was not tried. Per the M9 authorization, the project title stays unchanged and no title ADR was added; the feature is described everywhere as implemented but not meeting its target.
- **ADR-035 corrected:** the healthy reference covers 90 of 100 engines (5,329 rows), not every engine. An engine needs ≥ 155 cycles of life (window 30 + RUL ≥ 125). Recorded as a correction note in 13.
- **Defects fixed within M9 (no contract affected):**
  - `spearmanr` emitted `ConstantInputWarning` on the synthetic fixture (constant last-cycle inputs). Fixed in project code: a guard returns `None` ("undefined") for constant or < 3-point inputs, so no warning filter was added.
  - The `AnomalyResponse` OpenAPI example carried placeholder numbers; it now shows the recorded engine-34 values.
  - The dashboard's anomaly table formatted a possibly-null trend median; it now shows "—".
- The synthetic test fixture lowers the healthy bound to true RUL ≥ 20 (`healthy_min_rul=20`), because synthetic engines live only 50–75 cycles. Real training uses `RUL_CAP`, and a test asserts that training refuses when no healthy rows exist.
- One existing test changed: `/health` without a model, an exact-body assertion, gained `"anomaly_loaded": false`. All other existing assertions are unchanged.
- The anomaly metadata records `git_dirty: true` (trained on the uncommitted M9 tree on top of `a25df0e`); its `model_version` is therefore `anomaly-20261108-a25df0e`.
- Coverage of `src/predmaint`: 95.92 % (from 95.42 %); `models/anomaly.py` 99 %.

## Screenshots (2026-11-08, docs only, after M9)

The owner supplied four dashboard screenshots as attachments in this session. None were in the repository. Each was identified from its content (active tab), not its filename, and copied to the path the README already references. No code changed, and the README needed no change.

| Source (session attachment) | Destination | Shows | Pixels | Bytes |
|---|---|---|---|---|
| `1.png` | `docs/img/dashboard_explorer.png` | Engine explorer, test engine 1 at 31 cycles: RUL 125.0, interval 118.5–125.0, HEALTHY, anomaly 0.67 NORMAL, sensor chart | 1880 × 1388 | 398,676 |
| `5.png` (replaces `2.png`) | `docs/img/dashboard_trajectory.png` | RUL trajectory, train engine 1, computed: predicted vs capped true RUL over cycles 30–192, shaded 80 % band, dashed health bands, and the anomaly-score chart below | 1296 × 1352 | 207,250 |
| `6.png` (replaces `3.png`) | `docs/img/dashboard_upload.png` | Upload of `examples/predict_request_late.json`: HTTP 200, CRITICAL, `predicted_rul` 6.3, `rul_lower` 4.84, `rul_upper` 18.76 (engine 34) | 1290 × 1342 | 140,175 |
| `4.png` | `docs/img/dashboard_model_info.png` | Model info: model card, prediction intervals, anomaly detector (NOT met), metrics | 2880 × 6000 | 814,704 |

The first trajectory and upload screenshots showed empty tabs, so the owner supplied replacements (`5.png`, `6.png`), which overwrote the same paths. **All four screenshots now show their features.** All files are under 2 MB, so none was resized. The upload result matches the sidecar and the M9 smoke test (6.3 in [4.84, 18.76], critical). The explorer values match the M9 `curl` output for `examples/predict_request.json`, which closes AC-11 (11). Each README image link resolves to an existing file.


| Item | Value | Source |
|---|---|---|
| **Served model** | `primary-20261108-f4c86df` (HGB, lr 0.05, max_iter 200, 15 leaves, l2 0) | `models/model_metadata.json`, 05 §7c |
| Primary official test MAE / RMSE / NASA | 12.567 / 16.610 / 441.5 | M3 run 2026-11-08 (05 §7a) |
| Primary CV MAE / RMSE (mean ± std) | 11.843 ± 0.676 / 16.559 ± 0.695 | same |
| Baseline official test MAE / RMSE / NASA (clamp [0, 125]) | 15.347 / 19.144 / 590.6 | same; M2 value with clamp ≥ 0 only: 15.403 / 19.205 / 595.2 |
| Baseline CV MAE / RMSE (mean ± std) | 15.112 ± 1.156 / 19.152 ± 1.182 | same; M2: 15.504 ± 1.052 / 19.287 ± 1.181 |
| Primary training time (12-config search + final fit + test) | 76.6–80 s wall; 582 s CPU (user + sys) | `time` |
| Baseline training wall time | 1.4 s total | M2 |
| Artifact sizes | `model.joblib` / `primary.joblib` 451,385 B; `baseline.joblib` 5,297 B | `ls -l models/*.joblib` |
| Coverage `src/predmaint` | 94 % (747 stmts, 45 missed) | `pytest -q --cov=predmaint` |
| `/predict` latency (30 cycles, in-process, 100 calls) | p50 10.9 ms, p95 13.6 ms | M4 NFR-003 command |
| Coverage `src/predmaint` (after M4) | 94 % (971 stmts, 55 missed) | `pytest -q --cov=predmaint` |
| Late-life example (test 34, 203 cycles) | predicted 6.3 vs true 7 (`critical`) | sidecar + M5 dashboard session |
| Coverage `src/predmaint` (after M5) | 95 % (1,095 stmts, 55 missed; `dashboard/app.py` omitted) | `pytest -q --cov=predmaint` |
| Docker image (arm64) | 1.1 GB on disk (`docker images`); 242,889,009 bytes content size; build 57 s cold | M6 |
| Smoke test (arm64) | PASSED, exit 0; forced failure exit 1 | M6 |
| Coverage (after M6) | 94.98 % (gate 85 %) | `pytest -q` |
| **Final (M7)** | 187 passed, 2 skipped (opt-in); 11 real-data tests passed; coverage 94.98 % (gate 85 %); CI run 42509206610 green (lint-test + docker) | M7 |
| **M8** | 208 passed, 2 skipped; 12 real-data tests; coverage 95.42 %; intervals CV coverage 0.509 / 0.737 (unclipped / clipped), test 0.740 / 0.820; smoke test exit 0 | M8 |
| **M9** | 251 passed, 2 skipped; 13 real-data tests; coverage 95.92 %; anomaly trend median 0.680 train OOF (target > 0.7, **not met**), 0.213 test; false-alarm rate 0.0101; detection at median 0.758 of life / 49 cycles RUL; test flags 0.750 (RUL ≤ 20) → 0.000 (> 125); smoke test exit 0 | M9 |
