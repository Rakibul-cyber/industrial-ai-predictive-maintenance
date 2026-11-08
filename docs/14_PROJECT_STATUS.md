# 14 — Project Status

| Field | Value |
|---|---|
| Current milestone | **M3 — Primary model, selection, served artifact, inference contract: complete, awaiting review** |
| Last updated | 2026-11-08 |
| Git ref | `main` = `origin/main` @ `f4c86df` "feat: implement feature engineering, model training, and evaluation framework" (M2, committed and pushed by the user; CI run 42492050196 green), on top of `4e8c04c` (M1) and `71295ba` (M0). The working tree was clean at the start of M3; M3 work is **uncommitted** for review. Remote `origin` = `https://github.com/Rakibul-cyber/industrial-ai-predictive-maintenance.git` (public). |
| Next approved action | **Await user review of M3** and authorization for Milestone 4 (and, separately, any commit/push of M3 — the next push also exercises the updated CI workflow). |

## Milestone tracker

| Milestone | Status | Evidence |
|---|---|---|
| M0 Planning | **Approved** 2026-11-08 (architecture frozen; ADR-001…021 Accepted, ADR-022 added) | M0 section below |
| M1 Scaffold, CI, data | **Accepted** 2026-11-08; committed `4e8c04c`, pushed; CI green | M1 section + CI section below |
| M2 Labels, features, baseline (re-scoped, CR-002) | **Accepted** 2026-11-08; committed `f4c86df`, pushed; CI green | M2 section below |
| M3 Primary model, selection, inference contract (CR-003) | **Complete** 2026-11-08 (uncommitted, awaiting review) | M3 section below |
| M4 API | Not started | — |
| M5 Dashboard | Not started | — |
| M6 Docker | Not started | — |
| M7 Finalisation | Not started | — |

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
| U2 | Third-party mirror availability — **UNVERIFIED** (only needed if S3 fails) | Open | 12 R1 |
| U3 | Near-constant sensor list — **resolved in M2**: ≤ 2-unique rule verified on real data (04 §5a) | Closed | 04 §5a, 12 R12 |
| U4 | NASA score official implementation — **UNVERIFIED** (interpretation fixed by ADR-009) | Open | 12 R5 |
| U5 | GitHub Actions runner resolves Python 3.13 — **resolved**: run 42491244396 on `4e8c04c` succeeded, `actions/setup-python@v5` step with `python-version: "3.13"` passed. Exact patch version not visible (logs need auth). | Closed | CI section, 12 R11 |
| U6 | Dockerfile/compose behaviour (not writable in M0) — **UNVERIFIED** | Open | 08 M6, 12 R8 |
| U7 | amd64 image build — **UNVERIFIED** (wheels confirmed) | Open | 12 R10 |
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

## Measured results

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
