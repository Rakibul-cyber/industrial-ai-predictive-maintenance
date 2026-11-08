# 14 — Project Status

| Field | Value |
|---|---|
| Current milestone | **M1 — Scaffold, CI, data acquisition & validation: complete, awaiting review** |
| Last updated | 2026-11-08 |
| Git ref | branch `main` @ `71295ba` "docs: Milestone 0 planning baseline (architecture frozen)" (not pushed). M1 work is **uncommitted** in the working tree for review; remote `origin` = `https://github.com/Rakibul-cyber/industrial-ai-predictive-maintenance.git` (public). |
| Next approved action | **Await user review of M1** and authorization for Milestone 2 (and, separately, any commit/push of M1). |

## Milestone tracker

| Milestone | Status | Evidence |
|---|---|---|
| M0 Planning | **Approved** 2026-11-08 (architecture frozen; ADR-001…021 Accepted, ADR-022 added) | M0 section below |
| M1 Scaffold, CI, data | **Complete** 2026-11-08 (uncommitted, awaiting review) | M1 section below; CI run pending authorized push |
| M2 Labels & features | Not started | — |
| M3 Training & artifacts | Not started | — |
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
| U3 | Near-constant sensor list / adequacy of drop rule — **UNVERIFIED** | Open | 08 M2-T1, 12 R12 |
| U4 | NASA score official implementation — **UNVERIFIED** (interpretation fixed by ADR-009) | Open | 12 R5 |
| U5 | GitHub Actions runner resolves Python 3.13 — **UNVERIFIED**; `.github/workflows/ci.yml` exists but has never run — first run needs a user-authorized push | Open | 08 M1-T7, 12 R11 |
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

## Measured results

None yet — model metrics, training time, latency, and coverage are recorded here from M3 onwards.
