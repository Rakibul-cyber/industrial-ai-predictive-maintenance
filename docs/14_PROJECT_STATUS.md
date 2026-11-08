# 14 — Project Status

| Field | Value |
|---|---|
| Current milestone | **M1 — Scaffold, CI, data acquisition & validation: in progress** |
| Last updated | 2026-11-08 |
| Git ref | branch `main`; M0 planning baseline committed as "docs: Milestone 0 planning baseline (architecture frozen)" (not pushed); remote `origin` = `https://github.com/Rakibul-cyber/industrial-ai-predictive-maintenance.git` (public). |
| Next approved action | Implement Milestone 1 (authorized 2026-11-08). |

## Milestone tracker

| Milestone | Status | Evidence |
|---|---|---|
| M0 Planning | **Approved** 2026-11-08 (architecture frozen; ADR-001…021 Accepted, ADR-022 added) | M0 section below |
| M1 Scaffold, CI, data | In progress | — |
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
| U1 | Dataset licence / usage terms — **UNVERIFIED** | Open | 08 M1-T8, 12 R1 |
| U2 | Third-party mirror availability — **UNVERIFIED** (only needed if S3 fails) | Open | 12 R1 |
| U3 | Near-constant sensor list / adequacy of drop rule — **UNVERIFIED** | Open | 08 M2-T1, 12 R12 |
| U4 | NASA score official implementation — **UNVERIFIED** (interpretation fixed by ADR-009) | Open | 12 R5 |
| U5 | GitHub Actions runner resolves Python 3.13 — **UNVERIFIED**; first CI run needs a user-authorized push | Open | 08 M1-T7, 12 R11 |
| U6 | Dockerfile/compose behaviour (not writable in M0) — **UNVERIFIED** | Open | 08 M6, 12 R8 |
| U7 | amd64 image build — **UNVERIFIED** (wheels confirmed) | Open | 12 R10 |
| U8 | Column → physical sensor name mapping — **UNVERIFIED**, not used (ADR-020) | Accepted limitation | 12 R2 |

## Measured results

None yet — model metrics, training time, latency, and coverage are recorded here from M3 onwards.
