# 03 — Tech Stack

All versions below were **confirmed on 2026-11-08** in a throwaway venv (`.venv-verify/`, since deleted) on macOS 27.0.1 arm64 with **Python 3.13.3**: installed together with `pip install`, imported successfully, and `pip check` reported "No broken requirements found." Linux wheels (cp313, `manylinux_2_28` / `manylinux_2_17`) were downloaded successfully for **both x86_64 and aarch64** for the runtime set, which matters for the Docker image.

## 1. Runtime platform

| Item | Choice | Evidence |
|---|---|---|
| Python | **3.13** (local 3.13.3) | `python3 --version` → `Python 3.13.3`. `python` is not on PATH; always use `python3` / venv `python`. |
| Docker base image | `python:3.13-slim` | `docker run python:3.13-slim` → Python 3.13.18, Debian 13.7, glibc 2.41 (≥ 2.28 required by pandas/numpy/sklearn wheels). Tag is mutable — digest pinned in M6 (R9). |
| Container runtime | Docker 29.6.2, Compose v5.3.1 | `docker --version`, `docker compose version`; daemon reachable (`linux/aarch64`). |
| OS (dev) | macOS 27.0.1, arm64 | `sw_vers`, `uname -m`. |

## 2. One choice per responsibility

| Responsibility | Choice | Pin | Rationale |
|---|---|---|---|
| Dataframes | pandas | `3.1.1` | Standard; groupby-rolling for per-unit features. |
| Numerics | numpy | `2.5.4` | Required by pandas/sklearn. |
| ML (baseline + primary) | scikit-learn | `1.9.1` | Ridge, HistGradientBoostingRegressor, GroupKFold, Pipeline in one dependency (ADR-006/007). |
| Model serialisation | joblib | `1.6.0` | sklearn-recommended persistence (ADR-011). |
| API framework | fastapi | `0.162.2` | Typed request validation + OpenAPI docs (ADR-012). Pulls starlette `1.10.0`. |
| ASGI server | uvicorn | `0.54.0` | Standard FastAPI server. |
| Schemas | pydantic (v2) | `2.14.3` | FastAPI's validation layer; `allow_inf_nan=False`, `extra="forbid"`. |
| Dashboard | streamlit | `1.67.0` | Fast UI for a data app (ADR-014). |
| HTTP client (dashboard + dataset download) | requests | `2.34.2` | Already a declared dependency of streamlit 1.67.0 (`pip show streamlit`), so no new dependency; pinned explicitly because we import it. Also used by `data/download.py` because stdlib `urllib` fails TLS verification on python.org macOS builds (CR-001). |
| Test runner | pytest | `9.2.0` | Standard. |
| Coverage | pytest-cov | `7.1.0` | NFR-005 measurement. |
| API test transport | **httpx2** | `2.16.0` | See ADR-013: with starlette 1.10.0, `fastapi.testclient` + `httpx` emits `StarletteDeprecationWarning: Using httpx with starlette.testclient is deprecated; install httpx2 instead.` With httpx2 installed, TestClient imported `httpx2` (not `httpx`) and a `/health` call returned 200 under `-W error`. |
| Lint + format | ruff | `0.17.4` | One tool for both (ADR-016). |
| CI | GitHub Actions | `actions/setup-python` with `python-version: "3.13"` | **UNVERIFIED** that the hosted runner resolves 3.13 — verified on the first CI run in M1. |

## 3. Requirements files (created in M1)

- `requirements.txt` — runtime: pandas, numpy, scikit-learn, joblib, fastapi, uvicorn, pydantic, streamlit, requests (exact `==` pins above).
- `requirements-dev.txt` — `-r requirements.txt` + pytest, pytest-cov, httpx2, ruff.
- `pyproject.toml` — package metadata (`predmaint`, `requires-python = ">=3.13,<3.14"`), setuptools `src/` layout, ruff and pytest configuration. Dependencies are listed in the requirements files only (single source for pins).

## 4. Evaluated and rejected

| Candidate | Result | Decision |
|---|---|---|
| lightgbm 4.7.0 | Installs, but `import lightgbm` **fails**: `OSError: dlopen(... lib_lightgbm.dylib) Library not loaded: @rpath/libomp.dylib` (no Homebrew `libomp` on this machine). | Excluded (ADR-007). HistGradientBoostingRegressor needs no system library. |
| mlflow 3.18.0 | `pip install --dry-run` would add 40+ packages (Flask, SQLAlchemy, alembic, gunicorn, databricks-sdk, matplotlib, …). | Excluded (ADR-011); `model_metadata.json` covers lineage for one model. |
| httpx 0.28.1 | Works but triggers the deprecation warning above. | Replaced by httpx2 (ADR-013). |

Notable transitive versions observed: starlette 1.10.0, scipy 1.18.1, pyarrow 26.0.1, anyio 4.16.0, httpcore2 2.16.0.
