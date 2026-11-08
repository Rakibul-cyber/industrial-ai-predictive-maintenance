# 10 — Deployment Plan

Target: **local machine only** (developer laptop / reviewer machine). Cloud deployment is out of scope (00 §6). Cost: **$0** (local compute; dataset is a free public download; GitHub Actions on a public repository — `origin` = `github.com/Rakibul-cyber/industrial-ai-predictive-maintenance`, confirmed `"visibility": "public"`, empty, default branch `main` via the GitHub REST API on 2026-11-08).

## 1. Prerequisites

| Tool | Version verified on dev machine (2026-11-08) |
|---|---|
| Python | 3.13 (`python3`; 3.13.3 locally) |
| pip | 25.3 |
| Docker Engine / Compose | 29.6.2 / v5.3.1 |
| git | 2.54.0 |
| Network | HTTPS to `phm-datasets.s3.amazonaws.com` (data download only) |

## 2. Local (non-Docker) run — full sequence

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .
cp .env.example .env                       # optional; defaults work without it
python scripts/download_data.py            # → data/raw/CMAPSSData/ (checksum-verified)
python scripts/train.py                    # → models/model.joblib + models/model_metadata.json
pytest -v
uvicorn predmaint.api.main:app --host 0.0.0.0 --port 8000      # terminal 1
streamlit run src/predmaint/dashboard/app.py --server.port 8501  # terminal 2
```

Open http://localhost:8501 (dashboard) and http://localhost:8000/docs (API docs).

## 3. Environment variables (documented in `.env.example`, M1)

| Variable | Used by | Default | Compose value |
|---|---|---|---|
| `DATA_DIR` | scripts, dashboard | `data` | `/app/data` |
| `MODEL_DIR` | train script, API | `models` | `/app/models` |
| `LOG_LEVEL` | API | `INFO` | `INFO` |
| `API_BASE_URL` | dashboard | `http://localhost:8000` | `http://api:8000` |
| `API_TIMEOUT_SECONDS` | dashboard | `10` | `10` |

No secrets exist in the MVP. `.env` is gitignored.

## 4. Docker (implemented in M6; ADR-015)

**Image (one `Dockerfile`):** `FROM python:3.13-slim@sha256:<digest pinned in M6>` → non-root user `app` → `WORKDIR /app` → `pip install --no-cache-dir -r requirements.txt` → copy `pyproject.toml` + `src/` → `pip install --no-deps .` → `EXPOSE 8000 8501` → default `CMD ["uvicorn", "predmaint.api.main:app", "--host", "0.0.0.0", "--port", "8000"]`. Dev dependencies, data, artifacts, and `.git` are excluded via `.dockerignore`.

**Compose (`docker-compose.yml`):**

| Service | Command | Ports | Volumes | Healthcheck |
|---|---|---|---|---|
| `api` | default CMD | `8000:8000` | `./models:/app/models:ro` | `python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"` every 10 s, 5 retries (slim image has no curl) |
| `dashboard` | `streamlit run src/predmaint/dashboard/app.py --server.port 8501 --server.address 0.0.0.0` | `8501:8501` | `./data:/app/data:ro` | — ; `depends_on: api` with `condition: service_started` so the dashboard can show "API unavailable" itself |

Note: the dashboard command path requires `src/` to be present in the image (it is copied for the package install).

**Commands:**

```bash
python scripts/download_data.py && python scripts/train.py   # artifacts must exist on host first
docker compose up -d --build
docker compose ps                     # api should be "healthy"
curl -s localhost:8000/health
docker compose logs -f api            # troubleshooting
docker compose down
```

## 5. Artifact availability

- Artifacts are produced on the host by `scripts/train.py` and mounted read-only into `api`. They are not baked into the image, so retraining never requires a rebuild and the image contains no data-derived content.
- **Demo reproducibility (ADR-022):** after `python scripts/download_data.py`, one documented train command (`python scripts/train.py`) followed by `docker compose up` is the complete path to a running demo.
- If `models/` is empty: `api` starts, and `/health`, `/predict`, `/model/info` return HTTP 503 with the standard error body (`MODEL_UNAVAILABLE`, ADR-022); the container stays `unhealthy` and logs state the missing path. Fix: run `scripts/train.py`, then `docker compose restart api`.
- `model_metadata.json` is committed in M7 so the reported metrics are reviewable. Whether `model.joblib` is small enough to commit (removing the train step from the demo path) is decided in M6 (task M6-T4) with its measured size; until then it is not committed (regenerable).

## 6. Health checks

- `GET /health` = readiness (model loaded). Used by the compose healthcheck and the dashboard status indicator.
- Dashboard: HTTP 200 on `/` (verified via `curl -sI localhost:8501`).

## 7. Supported architectures

Linux cp313 wheels for every runtime dependency were confirmed for **x86_64 and aarch64** (03). The image is built and smoke-tested natively on the dev machine (arm64) in M6; an amd64 build is **UNVERIFIED** (R10) and not required for the MVP.
