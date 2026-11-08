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
python scripts/download_data.py            # → data/raw/CMAPSSData/ (checksum-verified)
python scripts/train.py --model baseline   # → models/baseline.*, reports/metrics_baseline.json
python scripts/train.py --model primary    # → models/primary.*, reports/metrics_primary.json (~80 s on 8 cores)
python scripts/train.py --select           # → models/model.joblib + model_metadata.json, reports/model_comparison.*
pytest -q
python -m predmaint.api.main               # terminal 1: API on $API_HOST:$API_PORT (127.0.0.1:8000)
streamlit run src/predmaint/dashboard/app.py --server.port 8501  # terminal 2
```

Open http://localhost:8501 (dashboard) and http://localhost:8000/docs (API docs).

## 3. Environment variables (documented in `.env.example`)

Nothing loads `.env` automatically (no python-dotenv; adding it would need an ADR). To use a `.env`, export it into the shell: `cp .env.example .env && set -a && . ./.env && set +a`.

| Variable | Used by | Default | Compose value |
|---|---|---|---|
| `DATA_DIR` | scripts, dashboard | `data` | `/app/data` |
| `MODEL_DIR` | train script (output dir); API default for `MODEL_PATH` | `models` | `/app/models` |
| `MODEL_PATH` | API: served model file (metadata `<stem>_metadata.json` beside it) | `$MODEL_DIR/model.joblib` | `/app/models/model.joblib` |
| `API_HOST` | `python -m predmaint.api.main` | `127.0.0.1` | `0.0.0.0` |
| `API_PORT` | `python -m predmaint.api.main` | `8000` | `8000` |
| `LOG_LEVEL` | API (predmaint loggers + uvicorn) | `INFO` | `INFO` |
| `API_BASE_URL` | dashboard | `http://localhost:8000` | `http://api:8000` |
| `API_TIMEOUT_SECONDS` | dashboard | `10` | `10` |

No secrets exist in the MVP. `.env` is gitignored.

## 4. Docker (implemented in M6; ADR-015)

**Image (one `Dockerfile`):** `FROM python:3.13-slim@sha256:<digest pinned in M6>` → non-root user `app` → `WORKDIR /app` → `pip install --no-cache-dir -r requirements.txt` → copy `pyproject.toml` + `src/` → `pip install --no-deps .` → `EXPOSE 8000 8501` → default `CMD ["python", "-m", "predmaint.api.main"]` with `API_HOST=0.0.0.0`. Dev dependencies, data, artifacts, and `.git` are excluded via `.dockerignore`.

**Compose (`docker-compose.yml`):**

| Service | Command | Ports | Volumes | Healthcheck |
|---|---|---|---|---|
| `api` | default CMD | `8000:8000` | `./models:/app/models:ro` | `python -c "import json,sys,urllib.request; sys.exit(0 if json.load(urllib.request.urlopen('http://localhost:8000/health'))['model_loaded'] else 1)"` — checks `model_loaded`, because `/health` is 200 even without a model (ADR-028) every 10 s, 5 retries (slim image has no curl) |
| `dashboard` | `streamlit run src/predmaint/dashboard/app.py --server.port 8501 --server.address 0.0.0.0` | `8501:8501` | `./data:/app/data:ro` | — ; `depends_on: api` with `condition: service_started` so the dashboard can show "API unavailable" itself |

Note: the dashboard command path requires `src/` to be present in the image (it is copied for the package install).

**Commands:**

```bash
python scripts/download_data.py && python scripts/train.py --model baseline && python scripts/train.py --model primary && python scripts/train.py --select   # artifacts must exist on host first
docker compose up -d --build
docker compose ps                     # api should be "healthy"
curl -s localhost:8000/health
docker compose logs -f api            # troubleshooting
docker compose down
```

## 5. Artifact availability

- Artifacts are produced on the host by `scripts/train.py` and mounted read-only into `api`. They are not baked into the image, so retraining never requires a rebuild and the image contains no data-derived content.
- **Demo reproducibility (ADR-022):** after `python scripts/download_data.py`, one documented train command (`python scripts/train.py`) followed by `docker compose up` is the complete path to a running demo.
- If `models/` is empty: `api` starts; `/health` returns 200 with `model_loaded: false` (ADR-028); `/predict` and `/model/info` return HTTP 503 with the standard error body (`MODEL_UNAVAILABLE`, ADR-022); the container stays `unhealthy` and logs state the missing path. Fix: run `scripts/train.py`, then `docker compose restart api`.
- `model_metadata.json` is committed in M7 so the reported metrics are reviewable. Whether `model.joblib` is small enough to commit (removing the train step from the demo path) is decided in M6 (task M6-T4) with its measured size; until then it is not committed (regenerable).

## 6. Health checks

- `GET /health` = liveness (always 200) + readiness (`model_loaded`). The compose healthcheck and the dashboard status indicator read `model_loaded`.
- Dashboard: HTTP 200 on `/` (verified via `curl -sI localhost:8501`).

## 7. Supported architectures

Linux cp313 wheels for every runtime dependency were confirmed for **x86_64 and aarch64** (03). The image is built and smoke-tested natively on the dev machine (arm64) in M6; an amd64 build is **UNVERIFIED** (R10) and not required for the MVP.
