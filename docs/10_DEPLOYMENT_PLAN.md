# 10 — Deployment Plan

Target: **local machine only** (developer laptop / reviewer machine). Cloud deployment, Kubernetes, reverse proxies and secrets management are out of scope (00 §6). Cost: **$0**: local compute, a free public dataset download, and GitHub Actions on a public repository (`github.com/Rakibul-cyber/industrial-ai-predictive-maintenance`, visibility confirmed `public` via the GitHub REST API on 2026-11-08).

## 1. Prerequisites

| Tool | Needed for | Version verified on the dev machine (2026-11-08) |
|---|---|---|
| Docker Engine + Compose v2 | Option A (Docker) | Docker 29.6.2, Compose v5.3.1, Apple M1 (arm64), 8 CPUs, 4 GB VM memory |
| Python 3.13 | Option B (local) | 3.13.3 (`python3`); image uses 3.13.18 |
| git | both | 2.54.0 |
| curl, python3 on the host | `scripts/smoke_test.sh` | present on macOS and GitHub's ubuntu-24.04 runner |
| Network | first start only | HTTPS to `phm-datasets.s3.amazonaws.com` (FD001, ~12 MB) |

## 2. Option A — Docker Compose (recommended)

The served model is committed (ADR-030) and baked into the image (ADR-032), so a fresh clone needs **no training**:

```bash
git clone https://github.com/Rakibul-cyber/industrial-ai-predictive-maintenance.git
cd industrial-ai-predictive-maintenance
export APP_UID=$(id -u) APP_GID=$(id -g)   # Linux: dashboard writes ./data as you (harmless on macOS)
docker compose up -d --build               # builds predmaint:latest; first start downloads FD001 into ./data
docker compose ps                          # both services "(healthy)"; dashboard waits for a healthy api
open http://localhost:8501                 # dashboard (Linux: xdg-open); API docs: http://localhost:8000/docs
docker compose logs -f api dashboard       # troubleshooting
docker compose down                        # stop and remove containers + network (./data is kept)
```

One-shot verification (what CI runs; exits non-zero on any failure, always tears down):

```bash
bash scripts/smoke_test.sh   # build → up (ports 18000/18501) → wait healthy → /health (model + intervals) → /predict (late example = critical, lower ≤ point ≤ upper) → dashboard health → down
```

## 3. Option B — local Python

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .
python scripts/download_data.py            # → data/raw/CMAPSSData/ (checksum-verified)
pytest -q                                  # incl. 85 % coverage gate
python -m predmaint.api.main               # terminal 1: API on $API_HOST:$API_PORT (127.0.0.1:8000), committed model
streamlit run src/predmaint/dashboard/app.py --server.address localhost --server.port 8501  # terminal 2
```

## 4. Environment variables

Defined in `.env.example`. Option B: nothing loads `.env` automatically (no python-dotenv), so export it with `set -a && . ./.env && set +a`. Option A: Docker Compose reads a `.env` file in the project directory **for variable substitution in `docker-compose.yml` only**; container paths and hosts are fixed in the compose file.

| Variable | Used by | Local default | In compose |
|---|---|---|---|
| `DATA_DIR` | scripts; dashboard (raw FD001 for explorer/trajectory) | `data` | `/app/data` (fixed) |
| `MODEL_DIR` | train script output dir; default base of `MODEL_PATH` | `models` | `/app/models` (image ENV) |
| `MODEL_PATH` | API: served model file (`<stem>_metadata.json` beside it) | `$MODEL_DIR/model.joblib` | `/app/models/model.joblib` (fixed) |
| `API_HOST` / `API_PORT` | `python -m predmaint.api.main` | `127.0.0.1` / `8000` | `0.0.0.0` / `8000` (fixed) |
| `LOG_LEVEL` | API loggers + uvicorn | `INFO` | `${LOG_LEVEL:-INFO}` |
| `API_BASE_URL` | dashboard | `http://localhost:8000` | `http://api:8000` (fixed) |
| `API_TIMEOUT_SECONDS` | dashboard, per API call | `10` | `${API_TIMEOUT_SECONDS:-10}` |
| `APP_UID` / `APP_GID` | compose: user the dashboard runs as (writes `./data`) | — | `${APP_UID:-1000}` / `${APP_GID:-1000}` |
| `API_HOST_PORT` / `DASHBOARD_HOST_PORT` | compose: host ports (CR-009) | — | `${API_HOST_PORT:-8000}` / `${DASHBOARD_HOST_PORT:-8501}` |

No secrets exist in the MVP. `.env` is gitignored.

## 5. Image and compose (ADR-015, ADR-030, ADR-031, ADR-032)

**Image `predmaint:latest`** (one `Dockerfile`, single stage):

- `FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90`: the multi-arch index digest (amd64 + arm64) of `python:3.13-slim` on 2026-11-08, i.e. Python 3.13.18 on Debian trixie (R9).
- `pip install --only-binary=:all: -r requirements.txt`: every dependency installs from a wheel, so no compiler is needed and none is present (verified: no `gcc`/`cc`/`make` in the image). A multi-stage build is therefore unnecessary.
- Copies `src/`, `scripts/`, `examples/`, and the six committed model files: `models/model.joblib` + `models/model_metadata.json`, since M8 `models/model_intervals.joblib` + `models/model_intervals_metadata.json` (ADR-034), since M9 `models/model_anomaly.joblib` + `models/model_anomaly_metadata.json` (ADR-035), and since M12 `models/model_feature_importance.json` (ADR-040). `.dockerignore` excludes `.git`, `.github`, `.venv*`, `data/`, `docs/`, `tests/`, `reports/`, `models/baseline*`, `models/primary*`, caches and `.env`.
- `PYTHONPATH=/app/src` (no package install step), `HOME=/tmp` (so Streamlit works under any host UID), non-root user `app` (uid/gid 1000), `EXPOSE 8000 8501`, default `CMD ["python", "-m", "predmaint.api.main"]`.
- **Size (arm64, 2026-11-08):** `docker images` → **1.1 GB** on disk; `docker image inspect --format '{{.Size}}'` → 242,889,009 bytes (layer content size). Unpacked site-packages are 655 MB, led by pyarrow 155 MB (a Streamlit dependency), scipy 117 MB, pandas 83 MB, scikit-learn 57 MB, numpy 40 MB, streamlit 39 MB. Build time 57 s cold, seconds when cached. After M8 (interval artifact added): `docker images` → **1.11 GB**; content size 243,180,146 bytes. After M9 (anomaly artifact, 4,845 + 3,927 bytes): **1.11 GB**; content size 243,192,895 bytes.

**Compose (`docker-compose.yml`, project `predmaint`):**

| Service | Command | Ports (host) | Volumes | Healthcheck | Other |
|---|---|---|---|---|---|
| `api` | `python -m predmaint.api.main` | `127.0.0.1:8000` | `./data:/app/data:ro` (not used by the API; lets `docker compose exec api` scripts read FD001) | `python -c` + urllib on `/health`: passes only if `model_loaded` is true (ADR-028), every 10 s, timeout 5 s, 5 retries, start period 20 s | builds the image; `restart: unless-stopped` |
| `dashboard` | `python scripts/download_data.py` (failure only warns) then `exec streamlit run … --server.address 0.0.0.0 --server.port 8501 --server.headless true` | `127.0.0.1:8501` | `./data:/app/data` (read-write: FD001 downloaded here) | `python -c` + urllib: `/_stcore/health` body must be `ok`, start period 90 s | `depends_on: api: condition: service_healthy`; `user: ${APP_UID:-1000}:${APP_GID:-1000}`; `API_BASE_URL=http://api:8000`; `pull_policy: never`; `restart: unless-stopped` |

Ports are published on **127.0.0.1 only**, so nothing is exposed on the LAN by default.

## 6. Model artifact and retraining (ADR-030)

- `models/model.joblib` (451,385 bytes, SHA-256 `39bfa6fd4a168b263acac4ccefa0be81a4fc838afa7eb14bf7306ad1099128c1`) and `models/model_metadata.json` are **committed together** and baked into the image. Candidate files (`baseline*`, `primary*`) stay ignored.
- Since M8: `models/model_intervals.joblib` (899,705 bytes; dict of the q0.1/q0.9 pipelines) + `models/model_intervals_metadata.json`, committed and baked in as well. They are loaded next to `MODEL_PATH`. If they are absent or built for another `model_version`, the API serves point predictions with null interval fields (ADR-034). `/health.intervals_loaded` shows the state.
- Since M9: `models/model_anomaly.joblib` (4,845 bytes; scaler, Ledoit-Wolf covariance, feature list, OOF threshold) + `models/model_anomaly_metadata.json`, committed and baked in. Loaded next to `MODEL_PATH`; if absent, unloadable or built for another `model_version`, `/anomaly` returns 503 `ANOMALY_UNAVAILABLE` and nothing else changes (ADR-035). `/health.anomaly_loaded` shows the state. Retraining it takes ~2 s and reproduced the committed file byte-for-byte in M9.
- Pickle safety net: scikit-learn is pinned exactly (1.9.1); `/model/info` reports `sklearn_version_match`; `tests/test_model_artifact.py` fails if the installed scikit-learn differs from the metadata's version, and checks that the committed model reproduces `examples/expected_predictions.json`. `tests/test_real_data.py` (with the dataset) checks that it reproduces the official-test RMSE in its metadata.
- **Retraining procedure:**

  ```bash
  python scripts/download_data.py
  python scripts/train.py --model baseline
  python scripts/train.py --model primary
  python scripts/train.py --select          # writes models/model.joblib + models/model_metadata.json
  # model_version embeds the git SHA, so refresh the recorded demo expectations (CR-006):
  python scripts/make_request_payload.py --unit 1 --out examples/predict_request.json --expected examples/expected_predictions.json
  python scripts/make_request_payload.py --unit 1 --last 30 --out examples/predict_request_min.json --expected examples/expected_predictions.json
  python scripts/make_request_payload.py --unit 34 --out examples/predict_request_late.json --expected examples/expected_predictions.json
  pytest -q                                 # artifact + real-data checks
  python scripts/train.py --model intervals # AFTER --select: interval models for the new served model (ADR-034)
  python scripts/train.py --model anomaly   # AFTER --select: anomaly detector for the new served model (ADR-035; ~2 s)
  python scripts/train.py --explain         # AFTER --select: permutation importance (ADR-040; ~15 s)
  # (then the three make_request_payload.py commands above, so the sidecar has the new bounds and anomaly block,
  #  and update docs/18_MODEL_CARD.md from the new files: it was generated in M12 by a one-off script that is
  #  not committed (14), and tests/test_model_card.py fails while its artifact hashes are stale)
  git add models/model.joblib models/model_metadata.json \
          models/model_intervals.joblib models/model_intervals_metadata.json \
          models/model_anomaly.joblib models/model_anomaly_metadata.json \
          models/model_feature_importance.json \
          reports/ examples/expected_predictions.json docs/18_MODEL_CARD.md   # all model files TOGETHER
  docker compose up -d --build              # rebuild: the model is baked into the image
  ```

- If the model cannot be loaded (e.g. `MODEL_PATH` overridden to a missing file), the API still starts: `/health` returns 200 with `model_loaded: false`, `/predict` and `/model/info` return 503, and the compose healthcheck keeps `api` **unhealthy**, so the dashboard's `service_healthy` dependency is not satisfied (compose behaviour; this failure path was not exercised in M6).

## 7. Dataset in containers (ADR-031)

- `./data` is bind-mounted into both services: read-write for the dashboard, read-only for the API. On every dashboard start, `scripts/download_data.py` runs. If FD001 is present with matching SHA-256 it prints "Dataset already present and checksums match; nothing downloaded" and the container is offline-capable. Otherwise it downloads the 12 MB archive from the verified NASA URL, checks all checksums and extracts. Verified in M6: first start downloaded; restart skipped.
- If the download fails (offline first start), the dashboard still starts; the explorer and trajectory show the download hint, and status, Upload and Model info keep working.
- Linux: the dashboard runs as `APP_UID:APP_GID` so it can write into the host-owned `./data`. Export them (Quickstart) — the default 1000:1000 only works if that is your UID. macOS Docker Desktop maps ownership, so the default works there.

## 8. Health checks

- `GET /health` = liveness (always 200) + readiness (`model_loaded`). The compose healthcheck and the dashboard status bar read `model_loaded`.
- Dashboard: `GET /_stcore/health` → `ok`.
- `docker compose ps` shows `(healthy)` for both when the stack is ready; the smoke test waits for exactly that (timeout 240 s, `SMOKE_TIMEOUT_S`).

## 9. Supported architectures

- **arm64:** built, run and smoke-tested on the dev machine (Apple M1) on 2026-11-08: `SMOKE TEST PASSED (arm64)`, exit 0.
- **amd64:** confirmed by CI. The `docker` job builds and smoke-tests the image on GitHub's `ubuntu-24.04` (x64) hosted runner on every push; first green run [42509206610](https://github.com/Rakibul-cyber/industrial-ai-predictive-maintenance/actions/runs/42509206610) on `ccf68a7`, docker job 46 s. The Linux bind-mount *write* path is not asserted by the smoke test (a download failure only warns), so it is not separately confirmed by CI (14).

## 10. Known limits

- Image size 1.1 GB on disk, dominated by scientific/Streamlit wheels; one image serves both roles (ADR-032), so the API image carries Streamlit/pyarrow.
- The dashboard's first start needs network access; the API never does.
- Retraining requires a rebuild (model baked in) and a commit of all six model files; the interval anomaly and feature-importance artifacts must be regenerated after every `--select`, otherwise the API ignores them (`for_model_version` mismatch).
- Ports 8000/8501 must be free for the demo stack (override with `API_HOST_PORT` / `DASHBOARD_HOST_PORT`). The smoke test uses its own compose project (`predmaint-smoke`) and its own host ports 18000/18501 (`SMOKE_API_PORT`, `SMOKE_DASHBOARD_PORT`), so it runs while a demo stack is up (CR-009).
- No TLS, auth, CORS, rate limiting or multi-worker serving: local demo only (ADR-012).
