# Industrial AI — Predictive Maintenance Platform

Remaining Useful Life (RUL) prediction for turbofan engines on the NASA C-MAPSS **FD001** dataset, served through a FastAPI service and a Streamlit dashboard.

**Status:** Milestones 1–4 implemented: data acquisition & validation, features, Ridge baseline and HistGradientBoosting primary model, CV-based selection, the inference contract, and the FastAPI service. Measured results: [reports/model_comparison.md](reports/model_comparison.md). The dashboard and Docker setup are not built yet. A CI badge will be added once the workflow badge URL is confirmed on `main`. This README is finalised in Milestone 7.

## What it will do

- Download the dataset reproducibly from NASA with checksum verification.
- Train a Ridge baseline and a HistGradientBoosting model with engine-grouped cross-validation, evaluate both on the official test set (MAE, RMSE, NASA score), and ship the better one with versioned metadata.
- Serve predictions via `POST /predict` (plus `/health`, `/model/info`).
- Show predictions and sensor trends in a dashboard that gets every result from the API.
- Run both services with `docker compose up`.

## Quickstart (current: Milestone 4)

Requires Python 3.13 (`python3`) and network access to NASA's S3 bucket.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .
python scripts/download_data.py      # downloads ~12 MB, verifies SHA-256, extracts to data/raw/CMAPSSData/
python -m predmaint.data.validate    # checks the FD001 data contract (docs/04 §6)
python scripts/train.py --model baseline   # Ridge: CV + official test metrics → models/, reports/
python scripts/train.py --model primary    # HistGradientBoosting: 12-config CV search (~80 s) + test metrics
python scripts/train.py --select           # promote the lower-CV-RMSE model → models/model.joblib, reports/model_comparison.md
ruff check . && ruff format --check .
pytest -q                            # real-data tests run when data/raw/ is present
```

## Run the API

Requires the served model from the Quickstart (`models/model.joblib`). Configuration comes from environment variables (`MODEL_PATH`, `API_HOST`, `API_PORT`, `LOG_LEVEL`; see [.env.example](.env.example) and [docs/10_DEPLOYMENT_PLAN.md](docs/10_DEPLOYMENT_PLAN.md) §3).

```bash
python -m predmaint.api.main          # http://127.0.0.1:8000 — interactive docs at /docs
curl -s localhost:8000/health
curl -s localhost:8000/model/info
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d @examples/predict_request_min.json
# → {"unit_id":"test-unit-1","predicted_rul":125.0,"health_status":"healthy","cycles_received":30,"cycles_used":30,"last_cycle":31,"model_version":"primary-20261108-f4c86df","warnings":[]}
```

`examples/` holds FD001 test engine 1 (all 31 cycles, and the last 30), with expected results in `examples/expected_predictions.json`. Without a model the API still starts: `/health` reports `"model_loaded": false`, and `/predict` and `/model/info` return 503. Full contract: [docs/06_API_SPECIFICATION.md](docs/06_API_SPECIFICATION.md).

The dashboard arrives in Milestone 5 ([docs/08_IMPLEMENTATION_PLAN.md](docs/08_IMPLEMENTATION_PLAN.md)).

## Planned stack

Python 3.13 · pandas · scikit-learn · FastAPI + Pydantic v2 + uvicorn · Streamlit · pytest · ruff · Docker Compose · GitHub Actions. Exact versions: [docs/03_TECH_STACK.md](docs/03_TECH_STACK.md).

## Documentation

Start with [docs/00_PROJECT_CHARTER.md](docs/00_PROJECT_CHARTER.md); current state is in [docs/14_PROJECT_STATUS.md](docs/14_PROJECT_STATUS.md); the build plan is [docs/08_IMPLEMENTATION_PLAN.md](docs/08_IMPLEMENTATION_PLAN.md). All documents: [docs/](docs/).

## Data source

NASA Prognostics Center of Excellence Data Repository. Dataset citation (as requested by the repository): A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA. Paper: A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08, 2008. No licence is stated by NASA for this dataset. The data is not redistributed in this repository; it is downloaded by script (see [docs/04_DATASET_SPECIFICATION.md](docs/04_DATASET_SPECIFICATION.md)).
