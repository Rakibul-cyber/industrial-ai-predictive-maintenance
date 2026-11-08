# Industrial AI — Predictive Maintenance Platform

Remaining Useful Life (RUL) prediction for turbofan engines on the NASA C-MAPSS **FD001** dataset, served through a FastAPI service and a Streamlit dashboard.

**Status:** Milestones 1–2 implemented (data acquisition & validation; features + Ridge baseline with measured metrics in [reports/metrics_baseline.json](reports/metrics_baseline.json)); no primary model, API, or dashboard yet. A CI badge will be added once the GitHub Actions workflow has run on `main`. This README is finalised in Milestone 7.

## What it will do

- Download the dataset reproducibly from NASA with checksum verification.
- Train a Ridge baseline and a HistGradientBoosting model with engine-grouped cross-validation, evaluate both on the official test set (MAE, RMSE, NASA score), and ship the better one with versioned metadata.
- Serve predictions via `POST /predict` (plus `/health`, `/model/info`).
- Show predictions and sensor trends in a dashboard that gets every result from the API.
- Run both services with `docker compose up`.

## Quickstart (current: Milestone 2)

Requires Python 3.13 (`python3`) and network access to NASA's S3 bucket.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .
cp .env.example .env                 # optional; defaults work without it
python scripts/download_data.py      # downloads ~12 MB, verifies SHA-256, extracts to data/raw/CMAPSSData/
python -m predmaint.data.validate    # checks the FD001 data contract (docs/04 §6)
python scripts/train.py --model baseline   # CV + official test metrics; writes models/ and reports/
ruff check . && ruff format --check .
pytest -q                            # real-data tests run when data/raw/ is present
```

The primary model, API, and dashboard arrive in Milestones 3–5 ([docs/08_IMPLEMENTATION_PLAN.md](docs/08_IMPLEMENTATION_PLAN.md)).

## Planned stack

Python 3.13 · pandas · scikit-learn · FastAPI + Pydantic v2 + uvicorn · Streamlit · pytest · ruff · Docker Compose · GitHub Actions. Exact versions: [docs/03_TECH_STACK.md](docs/03_TECH_STACK.md).

## Documentation

Start with [docs/00_PROJECT_CHARTER.md](docs/00_PROJECT_CHARTER.md); current state is in [docs/14_PROJECT_STATUS.md](docs/14_PROJECT_STATUS.md); the build plan is [docs/08_IMPLEMENTATION_PLAN.md](docs/08_IMPLEMENTATION_PLAN.md). All documents: [docs/](docs/).

## Data source

NASA Prognostics Center of Excellence Data Repository. Dataset citation (as requested by the repository): A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA. Paper: A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08, 2008. No licence is stated by NASA for this dataset. The data is not redistributed in this repository; it is downloaded by script (see [docs/04_DATASET_SPECIFICATION.md](docs/04_DATASET_SPECIFICATION.md)).
