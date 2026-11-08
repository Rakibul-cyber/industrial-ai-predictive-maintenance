# Industrial AI — Predictive Maintenance Platform

Remaining Useful Life (RUL) prediction for turbofan engines on the NASA C-MAPSS **FD001** dataset, served through a FastAPI service and a Streamlit dashboard.

**Status:** Planning complete (Milestone 0) — no application code yet. CI badge will be added once the GitHub Actions workflow (Milestone 1) has run on `main`. This README is finalised in Milestone 7.

## What it will do

- Download the dataset reproducibly from NASA with checksum verification.
- Train a Ridge baseline and a HistGradientBoosting model with engine-grouped cross-validation, evaluate both on the official test set (MAE, RMSE, NASA score), and ship the better one with versioned metadata.
- Serve predictions via `POST /predict` (plus `/health`, `/model/info`).
- Show predictions and sensor trends in a dashboard that gets every result from the API.
- Run both services with `docker compose up`.

## Planned stack

Python 3.13 · pandas · scikit-learn · FastAPI + Pydantic v2 + uvicorn · Streamlit · pytest · ruff · Docker Compose · GitHub Actions. Exact versions: [docs/03_TECH_STACK.md](docs/03_TECH_STACK.md).

## Documentation

Start with [docs/00_PROJECT_CHARTER.md](docs/00_PROJECT_CHARTER.md); current state is in [docs/14_PROJECT_STATUS.md](docs/14_PROJECT_STATUS.md); the build plan is [docs/08_IMPLEMENTATION_PLAN.md](docs/08_IMPLEMENTATION_PLAN.md). All documents: [docs/](docs/).

## Data source

NASA Prognostics Center of Excellence Data Repository — "Turbofan Engine Degradation Simulation Data Set". Reference: A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08, 2008. The data is not redistributed in this repository; it is downloaded by script (see [docs/04_DATASET_SPECIFICATION.md](docs/04_DATASET_SPECIFICATION.md)).
