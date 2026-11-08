# 07 — Repository Structure

Final structure after Milestone 7. Package name: `predmaint` (ADR-019). Files are created only in the milestone listed in [08](08_IMPLEMENTATION_PLAN.md).

```
industrial-ai-predictive-maintenance/
├── README.md                      # Project overview, quick start, results (finalised M7)
├── CLAUDE.md                      # Standing instructions for AI-assisted sessions
├── pyproject.toml                 # Package metadata (src layout), ruff + pytest config
├── requirements.txt               # Pinned runtime deps (03)
├── requirements-dev.txt           # -r requirements.txt + pinned test/lint deps
├── .env.example                   # Documented env vars (10 §3)
├── .gitignore
├── .dockerignore                  # Keeps .git, data/, docs/, tests/, reports/, candidate models out of the build (M6)
├── Dockerfile                     # Single-stage image for api + dashboard; base pinned by digest; model baked in (M6)
├── docker-compose.yml             # Services: api (8000), dashboard (8501), healthchecks, ./data bind mount (M6)
├── .github/workflows/ci.yml       # ruff + pytest on push/PR (M1)
├── src/predmaint/
│   ├── __init__.py                # Package version
│   ├── config.py                  # Single home of WINDOW_SIZE, RUL_CAP, seeds, bands + env-driven paths (shared by training and inference)
│   ├── data/                      # Acquisition, parsing, validation, labelling
│   │   ├── download.py            #   download + SHA-256 + safe nested extraction
│   │   ├── load.py                #   whitespace parsing with named columns
│   │   ├── validate.py            #   data-contract checks (04 §6)
│   │   └── labels.py              #   RUL labels with clip
│   ├── features/
│   │   └── build.py               # Sensor selection, causal rolling features, last-row extraction
│   ├── models/                    # Training, metrics, artifacts, inference
│   │   ├── registry.py            #   baseline/primary pipeline factories
│   │   ├── metrics.py             #   MAE, RMSE, NASA score
│   │   ├── train.py               #   CV, final fit, test evaluation, report (M2); selection (M3)
│   │   ├── artifacts.py           #   save/load joblib + metadata JSON
│   │   ├── selection.py           #   ADR-024 selection + promotion to model.joblib, comparison reports (M3)
│   │   └── predict.py             #   inference contract: load_model, predict_rul, health_status, errors
│   ├── api/                       # HTTP serving layer
│   │   ├── main.py                #   create_app(), lifespan model loading, routes, request logging; `python -m predmaint.api.main`
│   │   ├── schemas.py             #   Pydantic request/response models
│   │   ├── service.py             #   Model holder, request → predict_rul adapter, demo-payload helpers
│   │   └── errors.py              #   Error envelope + exception handlers
│   └── dashboard/                 # UI layer: talks to the API only (imports config + data.load)
│       ├── api_client.py          #   requests wrapper → ApiResult (body or docs/06 envelope); never raises
│       ├── helpers.py             #   true RUL, /predict bodies, trajectory sampling, chart data (no model code)
│       └── app.py                 #   Streamlit app: status bar + explorer, trajectory, upload, model-info tabs
├── scripts/                       # Thin CLI entrypoints only
│   ├── download_data.py           #   → predmaint.data.download
│   ├── train.py                   #   → predmaint.models.train
│   ├── make_request_payload.py    #   → builds a /predict body from test_FD001
│   └── smoke_test.sh              #   Docker smoke test: build, up, health, /predict, teardown (M6; CI docker job)
├── tests/                         # pytest suite (09)
│   ├── conftest.py                #   synthetic C-MAPSS-like fixtures, tiny trained model fixture
│   ├── test_download.py           #   M1
│   ├── test_load.py               #   M1
│   ├── test_validate.py           #   M1
│   ├── test_real_data.py          #   M1 (marker: data)
│   ├── test_labels.py             #   M2
│   ├── test_features.py           #   M2
│   ├── test_metrics.py            #   M2
│   ├── test_train.py              #   M2 (baseline), M3 (primary + selection)
│   ├── test_selection.py          #   M3
│   ├── test_inference_contract.py #   M3
│   ├── test_api.py                #   M4
│   ├── test_dashboard_client.py   #   M5: api_client + import boundary
│   ├── test_dashboard_helpers.py  #   M5: true RUL, payloads, trajectory
│   ├── test_dashboard_app.py      #   M5: streamlit AppTest smoke tests (API mocked)
│   ├── test_model_artifact.py     #   M6: committed model present, sklearn pinned, reproduces sidecar
│   └── test_docker_smoke.py       #   M6 (marker: docker): runs scripts/smoke_test.sh
├── data/                          # gitignored except .gitkeep
│   ├── .gitkeep
│   ├── raw/                       #   downloaded archive + extracted CMAPSSData/
│   └── processed/                 #   reserved; unused in MVP
├── models/                        # artifacts
│   ├── .gitkeep
│   ├── baseline.joblib            #   per-candidate artifact (M2), gitignored
│   ├── baseline_metadata.json     #   per-candidate metadata, gitignored (reports/ holds the committed record)
│   ├── primary.joblib             #   per-candidate artifact (M3), gitignored (+ primary_metadata.json)
│   ├── model.joblib               #   served (selected) model — committed (ADR-030)
│   └── model_metadata.json        #   served model metadata — committed with it
├── examples/                      # committed demo payloads (ADR-029: FD001 test engines 1 and 34)
│   ├── predict_request.json       #   all 31 cycles
│   ├── predict_request_min.json   #   last 30 cycles (= WINDOW_SIZE)
│   ├── predict_request_late.json  #   FD001 test engine 34 (smallest true RUL, 7), all 203 cycles
│   └── expected_predictions.json  #   sidecar: expected /predict result + true RUL for each
├── reports/                       # small committed metric reports
│   ├── metrics_baseline.json      #   written by scripts/train.py --model baseline (M2)
│   ├── metrics_primary.json       #   written by scripts/train.py --model primary (M3)
│   ├── model_comparison.json      #   written by scripts/train.py --select (M3)
│   └── model_comparison.md        #   human-readable comparison + selection statement (M3)
└── docs/                          # 00–15 planning and design documents
```

Rules: no notebooks as source of truth; no logic in `scripts/`; `dashboard/` depends only on `requests`, `pandas`, `numpy`, `streamlit`, `predmaint.config` and `predmaint.data.load` (tested).
