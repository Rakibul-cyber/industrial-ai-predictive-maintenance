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
├── .dockerignore                  # Keeps data/, models/*.joblib, .venv*, .git out of the image (M6)
├── Dockerfile                     # Single image for api + dashboard (M6)
├── docker-compose.yml             # Services: api (8000), dashboard (8501) (M6)
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
│   │   ├── train.py               #   CV, final fit, test evaluation, selection
│   │   ├── artifacts.py           #   save/load joblib + metadata JSON
│   │   └── predict.py             #   load_model, predict_rul, health_band, contract errors
│   ├── api/                       # HTTP serving layer
│   │   ├── main.py                #   FastAPI app, lifespan model loading, routes
│   │   ├── schemas.py             #   Pydantic request/response models
│   │   ├── service.py             #   Model holder + request → predict_rul adapter
│   │   └── errors.py              #   Error envelope + exception handlers
│   └── dashboard/                 # UI layer (never imports predmaint.models)
│       ├── client.py              #   requests-based API client
│       └── app.py                 #   Streamlit app
├── scripts/                       # Thin CLI entrypoints only
│   ├── download_data.py           #   → predmaint.data.download
│   ├── train.py                   #   → predmaint.models.train
│   └── make_request_payload.py    #   → builds a /predict body from test_FD001
├── tests/                         # pytest suite (09)
│   ├── conftest.py                #   synthetic C-MAPSS-like fixtures, tiny trained model fixture
│   ├── test_download.py           #   M1
│   ├── test_load.py               #   M1
│   ├── test_validate.py           #   M1
│   ├── test_real_data.py          #   M1 (marker: data)
│   ├── test_labels.py             #   M2
│   ├── test_features.py           #   M2
│   ├── test_metrics.py            #   M3
│   ├── test_train.py              #   M3
│   ├── test_inference_contract.py #   M3
│   ├── test_api.py                #   M4
│   ├── test_dashboard_client.py   #   M5
│   └── test_docker_smoke.py       #   M6 (marker: docker)
├── data/                          # gitignored except .gitkeep
│   ├── .gitkeep
│   ├── raw/                       #   downloaded archive + extracted CMAPSSData/
│   └── processed/                 #   reserved; unused in MVP
├── models/                        # artifacts
│   ├── .gitkeep
│   ├── model.joblib               #   gitignored
│   └── model_metadata.json        #   committed with reported results (M7)
└── docs/                          # 00–15 planning and design documents
```

Rules: no notebooks as source of truth; no logic in `scripts/`; `dashboard/` depends only on `client.py`, `requests`, `pandas`, `streamlit`, and `predmaint.data.load` (for parsing local/uploaded raw files).
