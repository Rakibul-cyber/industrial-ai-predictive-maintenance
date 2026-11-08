# One image for both services (ADR-032): the API (default CMD) and the dashboard (compose command).
# Single stage: every runtime dependency ships cp313 manylinux wheels for amd64 and arm64
# (docs/03), enforced below with --only-binary, so no compiler is ever installed.
# Base pinned by multi-arch index digest (risk R9): python:3.13-slim, Python 3.13.18, Debian trixie.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    HOME=/tmp \
    DATA_DIR=/app/data \
    MODEL_DIR=/app/models \
    MODEL_PATH=/app/models/model.joblib \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY requirements.txt .
RUN pip install --only-binary=:all: -r requirements.txt \
    && groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app/data && chown app:app /app/data

# Application code, CLI scripts (dataset download), demo payloads, the committed served model
# (ADR-030), its prediction-interval artifact (ADR-034) and the anomaly detector (ADR-035).
# HOME=/tmp lets the dashboard run under an arbitrary host UID (APP_UID in compose).
COPY src/ src/
COPY scripts/ scripts/
COPY examples/ examples/
COPY models/model.joblib models/model_metadata.json \
     models/model_intervals.joblib models/model_intervals_metadata.json \
     models/model_anomaly.joblib models/model_anomaly_metadata.json models/

USER app
EXPOSE 8000 8501
CMD ["python", "-m", "predmaint.api.main"]
