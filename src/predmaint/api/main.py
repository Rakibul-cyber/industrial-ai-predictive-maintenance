"""FastAPI application (docs/06). Run: ``python -m predmaint.api.main`` (reads API_HOST/API_PORT)
or ``uvicorn predmaint.api.main:app --host 127.0.0.1 --port 8000``.
"""

import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, Request, Response

from predmaint import __version__
from predmaint.api.errors import error_response, install_error_handlers
from predmaint.api.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from predmaint.api.service import ModelService
from predmaint.config import get_settings

logger = logging.getLogger("predmaint.api")

#: Committed demo payload (docs/06 §7); used as the /docs example when the server runs from the
#: repository root. Optional: the API works without it.
EXAMPLE_REQUEST = Path("examples/predict_request_min.json")

DESCRIPTION = """
Remaining-useful-life (RUL) prediction for NASA C-MAPSS FD001 turbofan engines.

Send at least 30 consecutive cycles of one engine to `POST /predict`. The served model,
its metrics and the selection rule are reported by `GET /model/info`. Every error uses the
envelope `{"error": {"code", "message", "details"}}`.
"""

_ERRORS = {
    422: {"model": ErrorResponse, "description": "Validation or input-contract error"},
    503: {"model": ErrorResponse, "description": "No model loaded (MODEL_UNAVAILABLE)"},
    500: {"model": ErrorResponse, "description": "Unexpected server error"},
}


def _openapi_examples() -> dict[str, Any] | None:
    try:
        payload = json.loads(EXAMPLE_REQUEST.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return {
        "fd001_test_engine_1": {
            "summary": "FD001 test engine 1, last 30 cycles",
            "value": payload,
        }
    }


def create_app(model_path: Path | None = None) -> FastAPI:
    """Build the API. ``model_path`` overrides ``$MODEL_PATH`` (used by tests)."""
    settings = get_settings()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("predmaint").setLevel(settings.log_level)
    path = model_path or settings.model_path

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        service = ModelService(path)
        service.load()
        app.state.service = service
        yield

    app = FastAPI(
        title="Predictive Maintenance — RUL API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    install_error_handlers(app)

    @app.middleware("http")
    async def log_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """One log line per request; unexpected exceptions become 500 INTERNAL_ERROR."""
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled error on %s %s", request.method, request.url.path)
            response = error_response(500, "INTERNAL_ERROR", "Internal server error")
        logger.info(
            "%s %s %d %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response

    examples = _openapi_examples()
    body = Body(openapi_examples=examples) if examples else Body()

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    def health(request: Request) -> HealthResponse:
        """Liveness and model readiness. Always 200; check `model_loaded` (ADR-028)."""
        service: ModelService = request.app.state.service
        model = service.model
        return HealthResponse(
            status="ok" if model else "degraded",
            model_loaded=model is not None,
            model_version=model.model_version if model else None,
            api_version=__version__,
        )

    @app.post("/predict", response_model=PredictResponse, responses=_ERRORS, tags=["prediction"])
    def predict(request: Request, payload: Annotated[PredictRequest, body]) -> PredictResponse:
        """Predict RUL at the last cycle from the last 30 consecutive cycles of one engine."""
        service: ModelService = request.app.state.service
        return service.predict(payload)

    @app.get(
        "/model/info",
        response_model=ModelInfoResponse,
        responses={503: _ERRORS[503]},
        tags=["service"],
    )
    def model_info(request: Request) -> ModelInfoResponse:
        """Served model identity, provenance, selection rule and metrics."""
        service: ModelService = request.app.state.service
        return service.model_info()

    return app


app = create_app()


def run() -> None:
    """Start uvicorn on ``$API_HOST:$API_PORT`` (access log off: the app logs each request)."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "predmaint.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        access_log=False,
    )


if __name__ == "__main__":
    run()
