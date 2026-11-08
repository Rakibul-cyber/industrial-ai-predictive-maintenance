"""Error envelope and the single exception-handler layer (docs/06 §6)."""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from predmaint.models.predict import (
    AnomalyNotAvailableError,
    InputContractError,
    InsufficientHistoryError,
    InvalidValuesError,
    MissingColumnsError,
    ModelNotAvailableError,
    NonConsecutiveCyclesError,
)

logger = logging.getLogger("predmaint.api")

#: Inference-contract exception → (HTTP status, error code). Order matters: most specific first.
CONTRACT_ERRORS: tuple[tuple[type[Exception], int, str], ...] = (
    (InsufficientHistoryError, 422, "INSUFFICIENT_CYCLES"),
    (MissingColumnsError, 422, "MISSING_SENSORS"),
    (NonConsecutiveCyclesError, 422, "NON_CONSECUTIVE_CYCLES"),
    (InvalidValuesError, 422, "VALIDATION_ERROR"),
    (ModelNotAvailableError, 503, "MODEL_UNAVAILABLE"),
    (AnomalyNotAvailableError, 503, "ANOMALY_UNAVAILABLE"),
)
_HTTP_CODES = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}


def error_response(
    status_code: int, code: str, message: str, details: list[dict[str, Any]] | None = None
) -> JSONResponse:
    """Build the standard ``{"error": {code, message, details}}`` response."""
    body = {"error": {"code": code, "message": message, "details": details or []}}
    return JSONResponse(status_code=status_code, content=body)


def _contract_error(_request: Request, exc: Exception) -> JSONResponse:
    for exc_type, status_code, code in CONTRACT_ERRORS:
        if isinstance(exc, exc_type):
            details: list[dict[str, Any]] = []
            if isinstance(exc, MissingColumnsError):
                details = [{"sensor": column} for column in exc.columns]
            return error_response(status_code, code, str(exc), details)
    return error_response(422, "VALIDATION_ERROR", str(exc))  # other InputContractError


def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    details = [
        {"loc": list(err.get("loc", ())), "msg": str(err.get("msg", "")), "type": err.get("type")}
        for err in exc.errors()
    ]
    return error_response(422, "VALIDATION_ERROR", "Request validation failed", details)


def _http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = _HTTP_CODES.get(exc.status_code, "HTTP_ERROR")
    return error_response(exc.status_code, code, str(exc.detail))


def install_error_handlers(app: FastAPI) -> None:
    """Register handlers so every error leaves the API in the docs/06 §6 envelope.

    Unexpected exceptions are converted to 500 ``INTERNAL_ERROR`` by the request-logging
    middleware in :mod:`predmaint.api.main` (full traceback logged, never returned).
    """
    app.add_exception_handler(InputContractError, _contract_error)
    app.add_exception_handler(ModelNotAvailableError, _contract_error)
    app.add_exception_handler(AnomalyNotAvailableError, _contract_error)
    app.add_exception_handler(RequestValidationError, _validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, _http_error)  # type: ignore[arg-type]
