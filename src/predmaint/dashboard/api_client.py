"""Thin HTTP client for the RUL API (docs/06). Never raises: every call returns an ApiResult.

Wraps ``/health``, ``/model/info``, ``/predict`` and ``/anomaly`` (M9).

Success bodies and the docs/06 error envelope are passed through unchanged. Network failures
(connection refused, timeout, …) become a client-side envelope with code ``API_UNREACHABLE``.
"""

from dataclasses import dataclass
from typing import Any

import requests

UNREACHABLE = "API_UNREACHABLE"
UNEXPECTED = "UNEXPECTED_RESPONSE"


@dataclass(frozen=True)
class ApiResult:
    """Outcome of one API call.

    ``body`` is the parsed JSON body exactly as the API sent it (success payload or
    ``{"error": {...}}`` envelope). For network failures ``status_code`` is ``None`` and ``body``
    is a synthesized envelope with code ``API_UNREACHABLE``.
    """

    ok: bool
    status_code: int | None
    body: dict[str, Any]

    @property
    def reachable(self) -> bool:
        return self.status_code is not None

    @property
    def error(self) -> dict[str, Any] | None:
        """The ``{"code", "message", "details"}`` part of an error envelope, else ``None``."""
        return None if self.ok else self.body.get("error")


def _envelope(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": []}}


class ApiClient:
    """Wraps ``GET /health``, ``GET /model/info`` and ``POST /predict``."""

    def __init__(
        self, base_url: str, timeout: float = 10.0, session: requests.Session | None = None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def health(self) -> ApiResult:
        return self._call("GET", "/health")

    def model_info(self) -> ApiResult:
        return self._call("GET", "/model/info")

    def predict(self, payload: dict[str, Any]) -> ApiResult:
        return self._call("POST", "/predict", payload)

    def anomaly(self, payload: dict[str, Any], include_series: bool = False) -> ApiResult:
        path = "/anomaly?include_series=true" if include_series else "/anomaly"
        return self._call("POST", path, payload)

    def _call(self, method: str, path: str, payload: dict[str, Any] | None = None) -> ApiResult:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(method, url, json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            return ApiResult(
                False,
                None,
                _envelope(
                    UNREACHABLE, f"API unreachable at {self.base_url} ({type(exc).__name__})"
                ),
            )
        try:
            body = response.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            snippet = response.text[:200]
            return ApiResult(
                False,
                response.status_code,
                _envelope(UNEXPECTED, f"HTTP {response.status_code}: {snippet}"),
            )
        if response.ok:
            return ApiResult(True, response.status_code, body)
        if isinstance(body.get("error"), dict):
            return ApiResult(False, response.status_code, body)
        return ApiResult(
            False,
            response.status_code,
            _envelope(UNEXPECTED, f"HTTP {response.status_code} without error envelope"),
        )
