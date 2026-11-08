"""Tests for the dashboard's API client and the dashboard import boundary."""

import ast
from pathlib import Path
from typing import Any

import pytest
import requests

from predmaint.dashboard.api_client import UNEXPECTED, UNREACHABLE, ApiClient

DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "src" / "predmaint" / "dashboard"


class FakeResponse:
    def __init__(self, status: int, body: Any = None, text: str = "") -> None:
        self.status_code, self._body, self.text = status, body, text

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no JSON")
        return self._body


class FakeSession:
    def __init__(self, response: FakeResponse | Exception) -> None:
        self.response = response
        self.calls: list[tuple[str, str, Any, float]] = []

    def request(self, method: str, url: str, json: Any = None, timeout: float = 0) -> Any:
        self.calls.append((method, url, json, timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _client(response: FakeResponse | Exception) -> tuple[ApiClient, FakeSession]:
    session = FakeSession(response)
    return ApiClient("http://api:8000/", timeout=3.0, session=session), session  # type: ignore[arg-type]


def test_success_passes_body_through() -> None:
    body = {"status": "ok", "model_loaded": True, "model_version": "v", "api_version": "0.1.0"}
    client, session = _client(FakeResponse(200, body))
    result = client.health()
    assert result.ok and result.reachable and result.status_code == 200
    assert result.body == body and result.error is None
    assert session.calls == [("GET", "http://api:8000/health", None, 3.0)]


def test_predict_posts_payload() -> None:
    client, session = _client(FakeResponse(200, {"predicted_rul": 50.0}))
    payload = {"records": [{"cycle": 1}]}
    assert client.predict(payload).body == {"predicted_rul": 50.0}
    assert session.calls[0][:3] == ("POST", "http://api:8000/predict", payload)


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (422, "INSUFFICIENT_CYCLES"),
        (422, "MISSING_SENSORS"),
        (422, "NON_CONSECUTIVE_CYCLES"),
        (422, "VALIDATION_ERROR"),
        (503, "MODEL_UNAVAILABLE"),
        (404, "NOT_FOUND"),
        (500, "INTERNAL_ERROR"),
    ],
)
def test_error_envelope_passes_through_intact(status: int, code: str) -> None:
    envelope = {"error": {"code": code, "message": "m", "details": [{"sensor": "s_7"}]}}
    client, _ = _client(FakeResponse(status, envelope))
    result = client.predict({"records": []})
    assert not result.ok and result.status_code == status
    assert result.body == envelope
    assert result.error == envelope["error"]


@pytest.mark.parametrize(
    "exc",
    [requests.ConnectionError("refused"), requests.Timeout("slow"), requests.RequestException()],
)
def test_network_failure_is_clean_unreachable_result(exc: Exception) -> None:
    client, _ = _client(exc)
    result = client.model_info()  # must not raise
    assert not result.ok and not result.reachable and result.status_code is None
    assert result.error["code"] == UNREACHABLE
    assert "API unreachable at http://api:8000" in result.error["message"]


def test_non_json_and_envelope_less_errors() -> None:
    client, _ = _client(FakeResponse(502, None, text="Bad Gateway"))
    result = client.health()
    assert result.error["code"] == UNEXPECTED and "502" in result.error["message"]
    client, _ = _client(FakeResponse(500, {"detail": "x"}))
    assert client.health().error["code"] == UNEXPECTED


def test_dashboard_imports_only_allowed_project_modules() -> None:
    """The dashboard talks to the API only: no model, feature, inference or API-server code."""
    allowed = {"predmaint.config", "predmaint.data.load"}
    for path in DASHBOARD_DIR.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.startswith("predmaint") and not name.startswith("predmaint.dashboard"):
                    assert name in allowed, f"{path.name} imports {name}"
