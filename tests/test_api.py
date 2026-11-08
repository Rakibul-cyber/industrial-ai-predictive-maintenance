"""API tests (docs/06) against a tiny synthetic model trained into a temp dir — never models/."""

import json
import logging
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from predmaint import __version__
from predmaint.api import service as service_module
from predmaint.api.main import create_app
from predmaint.api.schemas import MAX_RECORDS_PER_REQUEST
from predmaint.api.service import request_payload
from predmaint.data.load import load_fd001
from predmaint.models.predict import (
    InvalidValuesError,
    clear_model_cache,
    interval_path_for,
    load_interval_model,
    load_model,
    predict_rul,
)


@pytest.fixture
def model_path(served_model_dir: Path) -> Path:
    return served_model_dir / "models" / "model.joblib"


@pytest.fixture
def client(model_path: Path) -> Iterator[TestClient]:
    clear_model_cache()
    with TestClient(create_app(model_path=model_path)) as test_client:
        yield test_client


@pytest.fixture
def no_model_client(tmp_path: Path) -> Iterator[TestClient]:
    clear_model_cache()
    with TestClient(create_app(model_path=tmp_path / "missing" / "model.joblib")) as test_client:
        yield test_client


@pytest.fixture
def payload(served_model_dir: Path) -> dict[str, Any]:
    test = load_fd001(served_model_dir / "CMAPSSData").test
    return request_payload(test[test["unit"] == 2], unit_id="engine-2")  # 40 cycles


def _assert_envelope(response: Any, status: int, code: str) -> dict[str, Any]:
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"error"}, body
    assert set(body["error"]) == {"code", "message", "details"}
    assert body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]
    assert isinstance(body["error"]["details"], list)
    return body["error"]


# --- /health ------------------------------------------------------------------------------
def test_health_with_model(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"].startswith(("baseline-", "primary-"))
    assert body["api_version"] == __version__


def test_health_without_model_is_200_degraded(no_model_client: TestClient) -> None:
    response = no_model_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "model_loaded": False,
        "model_version": None,
        "api_version": __version__,
        "intervals_loaded": False,
    }


def test_model_path_env_pointing_nowhere(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_model_cache()
    monkeypatch.setenv("MODEL_PATH", "/nonexistent/model.joblib")
    with TestClient(create_app()) as test_client:
        health = test_client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is False
        _assert_envelope(test_client.get("/model/info"), 503, "MODEL_UNAVAILABLE")


# --- /predict -----------------------------------------------------------------------------
def test_predict_matches_predict_rul(
    client: TestClient, payload: dict[str, Any], model_path: Path
) -> None:
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    direct = predict_rul(
        payload["records"],
        load_model(model_path),
        load_interval_model(interval_path_for(model_path)),
    )
    assert body == {
        "unit_id": "engine-2",
        "predicted_rul": round(direct.predicted_rul, 2),
        "health_status": direct.health_status,
        "cycles_received": 40,
        "cycles_used": 30,
        "last_cycle": 40,
        "model_version": direct.model_version,
        "warnings": direct.warnings,
        "rul_lower": round(direct.rul_lower, 2),
        "rul_upper": round(direct.rul_upper, 2),
        "interval_level": 0.8,
    }
    assert 0 <= body["predicted_rul"] <= 125


def test_predict_unit_id_is_optional(client: TestClient, payload: dict[str, Any]) -> None:
    body = client.post("/predict", json={"records": payload["records"]}).json()
    assert body["unit_id"] is None


def test_insufficient_cycles(client: TestClient, payload: dict[str, Any]) -> None:
    short = {**payload, "records": payload["records"][:10]}
    error = _assert_envelope(client.post("/predict", json=short), 422, "INSUFFICIENT_CYCLES")
    assert "10 cycles" in error["message"]


def test_missing_sensor(client: TestClient, payload: dict[str, Any], model_path: Path) -> None:
    sensor = load_model(model_path).required_sensors[0]
    records = [dict(r) for r in payload["records"]]
    records[5][sensor] = None
    error = _assert_envelope(
        client.post("/predict", json={"records": records}), 422, "MISSING_SENSORS"
    )
    assert error["details"] == [{"sensor": sensor}]


@pytest.mark.parametrize("case", ["swapped", "gap", "duplicate"])
def test_non_consecutive_cycles(client: TestClient, payload: dict[str, Any], case: str) -> None:
    records = [dict(r) for r in payload["records"]]
    if case == "swapped":
        records[0], records[1] = records[1], records[0]
    elif case == "gap":
        for record in records[10:]:
            record["cycle"] += 1
    else:
        records[10]["cycle"] = records[9]["cycle"]
    _assert_envelope(
        client.post("/predict", json={"records": records}), 422, "NON_CONSECUTIVE_CYCLES"
    )


def test_invalid_values_error_maps_to_validation_error(
    client: TestClient, payload: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: object) -> None:
        raise InvalidValuesError("cycle numbers must be positive integers")

    monkeypatch.setattr(service_module, "predict_rul", fail)
    error = _assert_envelope(client.post("/predict", json=payload), 422, "VALIDATION_ERROR")
    assert error["message"] == "cycle numbers must be positive integers"


def test_predict_and_info_503_without_model(
    no_model_client: TestClient, payload: dict[str, Any]
) -> None:
    error = _assert_envelope(
        no_model_client.post("/predict", json=payload), 503, "MODEL_UNAVAILABLE"
    )
    assert "No model loaded" in error["message"]
    _assert_envelope(no_model_client.get("/model/info"), 503, "MODEL_UNAVAILABLE")


def test_unexpected_error_is_500_without_traceback(
    client: TestClient,
    payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def boom(*_args: object) -> None:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(service_module, "predict_rul", boom)
    with caplog.at_level(logging.ERROR, logger="predmaint.api"):
        response = client.post("/predict", json=payload)
    error = _assert_envelope(response, 500, "INTERNAL_ERROR")
    assert error["message"] == "Internal server error"
    assert "Traceback" not in response.text and "secret" not in response.text
    assert any("secret internal detail" in r.getMessage() or r.exc_info for r in caplog.records)


# --- request validation (Pydantic) → standard envelope ------------------------------------
def _bad_bodies(payload: dict[str, Any]) -> dict[str, Any]:
    first = payload["records"][0]
    return {
        "missing records": {"unit_id": "x"},
        "wrong type": {"records": [{**first, "cycle": "abc"}]},
        "fractional cycle": {"records": [{**first, "cycle": 1.5}]},
        "cycle below 1": {"records": [{**first, "cycle": 0}]},
        "empty list": {"records": []},
        "too many cycles": {"records": [first] * (MAX_RECORDS_PER_REQUEST + 1)},
        "extra field in record": {"records": [{**first, "s_99": 1.0}]},
        "extra top-level field": {**payload, "engine": 1},
        "unit_id too long": {**payload, "unit_id": "x" * 65},
    }


@pytest.mark.parametrize(
    "case",
    [
        "missing records",
        "wrong type",
        "fractional cycle",
        "cycle below 1",
        "empty list",
        "too many cycles",
        "extra field in record",
        "extra top-level field",
        "unit_id too long",
    ],
)
def test_request_validation_uses_envelope(
    client: TestClient, payload: dict[str, Any], case: str
) -> None:
    response = client.post("/predict", json=_bad_bodies(payload)[case])
    error = _assert_envelope(response, 422, "VALIDATION_ERROR")
    assert "detail" not in response.json()  # not FastAPI's default shape
    assert error["details"], error
    assert set(error["details"][0]) == {"loc", "msg", "type"}


@pytest.mark.parametrize("token", ["NaN", "Infinity"])
def test_non_finite_values_rejected(
    client: TestClient, payload: dict[str, Any], token: str
) -> None:
    """JSON NaN/Infinity literals are rejected by the schema (allow_inf_nan=False)."""
    body = json.dumps(payload)
    start = body.index('"s_2": ') + len('"s_2": ')
    raw = body[:start] + token + body[body.index(",", start) :]
    response = client.post("/predict", content=raw, headers={"Content-Type": "application/json"})
    _assert_envelope(response, 422, "VALIDATION_ERROR")


def test_malformed_json(client: TestClient) -> None:
    response = client.post(
        "/predict", content="{not json", headers={"Content-Type": "application/json"}
    )
    _assert_envelope(response, 422, "VALIDATION_ERROR")


# --- routing errors -----------------------------------------------------------------------
def test_unknown_route_404_envelope(client: TestClient) -> None:
    _assert_envelope(client.get("/does-not-exist"), 404, "NOT_FOUND")


def test_wrong_method_405_envelope(client: TestClient) -> None:
    _assert_envelope(client.get("/predict"), 405, "METHOD_NOT_ALLOWED")


# --- /model/info ----------------------------------------------------------------------------
def test_model_info_fields_and_types(client: TestClient, model_path: Path) -> None:
    response = client.get("/model/info")
    assert response.status_code == 200
    info = response.json()
    meta = load_model(model_path).metadata
    assert info["model_name"] == meta["model_name"]
    assert info["model_version"] == meta["model_version"]
    assert info["trained_at"] == meta["trained_at"]
    assert info["selected_by"] == "cv_rmse"
    assert isinstance(info["git_sha"], str)
    assert info["window_size"] == 30 and info["rul_cap"] == 125
    assert info["n_features"] == len(info["feature_names"]) == 1 + 3 * len(info["required_sensors"])
    assert info["health_bands"] == {"critical_below": 20, "warning_below": 60}
    assert set(info["metrics"]) == {"cv", "test", "test_clipped_truth"}
    assert set(info["metrics"]["cv"]) == {"mae_mean", "mae_std", "rmse_mean", "rmse_std"}
    assert set(info["metrics"]["test"]) == {"mae", "rmse", "nasa_score"}
    assert info["metrics"]["test"]["rmse"] == meta["metrics"]["test"]["rmse"]
    assert isinstance(info["sklearn_version_match"], bool)


# --- logging & docs -----------------------------------------------------------------------
def test_one_log_line_per_request(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="predmaint.api"):
        client.get("/health")
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("GET /health")]
    assert len(lines) == 1
    assert lines[0].split()[2] == "200" and lines[0].endswith("ms")


def test_openapi_metadata(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    assert spec["info"]["version"] == __version__
    assert "RUL" in spec["info"]["title"]
    assert {"/health", "/predict", "/model/info"} <= set(spec["paths"])


# --- prediction intervals (ADR-034) ---------------------------------------------------------
INTERVAL_SUMMARY_KEYS = {
    "n",
    "coverage_unclipped",
    "coverage_clipped",
    "mean_width",
    "median_width",
    "n_reordered",
}


def _client_with_interval_variant(
    served_model_dir: Path, tmp_path: Path, variant: str
) -> TestClient:
    models = tmp_path / "models"
    shutil.copytree(served_model_dir / "models", models)
    if variant == "absent":
        (models / "model_intervals.joblib").unlink()
        (models / "model_intervals_metadata.json").unlink()
    elif variant == "mismatch":
        meta_path = models / "model_intervals_metadata.json"
        meta = json.loads(meta_path.read_text())
        meta_path.write_text(json.dumps({**meta, "for_model_version": "primary-19990101-0000000"}))
    clear_model_cache()
    return TestClient(create_app(model_path=models / "model.joblib"))


def test_health_and_info_report_intervals(client: TestClient) -> None:
    assert client.get("/health").json()["intervals_loaded"] is True
    info = client.get("/model/info").json()["intervals"]
    assert info["quantiles"] == [0.1, 0.9] and info["interval_level"] == 0.8
    assert info["for_model_version"] == client.get("/health").json()["model_version"]
    for split in ("cv_oof", "test"):
        assert set(info[split]) == INTERVAL_SUMMARY_KEYS


def test_predict_interval_fields(client: TestClient, payload: dict[str, Any]) -> None:
    body = client.post("/predict", json=payload).json()
    assert body["interval_level"] == 0.8
    assert 0 <= body["rul_lower"] <= body["predicted_rul"] <= body["rul_upper"] <= 125


@pytest.mark.parametrize("variant", ["absent", "mismatch"])
def test_intervals_unavailable_serves_point_only(
    served_model_dir: Path,
    tmp_path: Path,
    client: TestClient,
    payload: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
    variant: str,
) -> None:
    with_intervals = client.post("/predict", json=payload).json()
    with (
        caplog.at_level(logging.WARNING, logger="predmaint.api"),
        _client_with_interval_variant(served_model_dir, tmp_path, variant) as other,
    ):
        health = other.get("/health").json()
        body = other.post("/predict", json=payload).json()
        info = other.get("/model/info").json()
    assert health["model_loaded"] is True and health["intervals_loaded"] is False
    assert (
        body["rul_lower"] is None and body["rul_upper"] is None and body["interval_level"] is None
    )
    assert body["predicted_rul"] == with_intervals["predicted_rul"]  # point unchanged
    assert info["intervals"] is None
    assert any("Prediction intervals" in r.getMessage() for r in caplog.records)
