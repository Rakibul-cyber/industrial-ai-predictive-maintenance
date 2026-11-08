"""Streamlit AppTest smoke tests with the API client patched to canned responses."""

import shutil
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from predmaint.dashboard.api_client import UNREACHABLE, ApiClient, ApiResult

APP = str(Path(__file__).resolve().parents[1] / "src" / "predmaint" / "dashboard" / "app.py")

HEALTH = {"status": "ok", "model_loaded": True, "model_version": "test-v1", "api_version": "0.1.0"}
INFO = {
    "model_name": "primary",
    "model_version": "test-v1",
    "trained_at": "2026-11-08T15:04:41Z",
    "git_sha": "f4c86df37c4cfc83af8b53a6c3026eecaa385341",
    "selected_by": "cv_rmse",
    "data_subset": "FD001",
    "train_sha256": "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8",
    "window_size": 30,
    "rul_cap": 125,
    "n_features": 7,
    "required_sensors": ["s_2", "s_3"],
    "feature_names": [
        "cycle",
        "s_2",
        "s_2_roll_mean",
        "s_2_roll_std",
        "s_3",
        "s_3_roll_mean",
        "s_3_roll_std",
    ],
    "health_bands": {"critical_below": 20, "warning_below": 60},
    "metrics": {
        "cv": {"mae_mean": 11.8, "mae_std": 0.7, "rmse_mean": 16.6, "rmse_std": 0.7},
        "test": {"mae": 12.6, "rmse": 16.6, "nasa_score": 441.5},
        "test_clipped_truth": {"mae": 11.5, "rmse": 15.3},
    },
    "sklearn_version_match": True,
}


def _prediction(payload: dict[str, Any]) -> dict[str, Any]:
    n = len(payload["records"])
    return {
        "unit_id": payload.get("unit_id"),
        "predicted_rul": 42.0,
        "health_status": "warning",
        "cycles_received": n,
        "cycles_used": 30,
        "last_cycle": payload["records"][-1]["cycle"],
        "model_version": "test-v1",
        "warnings": [],
    }


@pytest.fixture
def data_dir(synthetic_cmapss_dir: Path, tmp_path: Path) -> Path:
    root = tmp_path / "data"
    shutil.copytree(synthetic_cmapss_dir, root / "raw" / "CMAPSSData")
    return root


@pytest.fixture
def api_up(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(ApiClient, "health", lambda self: ApiResult(True, 200, HEALTH))
    monkeypatch.setattr(ApiClient, "model_info", lambda self: ApiResult(True, 200, INFO))

    def predict(self: ApiClient, payload: dict[str, Any]) -> ApiResult:
        sent.append(payload)
        return ApiResult(True, 200, _prediction(payload))

    monkeypatch.setattr(ApiClient, "predict", predict)
    return sent


def _metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def test_app_renders_status_and_prediction(
    data_dir: Path, api_up: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert _metric(at, "API reachable") == "yes"
    assert _metric(at, "Model loaded") == "yes"
    assert _metric(at, "Model version") == "test-v1"
    assert _metric(at, "Predicted RUL (cycles)") == "42.0"
    # default: test engine 1 (35 cycles, RUL_FD001 = 10) at its last cycle → true RUL 10
    assert _metric(at, "True RUL (cycles)") == "10"
    assert len(api_up[-1]["records"]) == 35

    at.slider[0].set_value(30).run()
    assert not at.exception
    assert _metric(at, "True RUL (cycles)") == "15"  # 10 + (35 - 30)
    assert len(api_up[-1]["records"]) == 30

    before = len(api_up)
    at.button[0].click().run()  # Compute RUL trajectory
    assert not at.exception
    # each rerun refreshes the explorer prediction (N = 30), then the trajectory calls N = 30, 35
    assert [len(p["records"]) for p in api_up[before:]] == [30, 30, 35]
    assert at.dataframe  # trajectory table rendered


def test_app_survives_api_down(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    down = ApiResult(False, None, {"error": {"code": UNREACHABLE, "message": "x", "details": []}})
    monkeypatch.setattr(ApiClient, "health", lambda self: down)
    monkeypatch.setattr(ApiClient, "model_info", lambda self: down)
    monkeypatch.setattr(ApiClient, "predict", lambda self, p: pytest.fail("predict called"))
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert _metric(at, "API reachable") == "no"
    assert _metric(at, "Predicted RUL (cycles)") == "—"
    assert any("API unreachable" in e.value for e in at.error)
    assert at.button[0].disabled


def test_app_without_dataset_keeps_status_and_model_info(
    api_up: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "nothing"))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert _metric(at, "Model loaded") == "yes"
    assert any("download_data.py" in i.value for i in at.info)
    assert at.table  # model info tables still render
    assert not api_up  # no predictions without data


def test_app_renders_interval_fields(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    info = {
        **INFO,
        "intervals": {
            "for_model_version": "test-v1",
            "method": "quantile HGB",
            "quantiles": [0.1, 0.9],
            "interval_level": 0.8,
            "trained_at": "2026-11-08T19:00:00Z",
            "cv_oof": {"n": 10, "coverage_unclipped": 0.51, "coverage_clipped": 0.74,
                       "mean_width": 37.2, "median_width": 36.7, "n_reordered": 3},
            "test": {"n": 4, "coverage_unclipped": 0.74, "coverage_clipped": 0.82,
                     "mean_width": 42.1, "median_width": 42.0, "n_reordered": 0},
        },
    }  # fmt: skip
    monkeypatch.setattr(ApiClient, "health", lambda self: ApiResult(True, 200, HEALTH))
    monkeypatch.setattr(ApiClient, "model_info", lambda self: ApiResult(True, 200, info))

    def predict(self: ApiClient, payload: dict[str, Any]) -> ApiResult:
        body = _prediction(payload) | {"rul_lower": 30.5, "rul_upper": 55.25, "interval_level": 0.8}
        return ApiResult(True, 200, body)

    monkeypatch.setattr(ApiClient, "predict", predict)
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("80% interval: 30.5 – 55.2 cycles" in c.value for c in at.caption)
    at.button[0].click().run()  # trajectory with a shaded band
    assert not at.exception
    assert {"rul_lower", "rul_upper"} <= set(at.dataframe[0].value.columns)
    assert any("Shaded: 80 % prediction interval" in c.value for c in at.caption)
    assert len(at.table) >= 3  # model card, intervals, metrics
