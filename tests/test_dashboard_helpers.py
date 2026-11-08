"""Tests for dashboard helpers: ground truth, payloads, trajectory sampling and computation."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from predmaint.dashboard.api_client import ApiResult
from predmaint.dashboard.helpers import (
    TrajectoryError,
    anomaly_badge,
    anomaly_caption,
    anomaly_chart_caption,
    anomaly_info_table,
    anomaly_series_frame,
    band_caption,
    compute_trajectory,
    engine_history,
    engine_payload,
    interval_band,
    interval_coverage_table,
    interval_text,
    sensor_chart_frame,
    trajectory_cycles,
    true_rul,
    true_rul_series,
    units,
)
from predmaint.data.load import load_fd001


@pytest.fixture
def data(synthetic_cmapss_dir: Path) -> Any:
    # train units: 40, 45, 50 cycles; test units: 35, 32, 38 cycles; RUL_FD001 = [10, 20, 30]
    return load_fd001(synthetic_cmapss_dir)


def test_true_rul_hand_computed() -> None:
    assert true_rul(0, 200, 30) == 170  # train: max_cycle - n
    assert true_rul(0, 200, 200) == 0
    assert true_rul(7, 203, 203) == 7  # test engine 34: RUL_FD001 at its last cycle
    assert true_rul(7, 203, 100) == 110  # 7 + (203 - 100)
    with pytest.raises(ValueError):
        true_rul(0, 50, 51)


def test_engine_history_truth_for_train_and_test(data: Any) -> None:
    train = engine_history(data, "train", 2)
    assert train.last_cycle == 45 and train.rul_at_last_cycle == 0
    assert train.true_rul(30) == 15
    test = engine_history(data, "test", 3)
    assert test.last_cycle == 38 and test.rul_at_last_cycle == 30
    assert test.true_rul(30) == 38  # 30 + (38 - 30)
    assert units(data, "test") == [1, 2, 3]
    with pytest.raises(ValueError):
        engine_history(data, "validation", 1)
    with pytest.raises(ValueError):
        engine_history(data, "train", 99)


def test_engine_payload_first_n_cycles(data: Any) -> None:
    payload = engine_payload(engine_history(data, "test", 1), 31)
    assert payload["unit_id"] == "test-unit-1"
    assert [r["cycle"] for r in payload["records"]] == list(range(1, 32))
    assert set(payload["records"][0]) == {"cycle", "setting_1", "setting_2", "setting_3"} | {
        f"s_{i}" for i in range(1, 22)
    }


def test_trajectory_cycles() -> None:
    assert trajectory_cycles(45, 30, 5) == [30, 35, 40, 45]
    assert trajectory_cycles(47, 30, 5) == [30, 35, 40, 45, 47]
    assert trajectory_cycles(30, 30, 5) == [30]
    assert trajectory_cycles(20, 30, 5) == []
    with pytest.raises(ValueError):
        trajectory_cycles(50, 30, 0)


def test_true_rul_series_caps_like_training(data: Any) -> None:
    series = true_rul_series(engine_history(data, "train", 3), [30, 40, 50], rul_cap=15)
    assert series["true_rul"].tolist() == [20, 10, 0]
    assert series["true_rul_capped"].tolist() == [15, 10, 0]


def _fake_predict(sent: list[int]) -> Callable[[dict[str, Any]], ApiResult]:
    def predict(payload: dict[str, Any]) -> ApiResult:
        n = len(payload["records"])
        sent.append(n)
        return ApiResult(True, 200, {"predicted_rul": 100.0 - n, "health_status": "healthy"})

    return predict


def test_compute_trajectory(data: Any) -> None:
    sent: list[int] = []
    frame = compute_trajectory(engine_history(data, "train", 2), _fake_predict(sent), 30, 125, 5)
    assert sent == [30, 35, 40, 45]
    assert list(frame.columns) == [
        "cycle",
        "predicted_rul",
        "health_status",
        "rul_lower",
        "rul_upper",
        "true_rul",
        "true_rul_capped",
    ]
    assert frame["rul_lower"].isna().all()  # the fake API sends no interval fields
    assert interval_band(frame) is None
    assert frame["predicted_rul"].tolist() == [70.0, 65.0, 60.0, 55.0]
    assert frame["true_rul"].tolist() == [15, 10, 5, 0]


def test_compute_trajectory_raises_on_api_error(data: Any) -> None:
    envelope = {"error": {"code": "MODEL_UNAVAILABLE", "message": "m", "details": []}}

    def failing(_payload: dict[str, Any]) -> ApiResult:
        return ApiResult(False, 503, envelope)

    with pytest.raises(TrajectoryError) as err:
        compute_trajectory(engine_history(data, "train", 2), failing, 30, 125, 5)
    assert err.value.cycle == 30 and err.value.body == envelope


def test_sensor_chart_frame_is_long_and_standardised(data: Any) -> None:
    history = engine_history(data, "train", 1)
    frame = sensor_chart_frame(history, ["s_1", "s_2"])  # s_1 is constant in the fixture
    assert list(frame.columns) == ["cycle", "sensor", "standardised value"]
    assert len(frame) == 2 * history.last_cycle
    s_1 = frame[frame["sensor"] == "s_1"]["standardised value"]
    s_2 = frame[frame["sensor"] == "s_2"]["standardised value"]
    assert (s_1 == 0).all()
    assert abs(s_2.mean()) < 1e-9 and abs(s_2.std(ddof=0) - 1) < 1e-9
    assert isinstance(frame, pd.DataFrame)


def test_compute_trajectory_carries_interval_bounds(data: Any) -> None:
    def predict(payload: dict[str, Any]) -> ApiResult:
        n = len(payload["records"])
        body = {"predicted_rul": 100.0 - n, "health_status": "healthy"}
        body |= {"rul_lower": 90.0 - n, "rul_upper": 110.0 - n, "interval_level": 0.8}
        return ApiResult(True, 200, body)

    frame = compute_trajectory(engine_history(data, "train", 2), predict, 30, 125, 5)
    band = interval_band(frame)
    assert band is not None and list(band.columns) == ["cycle", "rul_lower", "rul_upper"]
    assert band["rul_lower"].tolist() == [60.0, 55.0, 50.0, 45.0]
    assert (band["rul_lower"] <= frame["predicted_rul"]).all()


def test_interval_band_partial_and_missing_columns() -> None:
    frame = pd.DataFrame(
        {"cycle": [30, 35], "rul_lower": [1.0, float("nan")], "rul_upper": [5.0, float("nan")]}
    )
    assert interval_band(frame)["cycle"].tolist() == [30]
    assert interval_band(pd.DataFrame({"cycle": [30]})) is None


def test_interval_text_handles_nulls() -> None:
    body = {"rul_lower": 4.84, "rul_upper": 18.76, "interval_level": 0.8}
    assert interval_text(body) == "80% interval: 4.8 – 18.8 cycles"
    assert interval_text({"rul_lower": None, "rul_upper": None, "interval_level": None}) is None
    assert interval_text({}) is None
    assert interval_text({"rul_lower": 1.0, "rul_upper": 2.0}) == "Interval: 1.0 – 2.0 cycles"


def test_anomaly_badge_and_nulls() -> None:
    assert anomaly_badge({"is_anomalous": True}) == ("ANOMALOUS", "red")
    assert anomaly_badge({"is_anomalous": False}) == ("NORMAL", "green")
    assert anomaly_badge(None) is None
    assert anomaly_badge({}) is None
    assert anomaly_badge({"is_anomalous": None}) is None


def test_anomaly_series_frame() -> None:
    body = {
        "series": [{"cycle": 30, "normalized_score": 0.5}, {"cycle": 31, "normalized_score": 1.2}]
    }
    frame = anomaly_series_frame(body)
    assert list(frame.columns) == ["cycle", "normalized_score"]
    assert frame["cycle"].tolist() == [30, 31] and frame["normalized_score"].tolist() == [0.5, 1.2]
    assert anomaly_series_frame(None) is None
    assert anomaly_series_frame({"series": None}) is None
    assert anomaly_series_frame({"series": []}) is None


def test_band_caption_names_conformal_q() -> None:
    calibrated = {"calibrated": True, "conformal_q": 1.7418749}
    assert band_caption(calibrated) == (
        "Shaded: 80 % prediction interval (conformally calibrated, Q = 1.74). "
    )
    plain = "Shaded: 80 % prediction interval. "
    assert band_caption(None) == plain
    assert band_caption({"calibrated": False, "conformal_q": None}) == plain
    assert band_caption({"calibrated": True, "conformal_q": None}) == plain


def _cov(clipped: float, unclipped: float, width: float) -> dict[str, float]:
    return {"coverage_clipped": clipped, "coverage_unclipped": unclipped, "mean_width": width}


def test_interval_coverage_table_calibrated_and_fallback() -> None:
    calibrated = {
        "conformal_q": 1.7418749,
        "cv_oof": _cov(0.737, 0.509, 37.2),
        "test": _cov(0.82, 0.74, 42.1),
        "calibration": {
            "evaluation_half_after": _cov(0.7768, 0.536, 39.19),
            "evaluation_half_before": _cov(0.7339, 0.504, 37.15),
            "test_after": _cov(0.85, 0.75, 44.74),
            "test_before": _cov(0.82, 0.74, 42.13),
        },
    }
    table = interval_coverage_table(calibrated)
    assert list(table.index) == [
        "Calibrated (CQR, Q = 1.742)",
        "Uncalibrated (M8, for comparison)",
    ]
    assert table.iloc[0].tolist() == ["0.777", "39.2", "0.850", "0.750", "44.7"]
    assert table.iloc[1].tolist() == ["0.734", "37.1", "0.820", "0.740", "42.1"]
    old = interval_coverage_table({**calibrated, "calibration": None})
    assert list(old.index) == ["CV out-of-fold", "Official test"]
    assert old.loc["CV out-of-fold", "Coverage (clipped truth)"] == "0.737"


def test_anomaly_series_frame_with_raw_column() -> None:
    body = {
        "smoothed": True,
        "series": [
            {"cycle": 30, "normalized_score": 0.7, "raw_normalized_score": 0.66},
            {"cycle": 31, "normalized_score": 0.72, "raw_normalized_score": 0.9},
        ],
    }
    frame = anomaly_series_frame(body)
    assert list(frame.columns) == ["cycle", "normalized_score", "raw_normalized_score"]
    assert frame["raw_normalized_score"].tolist() == [0.66, 0.9]
    no_raw = {"series": [{"cycle": 30, "normalized_score": 0.7, "raw_normalized_score": None}]}
    assert list(anomaly_series_frame(no_raw).columns) == ["cycle", "normalized_score"]
    disabled = {**body, "smoothed": False}  # ADR-039: both columns are the raw score
    assert list(anomaly_series_frame(disabled).columns) == ["cycle", "normalized_score"]


def test_anomaly_chart_caption_states_what_is_served() -> None:
    assert anomaly_chart_caption({"smoothed": True}).startswith("Bold: EWM-smoothed")
    disabled = anomaly_chart_caption({"smoothed": False})
    assert "as served" in disabled and "evaluated but is not served" in disabled
    assert anomaly_chart_caption({}).startswith("Normalised anomaly score")


def test_anomaly_caption_smoothed_and_raw() -> None:
    smoothed = {
        "normalized_score": 1.0472,
        "smoothed": True,
        "cycles_in_smoothing": 174,
        "raw_normalized_score": 1.1015,
    }
    assert anomaly_caption(smoothed) == (
        "Anomaly score 1.05 (smoothed over 174 cycles; raw 1.10; > 1 = anomalous)"
    )
    assert anomaly_caption({"normalized_score": 0.6657}) == (
        "Anomaly score 0.67 (normalised; > 1 = anomalous)"
    )


def _evaluation(trend: float, false_alarm: float, never: float) -> dict[str, Any]:
    return {
        "train_oof": {
            "trend": {"median": trend},
            "false_alarm_rate_healthy": false_alarm,
            "detection": {
                "fraction_of_life": {"median": 0.758},
                "true_rul_at_detection": {"median": 49.0},
                "never_flagged_fraction": never,
            },
        }
    }


def test_anomaly_info_table_side_by_side_and_fallback() -> None:
    anomaly = {
        "threshold": 5.4036,
        "acceptance_met": False,
        "smoothing": {"method": "ewm", "halflife": 10, "adjust": True},
        "smoothing_enabled": False,
        "smoothed_threshold": 5.0906,
        "smoothed_acceptance_met": False,
        "evaluation_raw": _evaluation(0.680, 0.0101, 0.02),
        "evaluation_smoothed": _evaluation(0.693, 0.0101, 0.16),
    }
    table = anomaly_info_table(anomaly)
    assert list(table.columns) == [
        "raw score (M9, served)",
        "EWM half-life 10 (smoothed: evaluated, not served)",
    ]
    enabled = anomaly_info_table({**anomaly, "smoothing_enabled": True})
    assert list(enabled.columns) == ["raw score (M9)", "EWM half-life 10 (served)"]
    assert table.loc["Trend median (train OOF)"].tolist() == ["0.680", "0.693"]
    assert table.loc["Train engines never flagged"].tolist() == ["2%", "16%"]
    assert table.loc["Acceptance (median trend > 0.7)"].tolist() == [
        "FR-019: NOT met",
        "FR-021: NOT met",
    ]
    m9 = {
        "threshold": 5.4036,
        "acceptance_met": False,
        "trend_median_train_oof": None,
        "false_alarm_rate_healthy": 0.0101,
        "detection_fraction_of_life_median": None,
        "detection_true_rul_median": None,
        "never_flagged_fraction": 0.02,
    }
    old = anomaly_info_table(m9)
    assert list(old.columns) == ["raw score"]
    assert old.loc["Trend median (train OOF)", "raw score"] == "—"
    assert old.loc["Median detection", "raw score"] == "—"
