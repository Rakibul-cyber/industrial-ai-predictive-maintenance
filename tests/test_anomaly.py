"""Anomaly detection (ADR-035): detector behaviour, OOF threshold, evaluation, inference."""

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predmaint.data.load import COLUMNS, SENSOR_COLUMNS, SETTING_COLUMNS, load_fd001
from predmaint.features.build import build_features
from predmaint.models import anomaly as anomaly_module
from predmaint.models.anomaly import (
    SMOOTHING,
    THRESHOLD_QUANTILE,
    _spearman_or_none,
    anomaly_feature_names,
    detection_timing,
    first_detection,
    fit_detector,
    flagged_by_bucket,
    last_cycle_values,
    oof_scores,
    per_engine_trend,
    run_anomaly_training,
    smooth_per_engine,
    trend_by_true_rul,
)
from predmaint.models.predict import (
    AnomalyModel,
    AnomalyNotAvailableError,
    InputContractError,
    anomaly_path_for,
    clear_model_cache,
    detector_scores,
    ewm_smooth,
    load_anomaly_model,
    load_model,
    predict_rul,
    score_anomaly,
)

QUIET = {"verify_checksums": False, "expected": None, "echo": lambda _m: None}
SENSORS = ["s_2", "s_3", "s_4", "s_7"]
WINDOW = 30
ONSET = 100


def _engine(unit: int, n_cycles: int, rng: np.random.Generator, drift_from: int | None) -> dict:
    cycle = np.arange(1, n_cycles + 1)
    data = {"unit": np.full(n_cycles, unit), "cycle": cycle}
    for column in SETTING_COLUMNS:
        data[column] = np.zeros(n_cycles)
    for i, column in enumerate(SENSOR_COLUMNS):
        values = 100.0 + i + rng.normal(0.0, 0.5, n_cycles)
        if drift_from is not None and column in SENSORS:
            values += np.where(cycle > drift_from, 0.05 * (cycle - drift_from), 0.0)
        data[column] = values
    return data


def _synthetic_fleet(n_healthy: int = 8, seed: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Healthy engines (stationary noise) and one engine that drifts after cycle ``ONSET``."""
    rng = np.random.default_rng(seed)
    healthy = pd.concat(
        [pd.DataFrame(_engine(u, 150, rng, None)) for u in range(1, n_healthy + 1)],
        ignore_index=True,
    )[COLUMNS]
    drifting = pd.DataFrame(_engine(99, 200, rng, ONSET))[COLUMNS]
    return healthy, drifting


@pytest.fixture
def fleet() -> tuple[pd.DataFrame, pd.DataFrame]:
    healthy, drifting = _synthetic_fleet()
    return (
        build_features(healthy, SENSORS, WINDOW),
        build_features(drifting, SENSORS, WINDOW),
    )


@pytest.fixture
def models(served_model_dir: Path) -> Path:
    clear_model_cache()
    return served_model_dir / "models"


@pytest.fixture
def history(served_model_dir: Path) -> pd.DataFrame:
    test = load_fd001(served_model_dir / "CMAPSSData").test
    return test[test["unit"] == 4].reset_index(drop=True)  # 50 cycles


# --- detector on a synthetic drift engine ---------------------------------------------------
def test_score_rises_after_drift_and_detection_follows_onset(fleet) -> None:
    healthy_rows, drift_rows = fleet
    features = anomaly_feature_names(SENSORS)
    mask = np.ones(len(healthy_rows), dtype=bool)
    oof = oof_scores(healthy_rows, mask, features, n_splits=4)
    threshold = float(np.quantile(oof, THRESHOLD_QUANTILE))
    detector = fit_detector(healthy_rows, features)
    normalized = detector_scores(detector, drift_rows) / threshold
    cycles = drift_rows["cycle"].to_numpy()

    before, after = normalized[cycles <= ONSET], normalized[cycles > ONSET + WINDOW]
    assert after.mean() > 2 * before.mean()
    assert normalized.max() > 1.0  # threshold exceeded
    assert _spearman_or_none(cycles[cycles > ONSET], normalized[cycles > ONSET]) > 0.9
    detected = first_detection(cycles, normalized)
    assert detected is not None and detected > ONSET
    assert (before <= 1.0).mean() >= 0.95  # pre-onset rows mostly below threshold


def test_threshold_comes_from_out_of_fold_scores(fleet) -> None:
    healthy_rows, _ = fleet
    features = anomaly_feature_names(SENSORS)
    mask = np.ones(len(healthy_rows), dtype=bool)
    oof = float(np.quantile(oof_scores(healthy_rows, mask, features, 4), THRESHOLD_QUANTILE))
    detector = fit_detector(healthy_rows, features)
    in_sample = float(np.quantile(detector_scores(detector, healthy_rows), THRESHOLD_QUANTILE))
    assert in_sample < oof  # in-sample scoring is optimistic → a lower threshold


def test_served_fixture_records_oof_threshold(models: Path) -> None:
    meta = json.loads((models / "model_anomaly_metadata.json").read_text())
    check = meta["evaluation"]["threshold_check"]
    assert meta["threshold"] == check["oof_threshold"]
    assert check["in_sample_threshold_all_healthy_rows"] < check["oof_threshold"]
    assert meta["threshold_quantile"] == THRESHOLD_QUANTILE
    assert (
        load_anomaly_model(anomaly_path_for(models / "model.joblib")).threshold
        == (meta["threshold"])
    )


def test_isolation_forest_branch_and_unknown_method(fleet) -> None:
    healthy_rows, drift_rows = fleet
    features = anomaly_feature_names(SENSORS)
    detector = fit_detector(healthy_rows, features, "isolation_forest")
    scores = detector_scores(detector, drift_rows)
    assert scores.shape == (len(drift_rows),) and np.isfinite(scores).all()
    with pytest.raises(ValueError, match="unknown method"):
        fit_detector(healthy_rows, features, "autoencoder")
    with pytest.raises(ValueError, match="unknown anomaly method"):
        detector_scores({**detector, "method": "other"}, drift_rows)


# --- evaluation helpers -------------------------------------------------------------------
def test_first_detection_needs_three_consecutive_cycles() -> None:
    cycles = np.arange(31, 41)
    scores = np.array([0.5, 1.2, 1.3, 0.9, 1.1, 1.2, 1.5, 2.0, 0.2, 3.0])
    assert first_detection(cycles, scores) == 35
    assert first_detection(cycles, np.full(10, 0.99)) is None
    assert first_detection(cycles, np.full(10, 1.0)) is None  # strictly greater than 1


def test_spearman_or_none_handles_constant_and_short_inputs() -> None:
    assert _spearman_or_none(np.array([1.0, 2.0, 3.0]), np.array([3.0, 3.0, 3.0])) is None
    assert _spearman_or_none(np.array([1.0, 2.0]), np.array([1.0, 2.0])) is None
    assert _spearman_or_none(np.arange(5.0), np.arange(5.0) ** 2) == pytest.approx(1.0)


def test_per_engine_trend_and_timing_hand_values() -> None:
    rows = pd.DataFrame(
        {
            "unit": [1] * 5 + [2] * 5 + [3] * 2,
            "cycle": [31, 32, 33, 34, 35] * 2 + [31, 32],
            "rul_raw": [4, 3, 2, 1, 0, 9, 8, 7, 6, 5, 1, 0],
        }
    )
    normalized = np.array([0.1, 0.2, 1.1, 1.2, 1.3, 0.5, 0.4, 0.3, 0.2, 0.1, 0.7, 0.7])
    trend = per_engine_trend(rows, normalized)
    assert trend["n_engines"] == 3 and trend["n_engines_scored"] == 2  # unit 3 is too short
    assert trend["median"] == pytest.approx(0.0)  # +1 and -1
    assert trend["share_above_0_7"] == 0.5
    timing = detection_timing(rows, normalized)
    assert timing["never_flagged_fraction"] == pytest.approx(2 / 3)
    assert timing["fraction_of_life"]["median"] == pytest.approx(33 / 35)
    assert timing["true_rul_at_detection"]["median"] == 2


def test_flagged_by_bucket_hand_values() -> None:
    truth = np.array([5.0, 20.0, 21.0, 60.0, 100.0, 140.0])
    normalized = np.array([1.5, 0.5, 1.1, 0.9, 0.2, 0.1])
    buckets = {b["true_rul_bucket"]: b for b in flagged_by_bucket(truth, normalized)}
    assert buckets["0-20"] == {"true_rul_bucket": "0-20", "n": 2, "flagged_fraction": 0.5}
    assert buckets["21-60"]["flagged_fraction"] == 0.5
    assert buckets["61-125"]["n"] == 1 and buckets[">125"]["flagged_fraction"] == 0.0
    empty = flagged_by_bucket(np.array([5.0]), np.array([2.0]))
    assert {b["true_rul_bucket"]: b["flagged_fraction"] for b in empty}[">125"] is None


# --- training -------------------------------------------------------------------------------
def test_training_is_deterministic_and_leaves_other_models_untouched(
    served_model_dir: Path, tmp_path: Path
) -> None:
    models = tmp_path / "models"
    shutil.copytree(served_model_dir / "models", models)
    others = ("model.joblib", "model_metadata.json", "model_intervals.joblib")
    before = {name: (models / name).read_bytes() for name in others}
    clear_model_cache()
    run = {**QUIET, "healthy_min_rul": 20}
    first = run_anomaly_training(served_model_dir / "CMAPSSData", models, tmp_path / "r", **run)
    second = run_anomaly_training(served_model_dir / "CMAPSSData", models, tmp_path / "r", **run)
    assert {name: (models / name).read_bytes() for name in others} == before
    assert first["evaluation"] == second["evaluation"]
    assert first["for_model_version"] == json.loads(before["model_metadata.json"])["model_version"]
    report = json.loads((tmp_path / "r" / "metrics_anomaly.json").read_text())
    assert report["evaluation"] == json.loads(json.dumps(first["evaluation"]))
    assert report["evaluation"]["acceptance"]["met"] is True  # synthetic drift is monotonic


def test_fallback_is_tried_at_most_once(
    served_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "models"
    shutil.copytree(served_model_dir / "models", models)
    clear_model_cache()
    monkeypatch.setattr(anomaly_module, "FALLBACK_BELOW", 1.1)  # force the fallback path
    meta = run_anomaly_training(
        served_model_dir / "CMAPSSData", models, tmp_path, healthy_min_rul=20, **QUIET
    )
    fallback = meta["evaluation"]["fallback"]
    assert set(fallback["candidates"]) == {"mahalanobis_ledoitwolf", "isolation_forest"}
    assert "tried once" in fallback["decision"]
    assert meta["method"] in fallback["candidates"]


def test_training_refuses_without_served_model(served_model_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="--select"):
        run_anomaly_training(served_model_dir / "CMAPSSData", tmp_path, tmp_path, **QUIET)


def test_training_refuses_without_healthy_rows(served_model_dir: Path, tmp_path: Path) -> None:
    models = tmp_path / "models"
    shutil.copytree(served_model_dir / "models", models)
    clear_model_cache()
    with pytest.raises(ValueError, match="no healthy reference rows"):
        run_anomaly_training(served_model_dir / "CMAPSSData", models, tmp_path, **QUIET)


def test_training_refuses_other_data(served_model_dir: Path, tmp_path: Path) -> None:
    models, data = tmp_path / "models", tmp_path / "data"
    shutil.copytree(served_model_dir / "models", models)
    shutil.copytree(served_model_dir / "CMAPSSData", data)
    (data / "RUL_FD001.txt").write_text("41 \n30 \n20 \n10 \n")
    clear_model_cache()
    with pytest.raises(ValueError, match="data differ"):
        run_anomaly_training(data, models, tmp_path, healthy_min_rul=20, **QUIET)


# --- inference ------------------------------------------------------------------------------
def _enabled(detector: AnomalyModel) -> AnomalyModel:
    """The same detector with smoothing served (ADR-038 behaviour, disabled by ADR-039)."""
    return AnomalyModel(detector.detector, {**detector.metadata, "smoothing_enabled": True})


def test_served_score_is_raw_with_smoothing_disabled(models: Path, history: pd.DataFrame) -> None:
    """ADR-039: the metadata keeps the smoothing config but serves the raw M9 score."""
    model = load_model(models / "model.joblib")
    detector = load_anomaly_model(anomaly_path_for(models / "model.joblib"))
    assert detector.smoothing is not None and detector.smoothing_enabled is False
    served = score_anomaly(history, model, detector, include_series=True)
    assert served.smoothed is False and served.cycles_in_smoothing is None
    assert served.threshold == detector.threshold
    assert served.normalized_score == served.raw_normalized_score
    assert all(p["normalized_score"] == p["raw_normalized_score"] for p in served.series)
    # raw score: the last-cycle value depends only on the last window, like predict_rul
    tail = score_anomaly(history.tail(30), model, detector)
    assert tail.anomaly_score == pytest.approx(served.anomaly_score)
    assert tail.is_anomalous is served.is_anomalous


def test_smoothing_enabled_flag_semantics(models: Path) -> None:
    detector = load_anomaly_model(anomaly_path_for(models / "model.joblib"))
    no_flag = {k: v for k, v in detector.metadata.items() if k != "smoothing_enabled"}
    assert AnomalyModel(detector.detector, no_flag).smoothing_enabled is True  # M11 metadata
    m9 = {k: v for k, v in no_flag.items() if k not in ("smoothing", "smoothed")}
    assert AnomalyModel(detector.detector, m9).smoothing_enabled is False
    assert _enabled(detector).smoothing_enabled is True


def test_score_anomaly_series_and_last_value(models: Path, history: pd.DataFrame) -> None:
    model = load_model(models / "model.joblib")
    detector = _enabled(load_anomaly_model(anomaly_path_for(models / "model.joblib")))
    plain = score_anomaly(history, model, detector)
    full = score_anomaly(history, model, detector, include_series=True)
    assert plain.series is None
    assert len(full.series) == len(history) - model.window_size + 1
    assert [p["cycle"] for p in full.series] == list(range(30, 51))
    assert full.series[-1]["normalized_score"] == pytest.approx(full.normalized_score)
    assert full.series[-1]["raw_normalized_score"] == pytest.approx(full.raw_normalized_score)
    assert plain.normalized_score == pytest.approx(plain.anomaly_score / plain.threshold)
    assert plain.is_anomalous is (plain.normalized_score > 1)
    assert (plain.last_cycle, plain.cycles_received) == (50, 50)
    assert plain.model_version == detector.model_version
    assert plain.smoothed is True and plain.cycles_in_smoothing == 21
    assert plain.threshold == detector.smoothed_threshold
    raw_norm = [p["raw_normalized_score"] for p in full.series]
    raw = np.array(raw_norm) * detector.threshold
    smoothed = ewm_smooth(raw, SMOOTHING["halflife"]) / detector.smoothed_threshold
    assert [p["normalized_score"] for p in full.series] == pytest.approx(smoothed.tolist())


def test_history_dependence_raw_equal_smoothed_may_differ(
    models: Path, history: pd.DataFrame
) -> None:
    """ADR-038: the raw last-cycle score depends only on the last window; the smoothed one on
    every full-window cycle sent (reported as ``cycles_in_smoothing``)."""
    model = load_model(models / "model.joblib")
    detector = _enabled(load_anomaly_model(anomaly_path_for(models / "model.joblib")))
    full, tail, window = (
        score_anomaly(h, model, detector) for h in (history, history.tail(35), history.tail(30))
    )
    assert full.raw_normalized_score == pytest.approx(tail.raw_normalized_score)
    assert full.raw_normalized_score == pytest.approx(window.raw_normalized_score)
    assert (full.cycles_in_smoothing, tail.cycles_in_smoothing) == (21, 6)
    assert window.cycles_in_smoothing == 1
    # one smoothed cycle: EWM == raw, so the smoothed score equals the raw distance
    assert window.anomaly_score == pytest.approx(window.raw_normalized_score * detector.threshold)


def test_unsmoothed_metadata_behaves_like_m9(models: Path, history: pd.DataFrame) -> None:
    model = load_model(models / "model.joblib")
    detector = load_anomaly_model(anomaly_path_for(models / "model.joblib"))
    m9_meta = {k: v for k, v in detector.metadata.items() if k not in ("smoothing", "smoothed")}
    m9 = AnomalyModel(detector.detector, m9_meta)
    assert m9.smoothing is None and m9.smoothed_threshold is None
    result = score_anomaly(history, model, m9, include_series=True)
    assert result.smoothed is False and result.cycles_in_smoothing is None
    assert result.threshold == detector.threshold
    assert result.normalized_score == pytest.approx(result.raw_normalized_score)
    assert all(p["normalized_score"] == p["raw_normalized_score"] for p in result.series)


def test_unsupported_smoothing_config_is_unavailable(models: Path, history: pd.DataFrame) -> None:
    model = load_model(models / "model.joblib")
    detector = load_anomaly_model(anomaly_path_for(models / "model.joblib"))
    bad = AnomalyModel(
        detector.detector, {**detector.metadata, "smoothing": {"method": "median", "halflife": 10}}
    )
    with pytest.raises(AnomalyNotAvailableError, match="smoothing"):
        score_anomaly(history, model, bad)


def test_mismatched_detector_is_unavailable(models: Path, history: pd.DataFrame) -> None:
    model = load_model(models / "model.joblib")
    detector = load_anomaly_model(anomaly_path_for(models / "model.joblib"))
    stale = AnomalyModel(detector.detector, {**detector.metadata, "for_model_version": "old"})
    with pytest.raises(AnomalyNotAvailableError, match="for model 'old'"):
        score_anomaly(history, model, stale)
    other = AnomalyModel({**detector.detector, "features": ["s_99_roll_mean"]}, detector.metadata)
    with pytest.raises(AnomalyNotAvailableError, match="not produced"):
        score_anomaly(history, model, other)


def _break(history: pd.DataFrame, case: str) -> pd.DataFrame:
    broken = history.copy()
    if case == "short":
        return broken.head(10)
    if case == "missing":
        broken.loc[5, "s_2"] = np.nan
    elif case == "gap":
        broken.loc[10:, "cycle"] += 1
    elif case == "negative_cycle":
        broken.loc[0, "cycle"] = -1
    return broken


@pytest.mark.parametrize("case", ["short", "missing", "gap", "negative_cycle"])
def test_validation_errors_identical_to_predict(
    models: Path, history: pd.DataFrame, case: str
) -> None:
    model = load_model(models / "model.joblib")
    detector = load_anomaly_model(anomaly_path_for(models / "model.joblib"))
    bad = _break(history, case)
    with pytest.raises(InputContractError) as from_predict:
        predict_rul(bad, model)
    with pytest.raises(InputContractError) as from_anomaly:
        score_anomaly(bad, model, detector)
    assert type(from_anomaly.value) is type(from_predict.value)
    assert str(from_anomaly.value) == str(from_predict.value)


def test_load_anomaly_model_errors(
    models: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with pytest.raises(AnomalyNotAvailableError):
        load_anomaly_model(tmp_path / "model_anomaly.joblib")
    shutil.copytree(models, tmp_path / "m")
    meta_path = tmp_path / "m" / "model_anomaly_metadata.json"
    meta = json.loads(meta_path.read_text())
    del meta["for_model_version"]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(AnomalyNotAvailableError, match="for_model_version"):
        load_anomaly_model(tmp_path / "m" / "model_anomaly.joblib")
    (tmp_path / "m" / "model_anomaly.joblib").write_bytes(b"not a pickle")
    clear_model_cache()
    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(AnomalyNotAvailableError, match="could not load"),
    ):
        load_anomaly_model(tmp_path / "m" / "model_anomaly.joblib")


# --- smoothing (ADR-038, M11) -----------------------------------------------------------------
def test_ewm_smooth_hand_values() -> None:
    """halflife = 1 → decay 0.5: y_t = sum(0.5**i * x_(t-i)) / sum(0.5**i)."""
    smoothed = ewm_smooth(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), halflife=1)
    expected = [1.0, 2.5 / 1.5, 4.25 / 1.75, 6.125 / 1.875, 8.0625 / 1.9375]
    assert smoothed.tolist() == pytest.approx(expected, abs=1e-12)
    assert SMOOTHING == {"method": "ewm", "halflife": 10, "adjust": True}


def test_smoothing_on_synthetic_drift(fleet) -> None:
    healthy_rows, drift_rows = fleet
    features = anomaly_feature_names(SENSORS)
    mask = np.ones(len(healthy_rows), dtype=bool)
    oof = oof_scores(healthy_rows, mask, features, n_splits=4)
    oof_smoothed = smooth_per_engine(healthy_rows, oof, SMOOTHING["halflife"])
    raw_threshold = float(np.quantile(oof, THRESHOLD_QUANTILE))
    smoothed_threshold = float(np.quantile(oof_smoothed, THRESHOLD_QUANTILE))
    assert smoothed_threshold < raw_threshold  # smoothing shrinks the upper tail (direction only)

    detector = fit_detector(healthy_rows, features)
    raw = detector_scores(detector, drift_rows)
    smoothed = ewm_smooth(raw, SMOOTHING["halflife"])
    cycles = drift_rows["cycle"].to_numpy()
    pre = cycles <= ONSET
    assert np.var(smoothed[pre]) < np.var(raw[pre])
    smoothed_norm, raw_norm = smoothed / smoothed_threshold, raw / raw_threshold
    assert smoothed_norm.max() > 1.0
    assert cycles[np.argmax(smoothed_norm > 1.0)] > ONSET  # first crossing after onset
    raw_first, smoothed_first = (
        first_detection(cycles, raw_norm),
        first_detection(cycles, smoothed_norm),
    )
    assert raw_first is not None and smoothed_first is not None
    assert smoothed_first >= raw_first > ONSET


def test_smooth_per_engine_respects_engines_and_cycle_order() -> None:
    rows = pd.DataFrame({"unit": [2, 1, 2, 1, 2], "cycle": [31, 31, 30, 30, 32]})
    scores = np.array([20.0, 2.0, 10.0, 1.0, 30.0])
    out = smooth_per_engine(rows, scores, halflife=1)
    # unit 1: [1, 2] in cycle order; unit 2: [10, 20, 30]
    assert out[3] == 1.0 and out[1] == pytest.approx(2.5 / 1.5)
    assert out[2] == 10.0 and out[0] == pytest.approx(25 / 1.5)
    assert out[4] == pytest.approx((30 + 10 + 2.5) / 1.75)
    assert last_cycle_values(rows, scores).tolist() == [2.0, 30.0]


def test_trend_by_true_rul_groups() -> None:
    rows = pd.DataFrame({"unit": [1] * 4 + [2] * 4 + [3] * 4, "cycle": [30, 31, 32, 33] * 3})
    normalized = np.array([1, 2, 3, 4, 4, 3, 2, 1, 1, 3, 2, 4], dtype=float)
    groups = trend_by_true_rul(rows, normalized, pd.Series({1: 10.0, 2: 100.0, 3: 140.0}))
    assert [g["true_rul_group"] for g in groups] == ["<=60", "61-125", ">125"]
    assert [g["median"] for g in groups] == pytest.approx([1.0, -1.0, 0.8])
    assert [g["n_engines_scored"] for g in groups] == [1, 1, 1]


def test_served_fixture_records_smoothing(models: Path) -> None:
    meta = json.loads((models / "model_anomaly_metadata.json").read_text())
    assert meta["smoothing"] == SMOOTHING
    sm = meta["smoothed"]
    assert sm["threshold"] == sm["threshold_check"]["oof_threshold"]
    assert sm["threshold_check"]["raw_oof_threshold"] == meta["threshold"]
    acc = sm["evaluation"]["acceptance"]
    assert acc["met"] is (acc["trend_met"] and acc["false_alarm_met"])
    assert set(sm["evaluation"]["test"]) >= {"trend", "trend_by_true_rul", "flagged_at_last_cycle"}
    assert "trend_by_true_rul" in meta["evaluation"]["test"]
