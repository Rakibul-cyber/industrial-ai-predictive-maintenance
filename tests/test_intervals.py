"""Prediction intervals (ADR-034): training, invariants, and inference with/without intervals."""

import dataclasses
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predmaint.data.load import load_fd001
from predmaint.models.artifacts import load_artifacts
from predmaint.models.intervals import (
    ALPHA,
    QUANTILES,
    calibrate,
    conformal_quantile,
    engine_split,
    nonconformity,
    run_interval_training,
    summarize_intervals,
)
from predmaint.models.predict import (
    IntervalModel,
    clear_model_cache,
    interval_path_for,
    load_interval_model,
    load_model,
    order_bounds,
    predict_rul,
)

QUIET = {"verify_checksums": False, "expected": None, "echo": lambda _m: None}


@pytest.fixture
def models(served_model_dir: Path) -> Path:
    clear_model_cache()
    return served_model_dir / "models"


@pytest.fixture
def history(served_model_dir: Path) -> pd.DataFrame:
    test = load_fd001(served_model_dir / "CMAPSSData").test
    return test[test["unit"] == 4].reset_index(drop=True)


# --- training ---------------------------------------------------------------------------------
def test_interval_artifact_trained_for_served_model(models: Path) -> None:
    pipelines, meta = load_artifacts(models, "model_intervals")
    served = json.loads((models / "model_metadata.json").read_text())
    assert set(pipelines) == {"lower", "upper"}
    assert meta["for_model_version"] == served["model_version"]
    assert meta["quantiles"] == list(QUANTILES) and meta["interval_level"] == 0.8
    assert meta["hyperparameters"] == served["hyperparameters"]
    assert meta["feature_names"] == served["feature_names"]
    assert meta["config"] == served["config"]
    for q, name in zip(QUANTILES, ("lower", "upper"), strict=True):
        params = pipelines[name].named_steps["model"].get_params()
        assert params["loss"] == "quantile" and params["quantile"] == q
        assert params["early_stopping"] is False
    for split in ("cv_oof", "test"):
        overall = meta["evaluation"][split]["overall"]
        assert 0 <= overall["coverage_unclipped"] <= 1 and overall["mean_width"] >= 0
        assert [b["true_rul_bucket"] for b in meta["evaluation"][split]["buckets"]] == [
            "0-20",
            "21-60",
            "61-125",
            ">125",
        ]


def test_retraining_intervals_leaves_point_model_untouched(
    served_model_dir: Path, tmp_path: Path
) -> None:
    models = tmp_path / "models"
    shutil.copytree(served_model_dir / "models", models)
    before = (models / "model.joblib").read_bytes(), (models / "model_metadata.json").read_bytes()
    clear_model_cache()
    first = run_interval_training(served_model_dir / "CMAPSSData", models, tmp_path / "r", **QUIET)
    second = run_interval_training(served_model_dir / "CMAPSSData", models, tmp_path / "r", **QUIET)
    after = (models / "model.joblib").read_bytes(), (models / "model_metadata.json").read_bytes()
    assert before == after
    assert first["evaluation"] == second["evaluation"]  # deterministic
    assert (tmp_path / "r" / "metrics_intervals.json").is_file()


def test_refuses_without_served_model(served_model_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="--select"):
        run_interval_training(served_model_dir / "CMAPSSData", tmp_path, tmp_path, **QUIET)


def test_refuses_when_baseline_is_served(served_model_dir: Path, tmp_path: Path) -> None:
    models = tmp_path / "models"
    shutil.copytree(served_model_dir / "models", models)
    meta = json.loads((models / "baseline_metadata.json").read_text())
    shutil.copyfile(models / "baseline.joblib", models / "model.joblib")
    (models / "model_metadata.json").write_text(json.dumps({**meta, "model_version": "b-1"}))
    clear_model_cache()
    with pytest.raises(ValueError, match="hyperparameters"):
        run_interval_training(served_model_dir / "CMAPSSData", models, tmp_path, **QUIET)


# --- invariants -------------------------------------------------------------------------------
def test_order_bounds_invariants_on_random_inputs() -> None:
    rng = np.random.default_rng(0)
    point = np.clip(rng.normal(60, 50, 5000), 0, 125)
    lower = point + rng.normal(-10, 25, 5000)  # deliberately crosses the point sometimes
    upper = point + rng.normal(10, 25, 5000)
    lo, hi, n = order_bounds(point, lower, upper, 125)
    assert (lo >= 0).all() and (hi <= 125).all()
    assert (lo <= point).all() and (point <= hi).all()
    expected = np.sum((np.clip(lower, 0, 125) > point) | (np.clip(upper, 0, 125) < point))
    assert n == expected > 0


def test_order_bounds_hand_cases() -> None:
    lo, hi, n = order_bounds(
        np.array([50.0, 50.0, 0.0, 125.0]),
        np.array([40.0, 60.0, -5.0, 130.0]),  # ok, crossed, below 0, above cap
        np.array([70.0, 55.0, 3.0, 140.0]),
        125,
    )
    assert lo.tolist() == [40.0, 50.0, 0.0, 125.0]
    assert hi.tolist() == [70.0, 55.0, 3.0, 125.0]
    assert n == 1  # only the crossed lower bound moved to the point


def test_summarize_intervals_hand_values() -> None:
    truth = np.array([10.0, 30.0, 100.0, 140.0])
    lower = np.array([5.0, 35.0, 90.0, 110.0])
    upper = np.array([15.0, 45.0, 110.0, 125.0])
    summary = summarize_intervals(truth, lower, upper, rul_cap=125, n_reordered=2)
    overall = summary["overall"]
    assert overall["n"] == 4 and overall["n_reordered"] == 2
    assert overall["coverage_unclipped"] == 0.5  # 10 and 100 covered; 30 below; 140 above cap
    assert overall["coverage_clipped"] == 0.75  # 140 → 125 is covered
    assert overall["mean_width"] == pytest.approx((10 + 10 + 20 + 15) / 4)
    buckets = {b["true_rul_bucket"]: b for b in summary["buckets"]}
    assert buckets[">125"]["coverage_unclipped"] == 0.0
    assert buckets[">125"]["coverage_clipped"] == 1.0
    assert buckets["21-60"]["coverage_unclipped"] == 0.0
    empty = summarize_intervals(np.array([5.0]), np.array([0.0]), np.array([9.0]), 125, 0)
    assert {b["true_rul_bucket"]: b["n"] for b in empty["buckets"]}[">125"] == 0


# --- inference ----------------------------------------------------------------------------------
def test_inference_with_without_and_mismatched_intervals(
    models: Path, history: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    model = load_model(models / "model.joblib")
    intervals = load_interval_model(interval_path_for(models / "model.joblib"))
    without = predict_rul(history, model)
    with_iv = predict_rul(history, model, intervals)
    stale = IntervalModel(
        intervals.lower, intervals.upper, {**intervals.metadata, "for_model_version": "old"}
    )
    with caplog.at_level(logging.WARNING, logger="predmaint.models"):
        mismatched = predict_rul(history, model, stale)

    assert (
        without.rul_lower is None and without.rul_upper is None and without.interval_level is None
    )
    assert mismatched.rul_lower is None and mismatched.interval_level is None
    assert any("Ignoring prediction intervals" in r.getMessage() for r in caplog.records)
    assert with_iv.interval_level == 0.8
    assert 0 <= with_iv.rul_lower <= with_iv.predicted_rul <= with_iv.rul_upper <= 125
    # point prediction byte-identical in all three cases
    points = {r.predicted_rul.hex() for r in (without, with_iv, mismatched)}
    assert len(points) == 1
    strip = {
        "rul_lower": None,
        "rul_upper": None,
        "interval_level": None,
        "interval_calibrated": None,
    }
    assert dataclasses.replace(with_iv, **strip) == without == mismatched


def test_interval_feature_mismatch_is_ignored(models: Path, history: pd.DataFrame) -> None:
    model = load_model(models / "model.joblib")
    intervals = load_interval_model(interval_path_for(models / "model.joblib"))
    other = IntervalModel(
        intervals.lower, intervals.upper, {**intervals.metadata, "feature_names": ["cycle"]}
    )
    assert predict_rul(history, model, other).rul_lower is None


def test_interval_uses_only_last_window(models: Path, history: pd.DataFrame) -> None:
    model = load_model(models / "model.joblib")
    intervals = load_interval_model(interval_path_for(models / "model.joblib"))
    full, recent = (
        predict_rul(history, model, intervals),
        predict_rul(history.tail(30), model, intervals),
    )
    assert (full.rul_lower, full.rul_upper) == (recent.rul_lower, recent.rul_upper)


def test_load_interval_model_errors(models: Path, tmp_path: Path) -> None:
    from predmaint.models.predict import ModelNotAvailableError

    with pytest.raises(ModelNotAvailableError):
        load_interval_model(tmp_path / "model_intervals.joblib")
    shutil.copytree(models, tmp_path / "m")
    meta_path = tmp_path / "m" / "model_intervals_metadata.json"
    meta = json.loads(meta_path.read_text())
    del meta["for_model_version"]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ModelNotAvailableError, match="for_model_version"):
        load_interval_model(tmp_path / "m" / "model_intervals.joblib")
    (tmp_path / "m" / "model_intervals.joblib").write_bytes(b"not a pickle")
    clear_model_cache()
    with pytest.raises(ModelNotAvailableError, match="could not load"):
        load_interval_model(tmp_path / "m" / "model_intervals.joblib")


# --- conformal calibration (ADR-037, M10) -----------------------------------------------------
def test_nonconformity_hand_values() -> None:
    y = np.array([5.0, 15.0, 10.0])
    e = nonconformity(y, np.array([0.0, 0.0, 8.0]), np.array([10.0, 10.0, 12.0]))
    assert e.tolist() == [-5.0, 5.0, -2.0]  # inside → negative (distance to nearer bound)


def test_conformal_quantile_hand_values() -> None:
    # n = 5, alpha = 0.2: k = ceil(6 * 0.8) = ceil(4.8) = 5 → 5th smallest
    assert conformal_quantile(np.array([5.0, 1.0, 4.0, 2.0, 3.0]), 0.2) == 5.0
    # n = 9: k = ceil(10 * 0.8) = 8 exactly (no float round-up to 9)
    assert conformal_quantile(np.arange(1.0, 10.0), 0.2) == 8.0
    # n = 4, alpha = 0.5: k = ceil(2.5) = 3
    assert conformal_quantile(np.array([40.0, 10.0, 30.0, 20.0]), 0.5) == 30.0
    # n = 3, alpha = 0.2: k = ceil(3.2) = 4 > n → +inf (too few rows)
    assert conformal_quantile(np.array([1.0, 2.0, 3.0]), 0.2) == np.inf
    assert ALPHA == 0.2
    with pytest.raises(ValueError, match="no calibration scores"):
        conformal_quantile(np.array([]))
    with pytest.raises(ValueError, match="alpha"):
        conformal_quantile(np.array([1.0]), 1.0)


def test_engine_split_is_deterministic_disjoint_and_complete() -> None:
    units = np.repeat(np.arange(1, 101), 3)
    cal, ev = engine_split(units, 42)
    assert (cal, ev) == engine_split(units, 42)
    assert len(cal) == len(ev) == 50 and not set(cal) & set(ev)
    assert sorted(cal + ev) == list(range(1, 101))
    assert cal != engine_split(units, 7)[0]


def test_calibration_widens_never_narrows_and_keeps_invariants() -> None:
    rng = np.random.default_rng(1)
    point = np.clip(rng.normal(60, 50, 5000), 0, 125)
    lower = point + rng.normal(-10, 25, 5000)
    upper = point + rng.normal(10, 25, 5000)
    lo0, hi0, _ = order_bounds(point, lower, upper, 125)
    for q in (0.0, 1.7, 12.0):
        lo, hi, _ = order_bounds(point, lower, upper, 125, q)
        assert (lo <= lo0).all() and (hi >= hi0).all()  # Q >= 0 never narrows
        assert (lo >= 0).all() and (hi <= 125).all()
        assert (lo <= point).all() and (point <= hi).all()
    same = order_bounds(point, lower, upper, 125, None)
    assert np.array_equal(same[0], lo0) and np.array_equal(same[1], hi0)  # None = v1.1
    # standard CQR: a negative Q (over-covering raw interval) narrows; documented in ADR-037
    lo_n, hi_n, _ = order_bounds(np.array([50.0]), np.array([30.0]), np.array([70.0]), 125, -5.0)
    assert (lo_n[0], hi_n[0]) == (35.0, 65.0)


def test_cqr_reaches_nominal_coverage_when_rows_are_exchangeable() -> None:
    """i.i.d. rows (each its own 'engine'), raw interval far too narrow → CQR fixes coverage."""
    rng = np.random.default_rng(0)
    n = 6000
    x = rng.uniform(50, 100, n)
    y = x + rng.normal(0, 5, n)
    raw_lo, raw_hi = x - 2.0, x + 2.0  # ~31 % coverage before calibration
    out = calibrate(np.arange(n), y, y, x, raw_lo, raw_hi, None, seed=3)
    before = out["evaluation_half"]["before"]["overall"]["coverage_clipped"]
    after = out["evaluation_half"]["after"]["overall"]["coverage_clipped"]
    assert before < 0.4
    assert abs(after - 0.8) < 0.03
    assert out["q_eval"] > 0 and out["q_final"] > 0
    assert out["acceptance"]["met"] is True
    assert out["split"]["n_calibration_rows"] + out["split"]["n_evaluation_rows"] == n


def test_served_fixture_records_calibration(models: Path) -> None:
    meta = json.loads((models / "model_intervals_metadata.json").read_text())
    cal = meta["calibration"]
    assert meta["conformal_q"] == cal["q_final"] and meta["alpha"] == 0.2
    assert set(cal["evaluation_half"]) == {"before", "after"}
    assert set(cal["test"]) == {"before", "after"}
    assert cal["test"]["before"] == meta["evaluation"]["test"]  # before == v1.1 numbers
    assert cal["split"]["n_calibration_engines"] + cal["split"]["n_evaluation_engines"] == 6
    assert "exchangeable" in cal["caveat"]
    assert load_interval_model(interval_path_for(models / "model.joblib")).calibrated is True


def test_uncalibrated_metadata_behaves_like_v1_1(models: Path, history: pd.DataFrame) -> None:
    model = load_model(models / "model.joblib")
    calibrated = load_interval_model(interval_path_for(models / "model.joblib"))
    old_meta = {k: v for k, v in calibrated.metadata.items() if k not in ("conformal_q", "alpha")}
    old = IntervalModel(calibrated.lower, calibrated.upper, old_meta)
    assert old.conformal_q is None and old.calibrated is False
    v11, v12 = predict_rul(history, model, old), predict_rul(history, model, calibrated)
    assert v11.interval_calibrated is False and v12.interval_calibrated is True
    assert v11.predicted_rul == v12.predicted_rul
    window = history.tail(30)
    from predmaint.features.build import build_features

    features = build_features(window, model.required_sensors, model.window_size)
    raw_lo, raw_hi = calibrated.lower.predict(features), calibrated.upper.predict(features)
    point = np.array([v11.predicted_rul])
    assert (v11.rul_lower, v11.rul_upper) == tuple(
        float(b[0]) for b in order_bounds(point, raw_lo, raw_hi, 125)[:2]
    )
    q = calibrated.conformal_q
    assert (v12.rul_lower, v12.rul_upper) == tuple(
        float(b[0]) for b in order_bounds(point, raw_lo, raw_hi, 125, q)[:2]
    )
