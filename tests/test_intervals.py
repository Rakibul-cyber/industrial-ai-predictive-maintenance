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
from predmaint.models.intervals import QUANTILES, run_interval_training, summarize_intervals
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
    strip = {"rul_lower": None, "rul_upper": None, "interval_level": None}
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
