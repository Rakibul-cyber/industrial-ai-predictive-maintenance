"""The committed served artifact (ADR-030): present, loadable, version-pinned, reproducible.

Runs without the dataset, so CI checks the exact files that a fresh clone ships.
"""

import json
from pathlib import Path

import pytest
import sklearn

from predmaint.models.predict import (
    anomaly_mismatch,
    clear_model_cache,
    interval_mismatch,
    load_anomaly_model,
    load_interval_model,
    load_model,
    predict_rul,
    score_anomaly,
)
from predmaint.models.registry import pipeline_feature_names

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "model.joblib"
METADATA = ROOT / "models" / "model_metadata.json"
INTERVALS = ROOT / "models" / "model_intervals.joblib"
INTERVALS_METADATA = ROOT / "models" / "model_intervals_metadata.json"
ANOMALY = ROOT / "models" / "model_anomaly.joblib"
ANOMALY_METADATA = ROOT / "models" / "model_anomaly_metadata.json"
EXAMPLES = ROOT / "examples"


def test_committed_artifact_files_exist() -> None:
    assert MODEL.is_file(), "models/model.joblib must be committed (ADR-030)"
    assert METADATA.is_file(), "models/model_metadata.json must be committed (ADR-030)"


def test_installed_sklearn_matches_artifact() -> None:
    """joblib pickles are tied to the scikit-learn version: retrain + commit if this fails."""
    trained_with = json.loads(METADATA.read_text())["library_versions"]["scikit-learn"]
    assert sklearn.__version__ == trained_with, (
        f"installed scikit-learn {sklearn.__version__} != {trained_with} used to train "
        "models/model.joblib; retrain (docs/10 §6) and commit both model files together"
    )


def test_committed_artifact_loads_and_matches_metadata() -> None:
    clear_model_cache()
    model = load_model(MODEL)
    meta = json.loads(METADATA.read_text())
    assert model.metadata == meta
    assert pipeline_feature_names(model.pipeline) == meta["feature_names"]
    assert meta["selected_by"] == "cv_rmse"
    assert model.model_version == meta["model_version"]


@pytest.mark.parametrize("example", sorted(p.name for p in EXAMPLES.glob("predict_request*.json")))
def test_committed_artifact_reproduces_sidecar(example: str) -> None:
    """The committed model gives the recorded predictions for the committed example payloads."""
    clear_model_cache()
    model = load_model(MODEL)
    expected = json.loads((EXAMPLES / "expected_predictions.json").read_text())[example]
    payload = json.loads((EXAMPLES / example).read_text())
    result = predict_rul(payload["records"], model, load_interval_model(INTERVALS))
    assert round(result.predicted_rul, 2) == expected["predicted_rul"]
    assert round(result.rul_lower, 2) == expected["rul_lower"]
    assert round(result.rul_upper, 2) == expected["rul_upper"]
    assert result.interval_level == expected["interval_level"]
    assert result.health_status == expected["health_status"]
    assert result.model_version == expected["model_version"]


def test_committed_interval_artifact_matches_served_model() -> None:
    """ADR-034: the committed interval artifact belongs to the committed point model."""
    assert INTERVALS.is_file() and INTERVALS_METADATA.is_file()
    clear_model_cache()
    model, intervals = load_model(MODEL), load_interval_model(INTERVALS)
    assert interval_mismatch(intervals, model) is None
    assert intervals.metadata["library_versions"]["scikit-learn"] == sklearn.__version__


def test_committed_anomaly_artifact_matches_served_model() -> None:
    """ADR-035: the committed anomaly detector belongs to the committed point model."""
    assert ANOMALY.is_file() and ANOMALY_METADATA.is_file()
    clear_model_cache()
    model, detector = load_model(MODEL), load_anomaly_model(ANOMALY)
    assert anomaly_mismatch(detector, model) is None
    assert detector.metadata == json.loads(ANOMALY_METADATA.read_text())
    assert detector.threshold == detector.metadata["threshold"]
    assert detector.metadata["library_versions"]["scikit-learn"] == sklearn.__version__


@pytest.mark.parametrize("example", sorted(p.name for p in EXAMPLES.glob("predict_request*.json")))
def test_committed_anomaly_reproduces_sidecar(example: str) -> None:
    clear_model_cache()
    expected = json.loads((EXAMPLES / "expected_predictions.json").read_text())[example]["anomaly"]
    payload = json.loads((EXAMPLES / example).read_text())
    result = score_anomaly(payload["records"], load_model(MODEL), load_anomaly_model(ANOMALY))
    assert round(result.anomaly_score, 4) == expected["anomaly_score"]
    assert round(result.normalized_score, 4) == expected["normalized_score"]
    assert round(result.threshold, 4) == expected["threshold"]
    assert result.is_anomalous is expected["is_anomalous"]
    assert result.model_version == expected["model_version"]
