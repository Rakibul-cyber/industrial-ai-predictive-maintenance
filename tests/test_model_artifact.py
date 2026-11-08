"""The committed served artifact (ADR-030): present, loadable, version-pinned, reproducible.

Runs without the dataset, so CI checks the exact files that a fresh clone ships.
"""

import json
from pathlib import Path

import pytest
import sklearn

from predmaint.models.predict import clear_model_cache, load_model, predict_rul
from predmaint.models.registry import pipeline_feature_names

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "model.joblib"
METADATA = ROOT / "models" / "model_metadata.json"
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
    result = predict_rul(payload["records"], model)
    assert round(result.predicted_rul, 2) == expected["predicted_rul"]
    assert result.health_status == expected["health_status"]
    assert result.model_version == expected["model_version"]
