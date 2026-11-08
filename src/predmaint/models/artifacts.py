"""Save and load model artifacts: ``<name>.joblib`` + ``<name>_metadata.json`` (ADR-011, ADR-023).

joblib files are pickles: load them only from the trusted local model directory (NFR-010).
"""

import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

SCHEMA_VERSION = 1


def artifact_paths(model_dir: Path, name: str) -> tuple[Path, Path]:
    """Return ``(model_path, metadata_path)`` for artifact ``name`` in ``model_dir``."""
    return model_dir / f"{name}.joblib", model_dir / f"{name}_metadata.json"


def save_artifacts(
    pipeline: Pipeline, metadata: dict[str, Any], model_dir: Path, name: str
) -> tuple[Path, Path]:
    """Write the fitted pipeline and its metadata; returns the two paths written."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path, metadata_path = artifact_paths(model_dir, name)
    joblib.dump(pipeline, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return model_path, metadata_path


def load_artifacts(model_dir: Path, name: str) -> tuple[Pipeline, dict[str, Any]]:
    """Load a pipeline and its metadata.

    Raises:
        FileNotFoundError: if either file is missing.
        ValueError: if the metadata schema version is not supported.
    """
    model_path, metadata_path = artifact_paths(model_dir, name)
    for path in (model_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"artifact not found: {path} (run `python scripts/train.py`)")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{metadata_path}: schema_version {metadata.get('schema_version')!r} "
            f"!= supported {SCHEMA_VERSION}"
        )
    return joblib.load(model_path), metadata
