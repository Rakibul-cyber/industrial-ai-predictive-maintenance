"""Model holder and thin adapters between HTTP schemas and the inference contract.

All feature building and input validation happen in :func:`predmaint.models.predict.predict_rul`;
this module only converts shapes.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn

from predmaint.api.schemas import ModelInfoResponse, PredictRequest, PredictResponse
from predmaint.data.load import SENSOR_COLUMNS, SETTING_COLUMNS
from predmaint.models.predict import (
    LoadedModel,
    ModelNotAvailableError,
    load_model,
    predict_rul,
)

logger = logging.getLogger("predmaint.api")


class ModelService:
    """Holds the served model (or the reason it is unavailable) for the app's lifetime."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model: LoadedModel | None = None
        self.load_error: str | None = None

    def load(self) -> None:
        """Load the model once; on failure keep running in degraded mode (ADR-022/028)."""
        try:
            self.model = load_model(self.model_path)
        except ModelNotAvailableError as exc:
            self.model, self.load_error = None, str(exc)
            logger.warning("Model not available, serving in degraded mode: %s", exc)
            return
        logger.info("Loaded model %s from %s", self.model.model_version, self.model_path)

    def require_model(self) -> LoadedModel:
        """Return the model or raise :class:`ModelNotAvailableError` (→ 503)."""
        if self.model is None:
            raise ModelNotAvailableError(
                f"No model loaded from {self.model_path}: {self.load_error}"
            )
        return self.model

    def predict(self, request: PredictRequest) -> PredictResponse:
        """Run ``predict_rul`` on the request records and shape the response."""
        model = self.require_model()
        records = [record.model_dump() for record in request.records]
        result = predict_rul(records, model)
        return PredictResponse(
            unit_id=request.unit_id,
            predicted_rul=round(result.predicted_rul, 2),
            health_status=result.health_status,
            cycles_received=result.cycles_received,
            cycles_used=result.cycles_used,
            last_cycle=result.last_cycle,
            model_version=result.model_version,
            warnings=result.warnings,
        )

    def model_info(self) -> ModelInfoResponse:
        """Summarise ``model_metadata.json`` for ``GET /model/info``."""
        meta = self.require_model().metadata
        metrics = meta["metrics"]
        return ModelInfoResponse(
            model_name=meta["model_name"],
            model_version=meta["model_version"],
            trained_at=meta["trained_at"],
            git_sha=meta.get("git_sha") or "unknown",
            selected_by=meta.get("selected_by"),
            data_subset=meta["data"]["subset"],
            train_sha256=meta["data"]["train_sha256"],
            window_size=meta["config"]["window_size"],
            rul_cap=meta["config"]["rul_cap"],
            n_features=len(meta["feature_names"]),
            required_sensors=meta["required_sensors"],
            feature_names=meta["feature_names"],
            health_bands=meta["health_bands"],
            metrics={
                "cv": {k: v for k, v in metrics["cv"].items() if k != "folds"},
                "test": metrics["test"],
                "test_clipped_truth": metrics["test_clipped_truth"],
            },
            sklearn_version_match=meta["library_versions"]["scikit-learn"] == sklearn.__version__,
        )


def request_payload(frame: pd.DataFrame, unit_id: str | None = None) -> dict[str, Any]:
    """Build a ``POST /predict`` body from raw FD-format rows of one engine (oldest first)."""
    columns = ["cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS]
    ordered = frame.sort_values("cycle")[columns]
    records = []
    for row in ordered.to_dict("records"):
        record: dict[str, Any] = {"cycle": int(row["cycle"])}
        record.update({c: float(row[c]) for c in columns[1:]})
        records.append(record)
    return {"unit_id": unit_id, "records": records}


def expected_prediction(
    payload: dict[str, Any], model: LoadedModel, true_rul: int | None = None
) -> dict[str, Any]:
    """Offline reference for a demo payload: what ``POST /predict`` must return for it."""
    result = predict_rul(payload["records"], model)
    return {
        "unit_id": payload["unit_id"],
        "cycles_received": result.cycles_received,
        "last_cycle": result.last_cycle,
        "predicted_rul": round(result.predicted_rul, 2),
        "health_status": result.health_status,
        "model_version": result.model_version,
        "true_rul_at_last_cycle": true_rul,
    }
