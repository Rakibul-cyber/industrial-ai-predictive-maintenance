"""Inference contract (docs/05 §10): the seam the API calls.

``predict_from_features`` is the single post-processing step shared by offline evaluation and
online inference, so both produce identical numbers. Errors are plain Python exceptions; the
HTTP mapping belongs to the API layer (docs/06 §6).
"""

import functools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from predmaint.config import HEALTH_BANDS, get_settings
from predmaint.features.build import build_features
from predmaint.models.artifacts import load_artifacts

SERVED_ARTIFACT = "model"
HealthStatus = Literal["critical", "warning", "healthy"]


# --- Errors ----------------------------------------------------------------------------------
class ModelNotAvailableError(RuntimeError):
    """The served model artifact is missing or cannot be loaded (API: 503 MODEL_UNAVAILABLE)."""


class InputContractError(ValueError):
    """Base class for invalid prediction input (API: 422)."""


class InsufficientHistoryError(InputContractError):
    """Fewer than ``window_size`` cycles were provided (API: INSUFFICIENT_CYCLES)."""


class MissingColumnsError(InputContractError):
    """``cycle`` or a required sensor is absent or null (API: MISSING_SENSORS)."""

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = list(columns)
        super().__init__(f"required columns missing or null: {', '.join(self.columns)}")


class NonConsecutiveCyclesError(InputContractError):
    """Cycles are not strictly increasing by exactly 1 in the given order
    (API: NON_CONSECUTIVE_CYCLES)."""


class InvalidValuesError(InputContractError):
    """Values are non-finite or cycle numbers are not positive integers (API: VALIDATION_ERROR)."""


# --- Model -----------------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadedModel:
    """A fitted pipeline plus the metadata that defines its input contract."""

    pipeline: Pipeline
    metadata: dict[str, Any]

    @property
    def required_sensors(self) -> list[str]:
        return list(self.metadata["required_sensors"])

    @property
    def window_size(self) -> int:
        return int(self.metadata["config"]["window_size"])

    @property
    def rul_cap(self) -> int | None:
        return self.metadata["config"]["rul_cap"]

    @property
    def model_version(self) -> str:
        return str(self.metadata["model_version"])


@dataclass(frozen=True)
class PredictionResult:
    """Output of :func:`predict_rul`."""

    predicted_rul: float
    health_status: HealthStatus
    cycles_received: int
    cycles_used: int
    last_cycle: int
    model_version: str
    warnings: list[str] = field(default_factory=list)


def predict_from_features(
    pipeline: Pipeline, features: pd.DataFrame, rul_cap: int | None
) -> np.ndarray:
    """Predict RUL for feature rows, clamped to ``[0, rul_cap]`` (ADR-027)."""
    upper = np.inf if rul_cap is None else float(rul_cap)
    return np.clip(pipeline.predict(features), 0.0, upper)


def health_status(rul: float, bands: Mapping[str, float] = HEALTH_BANDS) -> HealthStatus:
    """ADR-025: ``critical`` below 20, ``warning`` below 60, otherwise ``healthy``."""
    if rul < bands["critical_below"]:
        return "critical"
    if rul < bands["warning_below"]:
        return "warning"
    return "healthy"


@functools.cache
def _load_cached(model_dir: Path) -> LoadedModel:
    try:
        pipeline, metadata = load_artifacts(model_dir, SERVED_ARTIFACT)
    except FileNotFoundError as exc:
        raise ModelNotAvailableError(str(exc)) from exc
    except Exception as exc:  # corrupt pickle/JSON, schema mismatch, incompatible library
        raise ModelNotAvailableError(f"could not load model from {model_dir}: {exc}") from exc
    for key in ("required_sensors", "feature_names", "config", "model_version"):
        if key not in metadata:
            raise ModelNotAvailableError(f"{model_dir}: metadata lacks {key!r}")
    return LoadedModel(pipeline=pipeline, metadata=metadata)


def load_model(model_dir: Path | None = None) -> LoadedModel:
    """Load the served model (``model.joblib`` + ``model_metadata.json``) once and cache it.

    ``model_dir`` defaults to ``$MODEL_DIR``. Failed loads are not cached, so a model added
    later is picked up on the next call; :func:`clear_model_cache` forces a reload.

    Raises:
        ModelNotAvailableError: if the files are missing or cannot be loaded.
    """
    directory = (model_dir or get_settings().model_dir).resolve()
    return _load_cached(directory)


def clear_model_cache() -> None:
    """Drop cached models (e.g. after promoting a new artifact)."""
    _load_cached.cache_clear()


# --- Prediction ------------------------------------------------------------------------------
def _validated_window(frame: pd.DataFrame, model: LoadedModel) -> pd.DataFrame:
    if len(frame) < model.window_size:
        raise InsufficientHistoryError(
            f"{len(frame)} cycles received; at least {model.window_size} are required"
        )
    required = ["cycle", *model.required_sensors]
    missing = [c for c in required if c not in frame.columns or frame[c].isna().any()]
    if missing:
        raise MissingColumnsError(missing)
    try:
        values = frame[required].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidValuesError(f"non-numeric values in required columns: {exc}") from exc
    if not np.isfinite(values).all():
        raise InvalidValuesError("required columns contain non-finite values")
    cycles = values[:, 0]
    if (cycles < 1).any() or (cycles != np.floor(cycles)).any():
        raise InvalidValuesError("cycle numbers must be positive integers")
    if (np.diff(cycles) != 1).any():
        raise NonConsecutiveCyclesError(
            "cycles must be strictly increasing by exactly 1 in the order given"
        )
    window = frame[required].iloc[-model.window_size :].copy()
    window["cycle"] = window["cycle"].astype("int64")
    window["unit"] = 0
    return window


def _range_warnings(window: pd.DataFrame, model: LoadedModel) -> list[str]:
    warnings = []
    for sensor, bounds in model.metadata.get("training_ranges", {}).items():
        lo, hi = window[sensor].min(), window[sensor].max()
        if lo < bounds["min"] or hi > bounds["max"]:
            warnings.append(
                f"OUT_OF_TRAINING_RANGE: {sensor} values [{lo:g}, {hi:g}] outside "
                f"training range [{bounds['min']:g}, {bounds['max']:g}]"
            )
    return warnings


def predict_rul(
    cycles: pd.DataFrame | Sequence[Mapping[str, Any]], model: LoadedModel
) -> PredictionResult:
    """Predict RUL for one engine from its recent cycles, in chronological order.

    Only the last ``window_size`` cycles are used, to build a single feature row with the
    shared feature function. Extra columns are ignored.

    Raises:
        InsufficientHistoryError: fewer than ``window_size`` cycles.
        MissingColumnsError: ``cycle`` or a required sensor missing or null.
        InvalidValuesError: non-finite values or non-integer / non-positive cycles.
        NonConsecutiveCyclesError: cycles not strictly increasing by 1.
    """
    frame = cycles if isinstance(cycles, pd.DataFrame) else pd.DataFrame(list(cycles))
    window = _validated_window(frame.reset_index(drop=True), model)
    features = build_features(window, model.required_sensors, model.window_size)
    if len(features) != 1:  # guaranteed by the checks above; defends the invariant
        raise InsufficientHistoryError("could not build a full feature window")
    predicted = float(predict_from_features(model.pipeline, features, model.rul_cap)[0])
    if not math.isfinite(predicted):
        raise InvalidValuesError("model produced a non-finite prediction")
    return PredictionResult(
        predicted_rul=predicted,
        health_status=health_status(predicted),
        cycles_received=len(frame),
        cycles_used=model.window_size,
        last_cycle=int(window["cycle"].iloc[-1]),
        model_version=model.model_version,
        warnings=_range_warnings(window, model),
    )
