"""Pydantic request/response models (docs/06 §2-§6)."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from predmaint.data.load import SENSOR_COLUMNS, SETTING_COLUMNS

#: Hard upper bound on records per request, to bound memory (docs/06 §3.2).
MAX_RECORDS_PER_REQUEST = 1000

_RECORD_CONFIG = ConfigDict(
    extra="forbid",
    allow_inf_nan=False,
    json_schema_extra={
        "examples": [
            {
                "cycle": 31,
                "setting_1": -0.0006,
                "setting_2": 0.0004,
                "setting_3": 100.0,
                **{s: 0.0 for s in SENSOR_COLUMNS},
            }
        ]
    },
)

#: One engine cycle in the raw FD001 layout. Sensors/settings are optional at schema level; the
#: model's ``required_sensors`` must be non-null (checked by the inference contract).
SensorRecord = create_model(
    "SensorRecord",
    __config__=_RECORD_CONFIG,
    __doc__="One operating cycle of one engine (raw FD001 columns; extra fields rejected).",
    cycle=(int, Field(ge=1, description="Cycle number; consecutive (+1) across records.")),
    **{
        name: (float | None, Field(default=None, description=f"{name} (finite)"))
        for name in SETTING_COLUMNS + SENSOR_COLUMNS
    },
)


class PredictRequest(BaseModel):
    """One engine's recent cycles, oldest first (at least ``window_size`` = 30)."""

    model_config = ConfigDict(extra="forbid")

    unit_id: str | None = Field(
        default=None, min_length=1, max_length=64, description="Optional id, echoed back."
    )
    records: list[SensorRecord] = Field(  # type: ignore[valid-type]
        min_length=1,
        max_length=MAX_RECORDS_PER_REQUEST,
        description=f"Chronological cycle records (1-{MAX_RECORDS_PER_REQUEST}).",
    )


class PredictResponse(BaseModel):
    """RUL prediction for the last cycle in the request."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "unit_id": "test-unit-1",
                    "predicted_rul": 125.0,
                    "health_status": "healthy",
                    "cycles_received": 31,
                    "cycles_used": 30,
                    "last_cycle": 31,
                    "model_version": "primary-20261108-f4c86df",
                    "warnings": [],
                    "rul_lower": 98.4,
                    "rul_upper": 125.0,
                    "interval_level": 0.8,
                }
            ]
        }
    )

    unit_id: str | None
    predicted_rul: float = Field(description="Remaining useful life in cycles, in [0, rul_cap].")
    health_status: Literal["critical", "warning", "healthy"]
    cycles_received: int
    cycles_used: int
    last_cycle: int
    model_version: str
    warnings: list[str]
    rul_lower: float | None = Field(
        default=None, description="Lower bound of the prediction interval, or null (ADR-034)."
    )
    rul_upper: float | None = Field(
        default=None, description="Upper bound of the prediction interval, or null (ADR-034)."
    )
    interval_level: float | None = Field(
        default=None, description="Nominal interval level (0.8), or null without intervals."
    )


class HealthResponse(BaseModel):
    """Liveness plus model readiness (ADR-028)."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None
    api_version: str
    intervals_loaded: bool


class CvMetrics(BaseModel):
    mae_mean: float
    mae_std: float
    rmse_mean: float
    rmse_std: float


class TestMetrics(BaseModel):
    mae: float
    rmse: float
    nasa_score: float


class ClippedTestMetrics(BaseModel):
    mae: float
    rmse: float


class ModelMetrics(BaseModel):
    cv: CvMetrics
    test: TestMetrics
    test_clipped_truth: ClippedTestMetrics


class IntervalCoverage(BaseModel):
    n: int
    coverage_unclipped: float
    coverage_clipped: float
    mean_width: float
    median_width: float
    n_reordered: int


class IntervalsInfo(BaseModel):
    """Prediction-interval artifact summary (ADR-034)."""

    model_config = ConfigDict(protected_namespaces=())

    for_model_version: str
    method: str
    quantiles: list[float]
    interval_level: float
    trained_at: str
    cv_oof: IntervalCoverage
    test: IntervalCoverage


class ModelInfoResponse(BaseModel):
    """Served model identity, provenance and metrics, from ``model_metadata.json``."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    trained_at: str
    git_sha: str
    selected_by: str | None
    data_subset: str
    train_sha256: str
    window_size: int
    rul_cap: int | None
    n_features: int
    required_sensors: list[str]
    feature_names: list[str]
    health_bands: dict[str, int]
    metrics: ModelMetrics
    sklearn_version_match: bool
    intervals: IntervalsInfo | None = None


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[dict[str, Any]]


class ErrorResponse(BaseModel):
    """The single error envelope used for every non-2xx response (docs/06 §6)."""

    error: ErrorBody
