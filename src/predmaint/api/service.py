"""Model holder and thin adapters between HTTP schemas and the inference contract.

All feature building and input validation happen in :func:`predmaint.models.predict.predict_rul`;
this module only converts shapes.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn

from predmaint.api.schemas import (
    AnomalyInfo,
    AnomalyResponse,
    CalibrationInfo,
    FeatureImportanceInfo,
    IntervalsInfo,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from predmaint.data.load import SENSOR_COLUMNS, SETTING_COLUMNS
from predmaint.models.predict import (
    AnomalyModel,
    AnomalyNotAvailableError,
    IntervalModel,
    LoadedModel,
    ModelNotAvailableError,
    anomaly_mismatch,
    anomaly_path_for,
    feature_importance_path_for,
    interval_mismatch,
    interval_path_for,
    load_anomaly_model,
    load_feature_importance,
    load_interval_model,
    load_model,
    predict_rul,
    score_anomaly,
)

logger = logging.getLogger("predmaint.api")


class ModelService:
    """Holds the served model (or the reason it is unavailable) for the app's lifetime."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model: LoadedModel | None = None
        self.load_error: str | None = None
        self.intervals: IntervalModel | None = None
        self.anomaly: AnomalyModel | None = None
        self.anomaly_error: str | None = "no point model loaded"
        self.feature_importance: dict[str, Any] | None = None

    def load(self) -> None:
        """Load the model once; on failure keep running in degraded mode (ADR-022/028)."""
        try:
            self.model = load_model(self.model_path)
        except ModelNotAvailableError as exc:
            self.model, self.load_error = None, str(exc)
            logger.warning("Model not available, serving in degraded mode: %s", exc)
            return
        logger.info("Loaded model %s from %s", self.model.model_version, self.model_path)
        self._load_intervals(self.model)
        self._load_anomaly(self.model)
        self._load_feature_importance(self.model)

    def _load_feature_importance(self, model: LoadedModel) -> None:
        """Load the matching importance report (ADR-040); otherwise warn and report null."""
        path = feature_importance_path_for(self.model_path)
        try:
            report = load_feature_importance(path)
        except ModelNotAvailableError as exc:
            logger.warning("Feature importance not available: %s", exc)
            return
        if report["for_model_version"] != model.model_version:
            logger.warning(
                "Feature importance ignored: computed for %r, served model is %r",
                report["for_model_version"],
                model.model_version,
            )
            return
        self.feature_importance = report

    def _load_anomaly(self, model: LoadedModel) -> None:
        """Load the matching anomaly detector; otherwise warn and leave /anomaly unavailable."""
        path = anomaly_path_for(self.model_path)
        try:
            anomaly = load_anomaly_model(path)
        except AnomalyNotAvailableError as exc:
            self.anomaly_error = str(exc)
            logger.warning("Anomaly detector not available: %s", exc)
            return
        reason = anomaly_mismatch(anomaly, model)
        if reason is not None:
            self.anomaly_error = reason
            logger.warning("Anomaly detector ignored: %s", reason)
            return
        self.anomaly, self.anomaly_error = anomaly, None
        logger.info("Loaded anomaly detector %s from %s", anomaly.model_version, path)

    def score(self, request: PredictRequest, include_series: bool) -> AnomalyResponse:
        """Run ``score_anomaly`` on the request (same validation as ``predict``)."""
        if self.anomaly is None or self.model is None:
            raise AnomalyNotAvailableError(f"No anomaly detector loaded: {self.anomaly_error}")
        records = [record.model_dump() for record in request.records]
        result = score_anomaly(records, self.model, self.anomaly, include_series)
        series = None
        if result.series is not None:
            series = [
                {
                    "cycle": p["cycle"],
                    "normalized_score": round(p["normalized_score"], 4),
                    "raw_normalized_score": round(p["raw_normalized_score"], 4),
                }
                for p in result.series
            ]
        return AnomalyResponse(
            unit_id=request.unit_id,
            last_cycle=result.last_cycle,
            anomaly_score=round(result.anomaly_score, 4),
            normalized_score=round(result.normalized_score, 4),
            threshold=round(result.threshold, 4),
            is_anomalous=result.is_anomalous,
            model_version=result.model_version,
            raw_normalized_score=None
            if result.raw_normalized_score is None
            else round(result.raw_normalized_score, 4),
            smoothed=result.smoothed,
            cycles_in_smoothing=result.cycles_in_smoothing,
            series=series,
        )

    def _load_intervals(self, model: LoadedModel) -> None:
        """Load the matching interval artifact; otherwise warn and serve point predictions only."""
        path = interval_path_for(self.model_path)
        try:
            intervals = load_interval_model(path)
        except ModelNotAvailableError as exc:
            logger.warning("Prediction intervals not available, serving point predictions: %s", exc)
            return
        reason = interval_mismatch(intervals, model)
        if reason is not None:
            logger.warning("Prediction intervals ignored, serving point predictions: %s", reason)
            return
        self.intervals = intervals
        logger.info("Loaded prediction intervals (%.0f %%) from %s", intervals.level * 100, path)

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
        result = predict_rul(records, model, self.intervals)
        return PredictResponse(
            unit_id=request.unit_id,
            predicted_rul=round(result.predicted_rul, 2),
            health_status=result.health_status,
            cycles_received=result.cycles_received,
            cycles_used=result.cycles_used,
            last_cycle=result.last_cycle,
            model_version=result.model_version,
            warnings=result.warnings,
            rul_lower=None if result.rul_lower is None else round(result.rul_lower, 2),
            rul_upper=None if result.rul_upper is None else round(result.rul_upper, 2),
            interval_level=result.interval_level,
            interval_calibrated=result.interval_calibrated,
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
            intervals=self._intervals_info(),
            anomaly=self._anomaly_info(),
            feature_importance=self._feature_importance_info(),
        )

    def _feature_importance_info(self) -> FeatureImportanceInfo | None:
        if self.feature_importance is None:
            return None
        report = self.feature_importance
        test = report["populations"]["test"]
        return FeatureImportanceInfo(
            population="test (official FD001 test set, last cycle of each engine)",
            target=test["target"],
            n_rows=test["n_rows"],
            n_repeats=report["n_repeats"],
            seed=report["seed"],
            baseline_rmse=test["baseline_rmse"],
            top_features=test["per_feature"][:10],
            per_sensor=test["per_sensor"],
            caveat=test["caveat"],
            computed_at=report["computed_at"],
        )

    def _anomaly_info(self) -> AnomalyInfo | None:
        if self.anomaly is None:
            return None
        meta = self.anomaly.metadata
        train = meta["evaluation"]["train_oof"]
        return AnomalyInfo(
            model_version=meta["model_version"],
            for_model_version=meta["for_model_version"],
            method=meta["method"],
            threshold=meta["threshold"],
            threshold_quantile=meta["threshold_quantile"],
            healthy_rows=meta["healthy_rows"],
            trained_at=meta["trained_at"],
            trend_median_train_oof=train["trend"]["median"],
            false_alarm_rate_healthy=train["false_alarm_rate_healthy"],
            detection_fraction_of_life_median=train["detection"]["fraction_of_life"]["median"],
            detection_true_rul_median=train["detection"]["true_rul_at_detection"]["median"],
            never_flagged_fraction=train["detection"]["never_flagged_fraction"],
            acceptance_met=meta["evaluation"]["acceptance"]["met"],
            smoothing=self.anomaly.smoothing,
            smoothing_enabled=self.anomaly.smoothing_enabled,
            smoothed_threshold=self.anomaly.smoothed_threshold,
            smoothed_acceptance_met=None
            if "smoothed" not in meta
            else meta["smoothed"]["evaluation"]["acceptance"]["met"],
            evaluation_raw=meta["evaluation"],
            evaluation_smoothed=None if "smoothed" not in meta else meta["smoothed"]["evaluation"],
        )

    def _intervals_info(self) -> IntervalsInfo | None:
        if self.intervals is None:
            return None
        meta = self.intervals.metadata
        return IntervalsInfo(
            for_model_version=meta["for_model_version"],
            method=meta["method"],
            quantiles=meta["quantiles"],
            interval_level=meta["interval_level"],
            trained_at=meta["trained_at"],
            cv_oof=meta["evaluation"]["cv_oof"]["overall"],
            test=meta["evaluation"]["test"]["overall"],
            calibrated=self.intervals.calibrated,
            conformal_q=self.intervals.conformal_q,
            alpha=meta.get("alpha"),
            calibration=_calibration_info(meta.get("calibration")),
        )


def _calibration_info(cal: dict[str, Any] | None) -> CalibrationInfo | None:
    if not cal:
        return None
    return CalibrationInfo(
        method=cal["method"],
        target=cal["target"],
        q_eval=cal["q_eval"],
        n_calibration_engines=cal["split"]["n_calibration_engines"],
        n_evaluation_engines=cal["split"]["n_evaluation_engines"],
        evaluation_half_before=cal["evaluation_half"]["before"]["overall"],
        evaluation_half_after=cal["evaluation_half"]["after"]["overall"],
        test_before=cal["test"]["before"]["overall"],
        test_after=cal["test"]["after"]["overall"],
        acceptance_criterion=cal["acceptance"]["criterion"],
        acceptance_met=cal["acceptance"]["met"],
        caveat=cal["caveat"],
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
    payload: dict[str, Any],
    model: LoadedModel,
    true_rul: int | None = None,
    intervals: IntervalModel | None = None,
    anomaly: AnomalyModel | None = None,
) -> dict[str, Any]:
    """Offline reference for a demo payload: what ``POST /predict`` (and ``/anomaly``) return."""
    result = predict_rul(payload["records"], model, intervals)
    scored = None if anomaly is None else score_anomaly(payload["records"], model, anomaly)
    return {
        "unit_id": payload["unit_id"],
        "cycles_received": result.cycles_received,
        "last_cycle": result.last_cycle,
        "predicted_rul": round(result.predicted_rul, 2),
        "health_status": result.health_status,
        "model_version": result.model_version,
        "rul_lower": None if result.rul_lower is None else round(result.rul_lower, 2),
        "rul_upper": None if result.rul_upper is None else round(result.rul_upper, 2),
        "interval_level": result.interval_level,
        "interval_calibrated": result.interval_calibrated,
        "true_rul_at_last_cycle": true_rul,
        "anomaly": None
        if scored is None
        else {
            "anomaly_score": round(scored.anomaly_score, 4),
            "normalized_score": round(scored.normalized_score, 4),
            "threshold": round(scored.threshold, 4),
            "is_anomalous": scored.is_anomalous,
            "model_version": scored.model_version,
            "raw_normalized_score": round(scored.raw_normalized_score, 4),  # type: ignore[arg-type]
            "smoothed": scored.smoothed,
            "cycles_in_smoothing": scored.cycles_in_smoothing,
        },
    }
