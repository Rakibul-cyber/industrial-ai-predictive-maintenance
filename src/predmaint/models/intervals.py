"""Prediction-interval training and evaluation (ADR-034, Milestone 8).

Two quantile HistGradientBoosting models (q = 0.1 and 0.9 → nominal 80 % interval) are trained
on exactly the served point model's features, train-row filter, seed and hyperparameters, all
read from ``models/model_metadata.json``. The point model is never retrained or modified.
Coverage and width are measured on 5-fold GroupKFold out-of-fold predictions and on the
official test set at each engine's last cycle; nothing is tuned toward the nominal level.
"""

import json
import platform
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from predmaint.data.download import sha256_file
from predmaint.data.labels import RUL_LABEL, add_rul
from predmaint.data.load import RUL_COLUMN, load_fd001
from predmaint.data.validate import FD001_EXPECTED, ExpectedFigures, validate_fd001
from predmaint.features.build import build_features, last_rows
from predmaint.models.artifacts import SCHEMA_VERSION, save_artifacts
from predmaint.models.predict import (
    INTERVAL_ARTIFACT,
    load_model,
    order_bounds,
    predict_from_features,
)
from predmaint.models.registry import make_primary, make_quantile_model

QUANTILES: tuple[float, float] = (0.1, 0.9)
INTERVAL_LEVEL = round(QUANTILES[1] - QUANTILES[0], 10)  # 0.8
#: Same buckets as reports/residuals_by_rul_bucket.json (unclipped true RUL).
BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0-20", -np.inf, 20.5),
    ("21-60", 20.5, 60.5),
    ("61-125", 60.5, 125.5),
    (">125", 125.5, np.inf),
)
RAW_RUL = "rul_raw"

Echo = Callable[[str], None]


def _coverage_block(
    truth: np.ndarray, lower: np.ndarray, upper: np.ndarray, rul_cap: int | None
) -> dict[str, Any]:
    clipped = truth if rul_cap is None else np.minimum(truth, rul_cap)
    width = upper - lower
    return {
        "n": int(len(truth)),
        "coverage_unclipped": float(np.mean((truth >= lower) & (truth <= upper))),
        "coverage_clipped": float(np.mean((clipped >= lower) & (clipped <= upper))),
        "mean_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
    }


def summarize_intervals(
    truth: np.ndarray, lower: np.ndarray, upper: np.ndarray, rul_cap: int | None, n_reordered: int
) -> dict[str, Any]:
    """Coverage/width overall and per unclipped true-RUL bucket (empty buckets → ``n = 0``)."""
    truth = np.asarray(truth, dtype=float)
    overall = _coverage_block(truth, lower, upper, rul_cap)
    overall["n_reordered"] = int(n_reordered)
    buckets = []
    for label, lo, hi in BUCKETS:
        mask = (truth > lo) & (truth <= hi)
        if mask.any():
            block = _coverage_block(truth[mask], lower[mask], upper[mask], rul_cap)
        else:
            block = dict.fromkeys(
                ("coverage_unclipped", "coverage_clipped", "mean_width", "median_width")
            )
            block["n"] = 0
        buckets.append({"true_rul_bucket": label, **block})
    return {"overall": overall, "buckets": buckets}


def _format_table(title: str, summary: dict[str, Any]) -> list[str]:
    o = summary["overall"]
    lines = [
        f"{title}: n={o['n']} coverage(unclipped)={o['coverage_unclipped']:.3f} "
        f"coverage(clipped)={o['coverage_clipped']:.3f} mean width={o['mean_width']:.2f} "
        f"median width={o['median_width']:.2f} reordered={o['n_reordered']}",
        "  bucket     n    cov_unclipped  cov_clipped  mean_width  median_width",
    ]
    for b in summary["buckets"]:
        if b["n"] == 0:
            lines.append(f"  {b['true_rul_bucket']:<8} {0:>5}    —")
            continue
        lines.append(
            f"  {b['true_rul_bucket']:<8} {b['n']:>5}    {b['coverage_unclipped']:>11.3f}  "
            f"{b['coverage_clipped']:>11.3f}  {b['mean_width']:>10.2f}  {b['median_width']:>12.2f}"
        )
    return lines


def run_interval_training(
    cmapss_dir: Path,
    model_dir: Path,
    reports_dir: Path,
    *,
    verify_checksums: bool = True,
    expected: ExpectedFigures | None = FD001_EXPECTED,
    echo: Echo = print,
) -> dict[str, Any]:
    """Train the q0.1/q0.9 models for the served model in ``model_dir`` and evaluate them.

    Writes ``<model_dir>/model_intervals.joblib`` (dict with ``lower``/``upper`` pipelines),
    ``<model_dir>/model_intervals_metadata.json`` and ``<reports_dir>/metrics_intervals.json``.

    Raises:
        FileNotFoundError: if the served model (``model.joblib`` + metadata) is missing.
        ValueError: if the served model is not the HGB primary, or the data differ from the
            data it was trained on.
        ChecksumMismatchError / DataValidationError: as for point-model training.
    """
    from predmaint.models.train import _git, _verify_checksums, group_folds

    started = time.perf_counter()
    if not (model_dir / "model_metadata.json").is_file():
        raise FileNotFoundError(
            f"{model_dir / 'model_metadata.json'} not found: train and select the point model "
            "first (scripts/train.py --model baseline/primary, then --select)"
        )
    point = load_model(model_dir / "model.joblib")
    meta = point.metadata
    if meta.get("estimator") != "hgb":
        raise ValueError(
            "prediction intervals reuse the served HistGradientBoosting hyperparameters "
            f"(ADR-034); the served model is {meta.get('model_name')!r} ({meta.get('estimator')})"
        )
    params, config = dict(meta["hyperparameters"]), meta["config"]
    sensors, columns = list(meta["required_sensors"]), list(meta["feature_names"])
    window, rul_cap, seed, n_splits = (
        config["window_size"],
        config["rul_cap"],
        config["random_seed"],
        config["cv_folds"],
    )

    if verify_checksums:
        _verify_checksums(cmapss_dir)
    data = load_fd001(cmapss_dir)
    validate_fd001(data, expected)
    shas = {
        "train_sha256": sha256_file(cmapss_dir / "train_FD001.txt"),
        "test_sha256": sha256_file(cmapss_dir / "test_FD001.txt"),
        "rul_sha256": sha256_file(cmapss_dir / "RUL_FD001.txt"),
    }
    if any(meta["data"].get(k) != v for k, v in shas.items()):
        raise ValueError("data differ from the data the served model was trained on")

    echo(
        f"Intervals for served model {point.model_version}: quantiles {QUANTILES} "
        f"(nominal {INTERVAL_LEVEL:.0%}), hyperparameters {params}"
    )
    labelled = add_rul(data.train, rul_cap)
    labelled[RAW_RUL] = add_rul(data.train, None)[RUL_LABEL]
    train = build_features(labelled, sensors, window, keep=[RUL_LABEL, RAW_RUL])
    y = train[RUL_LABEL].to_numpy(dtype=float)
    raw = train[RAW_RUL].to_numpy(dtype=float)

    # Out-of-fold evaluation (point clones are fitted per fold for ordering only; the served
    # point artifact is not touched).
    cv_started = time.perf_counter()
    oof_point, oof_lo, oof_hi = (np.empty(len(train)) for _ in range(3))
    for train_idx, val_idx in group_folds(train["unit"].to_numpy(), n_splits):
        fit_rows, val_rows = train.iloc[train_idx], train.iloc[val_idx]
        p = make_primary(columns, seed, params).fit(fit_rows, y[train_idx])
        lo = make_quantile_model(columns, seed, params, QUANTILES[0]).fit(fit_rows, y[train_idx])
        hi = make_quantile_model(columns, seed, params, QUANTILES[1]).fit(fit_rows, y[train_idx])
        oof_point[val_idx] = predict_from_features(p, val_rows, rul_cap)
        oof_lo[val_idx] = lo.predict(val_rows)
        oof_hi[val_idx] = hi.predict(val_rows)
    oof_lo, oof_hi, oof_reordered = order_bounds(oof_point, oof_lo, oof_hi, rul_cap)
    cv_summary = summarize_intervals(raw, oof_lo, oof_hi, rul_cap, oof_reordered)
    cv_seconds = time.perf_counter() - cv_started

    fit_started = time.perf_counter()
    lower = make_quantile_model(columns, seed, params, QUANTILES[0]).fit(train, y)
    upper = make_quantile_model(columns, seed, params, QUANTILES[1]).fit(train, y)
    fit_seconds = time.perf_counter() - fit_started

    last = last_rows(build_features(data.test, sensors, window))
    truth = data.rul[RUL_COLUMN].to_numpy(dtype=float)
    test_point = predict_from_features(point.pipeline, last, rul_cap)
    test_lo, test_hi, test_reordered = order_bounds(
        test_point, lower.predict(last), upper.predict(last), rul_cap
    )
    test_summary = summarize_intervals(truth, test_lo, test_hi, rul_cap, test_reordered)

    for line in _format_table(f"{n_splits}-fold GroupKFold out-of-fold", cv_summary):
        echo(line)
    for line in _format_table("Official test set (last cycle)", test_summary):
        echo(line)
    echo(
        f"Timings: CV {cv_seconds:.1f} s (3 models x {n_splits} folds), "
        f"final fit {fit_seconds:.1f} s (2 quantile models)"
    )

    trained_at = datetime.now(UTC).replace(microsecond=0)
    git_status = _git("status", "--porcelain")
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact": INTERVAL_ARTIFACT,
        "method": "HistGradientBoostingRegressor(loss='quantile') at q=0.1 and q=0.9 (ADR-034)",
        "quantiles": list(QUANTILES),
        "interval_level": INTERVAL_LEVEL,
        "for_model_version": point.model_version,
        "for_candidate_model_version": meta.get("candidate_model_version"),
        "trained_at": trained_at.isoformat().replace("+00:00", "Z"),
        "git_sha": _git("rev-parse", "HEAD") or "unknown",
        "git_dirty": None if git_status is None else bool(git_status),
        "python_version": platform.python_version(),
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "data": {"subset": meta["data"]["subset"], **shas},
        "config": dict(config),
        "hyperparameters": params,
        "required_sensors": sensors,
        "feature_names": columns,
        "post_processing": (
            "clamp to [0, rul_cap]; lower=min(lower, point); upper=max(upper, point)"
        ),
        "evaluation": {"cv_oof": cv_summary, "test": test_summary},
        "timings_seconds": {"cv": round(cv_seconds, 1), "final_fit": round(fit_seconds, 1)},
    }
    pipelines = {"lower": lower, "upper": upper}
    model_path, metadata_path = save_artifacts(pipelines, metadata, model_dir, INTERVAL_ARTIFACT)
    report = {k: v for k, v in metadata.items() if k not in ("required_sensors", "feature_names")}
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "metrics_intervals.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    echo(f"Saved {model_path}, {metadata_path}, {report_path}")
    echo(f"Elapsed: {time.perf_counter() - started:.1f} s")
    return metadata
