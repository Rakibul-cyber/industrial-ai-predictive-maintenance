"""Training and evaluation (docs/05 §5-§7).

Flow: checksum gate → load → validate → labels → sensor selection (train only) → features →
GroupKFold CV by unit → final fit on all eligible training rows → official test evaluation at
each test unit's last cycle → save artifact, metadata and a small committed metrics report.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from predmaint.config import (
    CV_FOLDS,
    DATASET_SUBSET,
    HEALTH_BANDS,
    RANDOM_SEED,
    RUL_CAP,
    WINDOW_SIZE,
    get_settings,
)
from predmaint.data.download import CMAPSS_SPEC, ChecksumMismatchError, sha256_file
from predmaint.data.labels import RUL_LABEL, add_rul
from predmaint.data.load import RUL_COLUMN, SENSOR_COLUMNS, SETTING_COLUMNS, load_fd001
from predmaint.data.validate import FD001_EXPECTED, ExpectedFigures, validate_fd001
from predmaint.features.build import (
    build_features,
    feature_names,
    last_rows,
    select_sensors,
    sensor_stats,
)
from predmaint.models.artifacts import SCHEMA_VERSION, save_artifacts
from predmaint.models.metrics import mae, nasa_score, rmse
from predmaint.models.predict import predict_from_features
from predmaint.models.registry import (
    BASELINE_PARAMS,
    ESTIMATOR_TAGS,
    MODEL_NAMES,
    PRIMARY_SEARCH_GRID,
    make_model,
)
from predmaint.models.selection import promote

Echo = Callable[[str], None]


def group_folds(groups: np.ndarray, n_splits: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, val_idx)`` from ``GroupKFold``: no group is in both sides."""
    placeholder = np.zeros(len(groups))
    yield from GroupKFold(n_splits=n_splits).split(placeholder, groups=groups)


def cross_validate(
    pipeline: Pipeline,
    features: pd.DataFrame,
    y: np.ndarray,
    n_splits: int,
    rul_cap: int | None,
) -> dict[str, Any]:
    """GroupKFold-by-unit CV; returns per-fold and mean/std (ddof=1) MAE and RMSE."""
    groups = features["unit"].to_numpy()
    folds = []
    for fold, (train_idx, val_idx) in enumerate(group_folds(groups, n_splits), start=1):
        model = clone(pipeline).fit(features.iloc[train_idx], y[train_idx])
        pred = predict_from_features(model, features.iloc[val_idx], rul_cap)
        folds.append(
            {
                "fold": fold,
                "n_val_units": int(len(np.unique(groups[val_idx]))),
                "n_val_rows": int(len(val_idx)),
                "mae": mae(y[val_idx], pred),
                "rmse": rmse(y[val_idx], pred),
            }
        )
    maes = np.array([f["mae"] for f in folds])
    rmses = np.array([f["rmse"] for f in folds])
    return {
        "mae_mean": float(maes.mean()),
        "mae_std": float(maes.std(ddof=1)),
        "rmse_mean": float(rmses.mean()),
        "rmse_std": float(rmses.std(ddof=1)),
        "folds": folds,
    }


def evaluate_on_test(
    pipeline: Pipeline, test_features: pd.DataFrame, rul_truth: pd.Series, rul_cap: int | None
) -> dict[str, Any]:
    """Predict at each test unit's last cycle and score against ``RUL_FD00x`` (row i ↔ unit i)."""
    last = last_rows(test_features)
    truth = rul_truth.to_numpy(dtype=float)
    if len(last) != len(truth) or not np.array_equal(last["unit"], np.arange(1, len(truth) + 1)):
        raise ValueError(
            f"{len(truth)} RUL values but feature rows for units {last['unit'].tolist()[:5]}…; "
            "every test unit needs at least window_size cycles"
        )
    pred = predict_from_features(pipeline, last, rul_cap)
    capped = np.minimum(truth, rul_cap) if rul_cap is not None else truth
    return {
        "test": {
            "mae": mae(truth, pred),
            "rmse": rmse(truth, pred),
            "nasa_score": nasa_score(truth, pred),
        },
        "test_clipped_truth": {"mae": mae(capped, pred), "rmse": rmse(capped, pred)},
        "n_test_units": int(len(truth)),
    }


def search_primary(
    columns: Sequence[str],
    features: pd.DataFrame,
    y: np.ndarray,
    *,
    grid: Sequence[Mapping[str, Any]],
    seed: int,
    n_splits: int,
    rul_cap: int | None,
    echo: Echo,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """ADR-026: score each configuration by GroupKFold CV; pick the lowest CV RMSE mean
    (tie-break: CV MAE mean, then grid order). The test set is never used here.

    Returns ``(best_params, best_cv, results)``.
    """
    results = []
    for i, params in enumerate(grid, start=1):
        started = time.perf_counter()
        cv = cross_validate(
            make_model("primary", columns, seed, params), features, y, n_splits, rul_cap
        )
        results.append({"params": dict(params), "cv": cv})
        echo(
            f"  [{i:2}/{len(grid)}] {dict(params)}  CV RMSE {cv['rmse_mean']:.3f} ± "
            f"{cv['rmse_std']:.3f}  MAE {cv['mae_mean']:.3f}  "
            f"({time.perf_counter() - started:.1f} s)"
        )
    best = min(results, key=lambda r: (r["cv"]["rmse_mean"], r["cv"]["mae_mean"]))
    return best["params"], best["cv"], results


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _verify_checksums(cmapss_dir: Path) -> None:
    for name, expected in CMAPSS_SPEC.verified_files.items():
        path = cmapss_dir / name
        actual = sha256_file(path) if path.is_file() else "missing"
        if actual != expected:
            raise ChecksumMismatchError(
                f"{path}: SHA-256 {actual} != documented {expected} (docs/04 §3); "
                "re-run `python scripts/download_data.py`"
            )


def run_training(
    model_name: str,
    cmapss_dir: Path,
    model_dir: Path,
    reports_dir: Path,
    *,
    window_size: int = WINDOW_SIZE,
    rul_cap: int | None = RUL_CAP,
    seed: int = RANDOM_SEED,
    cv_folds: int = CV_FOLDS,
    verify_checksums: bool = True,
    expected: ExpectedFigures | None = FD001_EXPECTED,
    search_grid: Sequence[Mapping[str, Any]] = PRIMARY_SEARCH_GRID,
    echo: Echo = print,
) -> dict[str, Any]:
    """Train, cross-validate and test-evaluate one candidate, then save its artifacts.

    Returns the metadata dict that was written to ``<model_dir>/<model_name>_metadata.json``.

    Raises:
        ChecksumMismatchError: if ``verify_checksums`` and the FD001 files differ from docs/04.
        DataValidationError: if the data violates the contract.
        ValueError: for an unknown ``model_name``.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(f"unknown model {model_name!r}; available: {', '.join(MODEL_NAMES)}")
    started = time.perf_counter()
    if verify_checksums:
        _verify_checksums(cmapss_dir)
    data = load_fd001(cmapss_dir)
    validate_fd001(data, expected)

    stats = sensor_stats(data.train)
    sensors = select_sensors(data.train)
    dropped = [s for s in SENSOR_COLUMNS if s not in sensors]
    columns = feature_names(sensors)
    echo(f"Sensor selection on train (drop if <= 2 unique values): dropped {dropped}")
    for sensor in dropped:
        echo(
            f"  dropped {sensor:5} nunique={stats.at[sensor, 'nunique']:<3} "
            f"std={stats.at[sensor, 'std']:.3g}"
        )
    echo(f"Kept {len(sensors)} sensors: {sensors}")
    echo(
        f"Features: {len(columns)} (cycle + 3 x {len(sensors)}), "
        f"window_size={window_size}, rul_cap={rul_cap}"
    )

    labelled = add_rul(data.train, rul_cap)
    train_features = build_features(labelled, sensors, window_size, keep=[RUL_LABEL])
    y = train_features[RUL_LABEL].to_numpy(dtype=float)
    echo(
        f"Training rows: {len(train_features)} of {len(data.train)} "
        f"(rows without a full {window_size}-cycle window dropped), "
        f"{train_features['unit'].nunique()} units"
    )

    search: dict[str, Any] | None = None
    if model_name == "primary":
        echo(
            f"Bounded search (ADR-026): {len(search_grid)} configurations, "
            f"scored by {cv_folds}-fold GroupKFold CV RMSE"
        )
        params, cv, results = search_primary(
            columns,
            train_features,
            y,
            grid=search_grid,
            seed=seed,
            n_splits=cv_folds,
            rul_cap=rul_cap,
            echo=echo,
        )
        search = {
            "scored_by": "cv_rmse_mean (tie-break cv_mae_mean)",
            "n_configurations": len(results),
            "grid": [r["params"] for r in results],
            "results": [
                {"params": r["params"], **{k: v for k, v in r["cv"].items() if k != "folds"}}
                for r in results
            ],
            "selected_params": params,
        }
        echo(f"Selected configuration: {params}")
    else:
        params = dict(BASELINE_PARAMS)
        cv = None
    pipeline = make_model(model_name, columns, seed, params)
    if cv is None:
        cv = cross_validate(pipeline, train_features, y, cv_folds, rul_cap)
    echo(f"{cv_folds}-fold GroupKFold CV by unit (capped labels):")
    for fold in cv["folds"]:
        echo(
            f"  fold {fold['fold']}: units={fold['n_val_units']:3} rows={fold['n_val_rows']:5} "
            f"MAE={fold['mae']:.3f} RMSE={fold['rmse']:.3f}"
        )
    echo(
        f"  CV MAE  = {cv['mae_mean']:.3f} ± {cv['mae_std']:.3f}\n"
        f"  CV RMSE = {cv['rmse_mean']:.3f} ± {cv['rmse_std']:.3f}"
    )

    fitted = clone(pipeline).fit(train_features, y)
    test_features = build_features(data.test, sensors, window_size)
    evaluation = evaluate_on_test(fitted, test_features, data.rul[RUL_COLUMN], rul_cap)
    test = evaluation["test"]
    echo(
        f"Official test set ({evaluation['n_test_units']} units, last observed cycle, "
        f"unclipped truth): MAE={test['mae']:.3f} RMSE={test['rmse']:.3f} "
        f"NASA score={test['nasa_score']:.1f}"
    )
    clipped = evaluation["test_clipped_truth"]
    echo(f"  (truth clipped at {rul_cap}: MAE={clipped['mae']:.3f} RMSE={clipped['rmse']:.3f})")

    trained_at = datetime.now(UTC).replace(microsecond=0)
    git_status = _git("status", "--porcelain")
    metrics = {
        "cv": cv,
        "test": test,
        "test_clipped_truth": clipped,
        "n_train_rows": int(len(train_features)),
        "n_test_units": evaluation["n_test_units"],
    }
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_name": model_name,
        "estimator": ESTIMATOR_TAGS[model_name],
        "model_version": f"{ESTIMATOR_TAGS[model_name]}-{trained_at:%Y%m%dT%H%M%SZ}",
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
        "data": {
            "subset": DATASET_SUBSET,
            **{
                f"{kind}_sha256": sha256_file(cmapss_dir / f"{prefix}_{DATASET_SUBSET}.txt")
                for kind, prefix in (("train", "train"), ("test", "test"), ("rul", "RUL"))
            },
        },
        "config": {
            "rul_cap": rul_cap,
            "window_size": window_size,
            "random_seed": seed,
            "cv_folds": cv_folds,
        },
        "required_sensors": sensors,
        "dropped_columns": SETTING_COLUMNS + dropped,
        "feature_names": columns,
        "training_ranges": {
            s: {"min": float(data.train[s].min()), "max": float(data.train[s].max())}
            for s in sensors
        },
        "health_bands": dict(HEALTH_BANDS),
        "hyperparameters": params,
        "search": search,
        "metrics": metrics,
    }
    model_path, metadata_path = save_artifacts(fitted, metadata, model_dir, model_name)
    report_keys = (
        "model_name",
        "estimator",
        "model_version",
        "trained_at",
        "git_sha",
        "git_dirty",
        "data",
        "config",
        "required_sensors",
        "dropped_columns",
        "hyperparameters",
        "search",
        "metrics",
    )
    report = {key: metadata[key] for key in report_keys}
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"metrics_{model_name}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    echo(f"Saved {model_path}, {metadata_path}, {report_path}")
    echo(f"Elapsed: {time.perf_counter() - started:.1f} s")
    return metadata


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint used by ``scripts/train.py``. Returns a process exit code.

    Usage: ``--model baseline``, ``--model primary``, ``--select``, then ``--model intervals``,
    ``--model anomaly`` and ``--explain``.
    """
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Train/evaluate an FD001 RUL candidate (--model) or select and promote the "
        "served model from trained candidates (--select, ADR-024)."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--model", choices=(*MODEL_NAMES, "intervals", "anomaly"))
    action.add_argument("--select", action="store_true")
    action.add_argument(
        "--explain", action="store_true", help="permutation feature importance (ADR-040)"
    )
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--model-dir", type=Path, default=settings.model_dir)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    try:
        if args.select:
            promote(args.model_dir, args.reports_dir)
        elif args.explain:
            from predmaint.models.explain import run_explanation

            run_explanation(args.data_dir / "raw" / "CMAPSSData", args.model_dir, args.reports_dir)
        elif args.model == "anomaly":
            from predmaint.models.anomaly import run_anomaly_training

            run_anomaly_training(
                args.data_dir / "raw" / "CMAPSSData", args.model_dir, args.reports_dir
            )
        elif args.model == "intervals":
            from predmaint.models.intervals import run_interval_training

            run_interval_training(
                args.data_dir / "raw" / "CMAPSSData", args.model_dir, args.reports_dir
            )
        else:
            run_training(
                args.model, args.data_dir / "raw" / "CMAPSSData", args.model_dir, args.reports_dir
            )
    except (FileNotFoundError, ValueError, ChecksumMismatchError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
