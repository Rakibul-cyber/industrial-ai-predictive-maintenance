"""Permutation feature importance of the served model (ADR-040, Milestone 12).

The served point-model ``Pipeline`` is used as loaded (no refit). Importance = increase in the
RMSE of the *served* predictions (clamped to ``[0, rul_cap]``) when one feature column is
permuted, mean ± std over ``n_repeats``. Two populations: all training rows (in-sample; ranking)
and the 100 official test rows at the last cycle (out-of-sample; noisy). The Ridge baseline's
standardized coefficients give a second, model-independent view of which sensors matter.
"""

import json
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from predmaint.data.download import sha256_file
from predmaint.data.labels import RUL_LABEL, add_rul
from predmaint.data.load import RUL_COLUMN, load_fd001
from predmaint.data.validate import FD001_EXPECTED, ExpectedFigures, validate_fd001
from predmaint.features.build import build_features, last_rows
from predmaint.models.artifacts import load_artifacts
from predmaint.models.predict import FEATURE_IMPORTANCE_FILE, load_model
from predmaint.models.registry import pipeline_feature_names

N_REPEATS = 10
TOP_FEATURES = 10
FEATURE_TYPES: tuple[str, ...] = ("cycle", "raw sensor", "rolling mean", "rolling std")
CAVEATS: tuple[str, ...] = (
    "Permutation importance measures how much the served model relies on a feature, not a "
    "causal effect.",
    "Correlated features share importance: permuting one of a sensor's raw/mean/std columns or "
    "one of several correlated sensors can understate the group (hence per-sensor sums).",
    "Training population: in-sample, so the RMSE level is optimistic; use it for ranking.",
    "Test population: 100 engines at their last cycle, out-of-sample but noisy; see the std "
    "across repeats.",
    "HistGradientBoostingRegressor has no native feature_importances_ in scikit-learn.",
)
Echo = Callable[[str], None]


def sensor_of(feature: str) -> str | None:
    """``"s_7_roll_mean"`` → ``"s_7"``; ``"cycle"`` → ``None``."""
    if feature == "cycle":
        return None
    for suffix in ("_roll_mean", "_roll_std"):
        if feature.endswith(suffix):
            return feature[: -len(suffix)]
    return feature


def feature_type(feature: str) -> str:
    """One of :data:`FEATURE_TYPES`."""
    if feature == "cycle":
        return "cycle"
    if feature.endswith("_roll_mean"):
        return "rolling mean"
    if feature.endswith("_roll_std"):
        return "rolling std"
    return "raw sensor"


def served_rmse_scorer(rul_cap: int | None) -> Callable[[Pipeline, pd.DataFrame, Any], float]:
    """Negative RMSE of predictions clamped to ``[0, rul_cap]`` (the served output, ADR-027)."""
    upper = np.inf if rul_cap is None else float(rul_cap)

    def score(estimator: Pipeline, x: pd.DataFrame, y: Any) -> float:
        predicted = np.clip(estimator.predict(x), 0.0, upper)
        return -float(np.sqrt(np.mean((np.asarray(y, dtype=float) - predicted) ** 2)))

    return score


def aggregate(
    importances: np.ndarray, features: Sequence[str], groups: Mapping[str, str | None]
) -> list[dict[str, Any]]:
    """Sum per-repeat importances within each group, then mean ± std over repeats.

    ``importances`` has shape ``(n_features, n_repeats)``; features mapped to ``None`` are
    skipped. Returned sorted by mean, descending.
    """
    rows = []
    names = sorted({g for g in groups.values() if g is not None})
    for group in names:
        idx = [i for i, f in enumerate(features) if groups[f] == group]
        per_repeat = importances[idx].sum(axis=0)
        rows.append(
            {
                "name": group,
                "n_features": len(idx),
                "mean": float(per_repeat.mean()),
                "std": float(per_repeat.std()),
            }
        )
    return sorted(rows, key=lambda r: r["mean"], reverse=True)


def population_importance(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: np.ndarray,
    rul_cap: int | None,
    *,
    n_repeats: int = N_REPEATS,
    seed: int = 42,
    n_jobs: int | None = -1,
) -> dict[str, Any]:
    """Permutation importance of ``pipeline`` on ``x`` (only the model's feature columns)."""
    features = list(x.columns)
    scorer = served_rmse_scorer(rul_cap)
    result = permutation_importance(
        pipeline,
        x,
        y,
        scoring=scorer,
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=n_jobs,
    )
    importances = np.asarray(result.importances, dtype=float)
    per_feature = sorted(
        (
            {
                "name": f,
                "sensor": sensor_of(f),
                "type": feature_type(f),
                "mean": float(importances[i].mean()),
                "std": float(importances[i].std()),
            }
            for i, f in enumerate(features)
        ),
        key=lambda r: r["mean"],
        reverse=True,
    )
    return {
        "n_rows": int(len(x)),
        "baseline_rmse": -scorer(pipeline, x, y),
        "per_feature": per_feature,
        "per_sensor": aggregate(importances, features, {f: sensor_of(f) for f in features}),
        "per_feature_type": aggregate(
            importances, features, {f: feature_type(f) for f in features}
        ),
    }


def ridge_sensor_weights(pipeline: Pipeline) -> list[dict[str, Any]]:
    """Ridge |coefficient| on standardized features, summed per sensor (``cycle`` excluded)."""
    features = pipeline_feature_names(pipeline)
    coef = np.abs(np.asarray(pipeline.named_steps["model"].coef_, dtype=float).ravel())
    totals: dict[str, float] = {}
    for f, c in zip(features, coef, strict=True):
        sensor = sensor_of(f)
        if sensor is not None:
            totals[sensor] = totals.get(sensor, 0.0) + float(c)
    return sorted(
        ({"name": s, "abs_coef_sum": v} for s, v in totals.items()),
        key=lambda r: r["abs_coef_sum"],
        reverse=True,
    )


def rank_agreement(ridge: list[dict[str, Any]], per_sensor: list[dict[str, Any]]) -> float | None:
    """Spearman correlation between Ridge per-sensor |coef| and permutation per-sensor mean;
    ``None`` when either side is constant (correlation undefined)."""
    importance = {r["name"]: r["mean"] for r in per_sensor}
    a = np.array([r["abs_coef_sum"] for r in ridge], dtype=float)
    b = np.array([importance[r["name"]] for r in ridge], dtype=float)
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return None
    return float(spearmanr(a, b).statistic)


def _baseline_pipeline(
    model_dir: Path,
    cmapss_dir: Path,
    shas: Mapping[str, str],
    expected: ExpectedFigures | None,
    echo: Echo,
) -> tuple[Pipeline, str, str]:
    """The Ridge baseline: the local candidate if trained on the same data, else retrained
    (deterministic) into a temporary directory. ``models/`` is never written."""
    from predmaint.models.train import run_training

    try:
        pipeline, meta = load_artifacts(model_dir, "baseline")
    except (FileNotFoundError, ValueError):
        pipeline, meta = None, {}
    if pipeline is not None and all(meta.get("data", {}).get(k) == v for k, v in shas.items()):
        return pipeline, str(meta["model_version"]), f"{model_dir / 'baseline.joblib'}"
    echo("Ridge baseline not available locally for these data: retraining it into a temp dir")
    with tempfile.TemporaryDirectory() as tmp:
        run_training(
            "baseline",
            cmapss_dir,
            Path(tmp) / "models",
            Path(tmp) / "reports",
            verify_checksums=False,
            expected=expected,
            echo=lambda _m: None,
        )
        pipeline, meta = load_artifacts(Path(tmp) / "models", "baseline")
    return pipeline, str(meta["model_version"]), "retrained (temporary directory)"


def run_explanation(
    cmapss_dir: Path,
    model_dir: Path,
    reports_dir: Path,
    *,
    verify_checksums: bool = True,
    expected: ExpectedFigures | None = FD001_EXPECTED,
    n_repeats: int = N_REPEATS,
    n_jobs: int | None = -1,
    echo: Echo = print,
) -> dict[str, Any]:
    """Compute and save permutation importance for the served model in ``model_dir``.

    Writes ``<reports_dir>/feature_importance.json`` and an identical
    ``<model_dir>/model_feature_importance.json`` (read by the API).

    Raises:
        FileNotFoundError: if the served model metadata is missing (run --select first).
        ValueError: if the data differ from the served model's training data.
    """
    from predmaint.models.train import _git, _verify_checksums

    started = time.perf_counter()
    if not (model_dir / "model_metadata.json").is_file():
        raise FileNotFoundError(
            f"{model_dir / 'model_metadata.json'} not found: train and select the point model "
            "first (scripts/train.py --model baseline/primary, then --select)"
        )
    point = load_model(model_dir / "model.joblib")
    meta, config = point.metadata, point.metadata["config"]
    sensors, features = list(meta["required_sensors"]), list(meta["feature_names"])
    window, rul_cap, seed = config["window_size"], config["rul_cap"], config["random_seed"]

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

    train = build_features(add_rul(data.train, rul_cap), sensors, window, keep=[RUL_LABEL])
    test = last_rows(build_features(data.test, sensors, window))
    kwargs = {"n_repeats": n_repeats, "seed": seed, "n_jobs": n_jobs}
    echo(
        f"Permutation importance for served model {point.model_version}: {len(features)} "
        f"features, n_repeats={n_repeats}, seed={seed}, n_jobs={n_jobs}, scoring = RMSE of "
        f"served (clamped) predictions"
    )
    t0 = time.perf_counter()
    train_pop = population_importance(
        point.pipeline, train[features], train[RUL_LABEL].to_numpy(dtype=float), rul_cap, **kwargs
    )
    train_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    test_pop = population_importance(
        point.pipeline,
        test[features],
        data.rul[RUL_COLUMN].to_numpy(dtype=float),
        rul_cap,
        **kwargs,
    )
    test_seconds = time.perf_counter() - t0
    train_pop.update(
        target=f"capped training RUL (min(RUL, {rul_cap})), the training target",
        caveat="in-sample: RMSE level optimistic, informative for ranking",
    )
    test_pop.update(
        target="unclipped true RUL (RUL_FD001), the official test metric",
        caveat="out-of-sample but only 100 rows: noisy, see std across repeats",
    )

    t0 = time.perf_counter()
    baseline, baseline_version, baseline_source = _baseline_pipeline(
        model_dir, cmapss_dir, shas, expected, echo
    )
    ridge = ridge_sensor_weights(baseline)
    ridge_seconds = time.perf_counter() - t0
    report: dict[str, Any] = {
        "artifact": "feature_importance",
        "model_version": point.model_version,
        "for_model_version": point.model_version,
        "computed_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "git_sha": _git("rev-parse", "HEAD") or "unknown",
        "method": (
            "sklearn.inspection.permutation_importance on the served Pipeline (no refit); "
            "importance = increase in RMSE (cycles) of served predictions clamped to "
            "[0, rul_cap] when one feature column is permuted; mean ± std over repeats; "
            "per-sensor and per-type values sum the features' importances within each repeat"
        ),
        "scikit_learn": sklearn.__version__,
        "n_repeats": n_repeats,
        "seed": seed,
        "n_jobs": n_jobs,
        "n_features": len(features),
        "feature_types": list(FEATURE_TYPES),
        "data": {"subset": meta["data"]["subset"], **shas},
        "populations": {"train": train_pop, "test": test_pop},
        "ridge_comparison": {
            "baseline_model_version": baseline_version,
            "source": baseline_source,
            "method": "abs(Ridge coefficient) on standardized features, summed per sensor",
            "per_sensor": ridge,
            "spearman_vs_permutation_per_sensor": {
                "train": rank_agreement(ridge, train_pop["per_sensor"]),
                "test": rank_agreement(ridge, test_pop["per_sensor"]),
            },
        },
        "caveats": list(CAVEATS),
        "timings_seconds": {
            "train_population": round(train_seconds, 1),
            "test_population": round(test_seconds, 1),
            "ridge": round(ridge_seconds, 1),
        },
    }
    _echo_report(echo, report)
    text = json.dumps(report, indent=2) + "\n"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "feature_importance.json"
    sidecar_path = model_dir / FEATURE_IMPORTANCE_FILE
    report_path.write_text(text, encoding="utf-8")
    sidecar_path.write_text(text, encoding="utf-8")
    echo(f"Saved {report_path} and {sidecar_path} (identical)")
    echo(f"Elapsed: {time.perf_counter() - started:.1f} s")
    return report


def _echo_report(echo: Echo, report: dict[str, Any]) -> None:
    train, test = report["populations"]["train"], report["populations"]["test"]
    echo(
        f"Baseline RMSE (served, clamped): train {train['baseline_rmse']:.3f} "
        f"(n={train['n_rows']:,}, capped target) | test {test['baseline_rmse']:.3f} "
        f"(n={test['n_rows']}, unclipped truth)"
    )
    echo("Per sensor (RMSE increase in cycles, mean ± std over repeats):")
    echo(f"  {'rank':<5}{'train':<26}{'test':<26}")
    for i, (a, b) in enumerate(zip(train["per_sensor"], test["per_sensor"], strict=True), 1):
        echo(
            f"  {i:<5}{a['name']:<7}{a['mean']:>7.3f} ± {a['std']:<7.3f}    "
            f"{b['name']:<7}{b['mean']:>7.3f} ± {b['std']:<7.3f}"
        )
    echo("Per feature type:")
    test_types = {r["name"]: r for r in test["per_feature_type"]}
    for r in train["per_feature_type"]:
        t = test_types[r["name"]]
        echo(
            f"  {r['name']:<13} (n={r['n_features']:>2})  train {r['mean']:>7.3f} ± "
            f"{r['std']:<6.3f}  test {t['mean']:>7.3f} ± {t['std']:.3f}"
        )
    echo("Top 10 features (test population):")
    for r in test["per_feature"][:TOP_FEATURES]:
        echo(f"  {r['name']:<15} {r['mean']:>7.3f} ± {r['std']:.3f}")
    ridge = report["ridge_comparison"]
    order = ", ".join(r["name"] for r in ridge["per_sensor"])
    rho = {
        k: "undefined" if v is None else f"{v:.3f}"
        for k, v in ridge["spearman_vs_permutation_per_sensor"].items()
    }
    echo(
        f"Ridge baseline {ridge['baseline_model_version']} ({ridge['source']}): per-sensor "
        f"|coef| order: {order}"
    )
    echo(
        f"Spearman rank agreement (Ridge |coef| vs permutation, per sensor): "
        f"train {rho['train']}, test {rho['test']}"
    )
    t = report["timings_seconds"]
    echo(
        f"Timing: train population {t['train_population']} s, test population "
        f"{t['test_population']} s, Ridge {t['ridge']} s"
    )
