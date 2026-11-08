"""Minimal unsupervised anomaly detection (ADR-035, Milestone 9).

"Anomaly" = deviation from a healthy-state reference: training rows with a full trailing window
and true (uncapped) RUL >= RUL_CAP. Features are the kept sensors' rolling means from the shared
feature function. Score = Mahalanobis distance under a Ledoit-Wolf covariance of standardised
healthy rows; threshold = 99th percentile of OUT-OF-FOLD healthy-row scores (GroupKFold by
engine). With no labels, the evaluation asks whether the score tracks degradation.
"""

import json
import platform
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from predmaint.data.download import sha256_file
from predmaint.data.labels import RUL_LABEL, add_rul
from predmaint.data.load import RUL_COLUMN, load_fd001
from predmaint.data.validate import FD001_EXPECTED, ExpectedFigures, validate_fd001
from predmaint.features.build import build_features, last_rows
from predmaint.models.artifacts import SCHEMA_VERSION, save_artifacts
from predmaint.models.predict import (
    ANOMALY_ARTIFACT,
    detector_scores,
    ewm_smooth,
    load_model,
    predict_from_features,
)

THRESHOLD_QUANTILE = 0.99
CONSECUTIVE = 3
TREND_ACCEPTANCE = 0.7  # FR-019: median per-engine Spearman on train OOF
FALLBACK_BELOW = 0.5  # ADR-035: try IsolationForest once if the median is below this
BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0-20", -np.inf, 20.5),
    ("21-60", 20.5, 60.5),
    ("61-125", 60.5, 125.5),
    (">125", 125.5, np.inf),
)
RAW_RUL = "rul_raw"
#: ADR-038 / FR-021: one half-life, fixed before any smoothed number was computed (not tuned).
SMOOTHING: dict[str, Any] = {"method": "ewm", "halflife": 10, "adjust": True}
FALSE_ALARM_LIMIT = 0.02  # FR-021 (b)
#: ADR-039: FR-021 was not met and smoothing delays detection, so the API serves the raw score.
#: The smoothed threshold and evaluation stay in the metadata as a documented negative result.
SMOOTHING_ENABLED = False
TRUNCATION_GROUPS: tuple[tuple[str, float, float], ...] = (
    ("<=60", -np.inf, 60.5),
    ("61-125", 60.5, 125.5),
    (">125", 125.5, np.inf),
)
Echo = Callable[[str], None]


def anomaly_feature_names(sensors: Sequence[str]) -> list[str]:
    """The kept sensors' rolling-mean columns (denoised; produced by ``features.build``)."""
    return [f"{s}_roll_mean" for s in sensors]


def fit_detector(
    healthy: pd.DataFrame,
    features: Sequence[str],
    method: str = "mahalanobis_ledoitwolf",
    seed: int = 42,
) -> dict[str, Any]:
    """Fit a detector on healthy rows. The threshold is set separately (out-of-fold)."""
    x = healthy[list(features)].to_numpy(dtype=float)
    if method == "mahalanobis_ledoitwolf":
        scaler = StandardScaler().fit(x)
        covariance = LedoitWolf().fit(scaler.transform(x))
        return {
            "method": method,
            "scaler": scaler,
            "covariance": covariance,
            "features": list(features),
            "threshold": np.nan,
        }
    if method == "isolation_forest":
        model = IsolationForest(random_state=seed).fit(x)
        return {"method": method, "model": model, "features": list(features), "threshold": np.nan}
    raise ValueError(f"unknown method {method!r}")


def oof_scores(
    rows: pd.DataFrame,
    healthy_mask: np.ndarray,
    features: Sequence[str],
    n_splits: int,
    method: str = "mahalanobis_ledoitwolf",
    seed: int = 42,
) -> np.ndarray:
    """Out-of-fold raw scores for every row: each fold's detector is fitted on the healthy rows
    of the other folds' engines only (GroupKFold by ``unit``)."""
    from predmaint.models.train import group_folds

    scores = np.empty(len(rows))
    for train_idx, val_idx in group_folds(rows["unit"].to_numpy(), n_splits):
        fit_rows = rows.iloc[train_idx][healthy_mask[train_idx]]
        detector = fit_detector(fit_rows, features, method, seed)
        scores[val_idx] = detector_scores(detector, rows.iloc[val_idx])
    return scores


def first_detection(cycles: np.ndarray, normalized: np.ndarray, consecutive: int = CONSECUTIVE):
    """First cycle starting a run of ``consecutive`` cycles with ``normalized > 1`` (or None)."""
    above = np.asarray(normalized) > 1.0
    run = 0
    for i, flag in enumerate(above):
        run = run + 1 if flag else 0
        if run == consecutive:
            return int(np.asarray(cycles)[i - consecutive + 1])
    return None


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if values.size == 0:
        return {"median": None, "p25": None, "p75": None}
    p25, med, p75 = np.percentile(values, [25, 50, 75])
    return {"median": float(med), "p25": float(p25), "p75": float(p75)}


def _spearman_or_none(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman correlation, or ``None`` when either input is constant (undefined)."""
    if len(a) < 3 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return None
    return float(spearmanr(a, b).statistic)


def engine_trends(rows: pd.DataFrame, normalized: np.ndarray) -> pd.Series:
    """Spearman(normalized score, cycle) per engine (NaN for < 3 points or a constant score)."""
    frame = pd.DataFrame(
        {"unit": rows["unit"].to_numpy(), "cycle": rows["cycle"].to_numpy(), "score": normalized}
    )
    rhos = {}
    for unit, g in frame.groupby("unit"):
        if len(g) < 3 or g["score"].nunique() < 2:
            rhos[unit] = np.nan
            continue
        rhos[unit] = float(spearmanr(g["cycle"], g["score"]).statistic)
    return pd.Series(rhos, dtype=float)


def per_engine_trend(rows: pd.DataFrame, normalized: np.ndarray) -> dict[str, Any]:
    """Per-engine Spearman(normalized score, cycle): median, IQR, share > 0.7."""
    rhos_arr = engine_trends(rows, normalized).to_numpy()
    valid = rhos_arr[np.isfinite(rhos_arr)]
    return {
        "n_engines": int(len(rhos_arr)),
        "n_engines_scored": int(valid.size),
        **_quantiles(valid),
        "share_above_0_7": float(np.mean(valid > TREND_ACCEPTANCE)) if valid.size else None,
    }


def detection_timing(rows: pd.DataFrame, normalized: np.ndarray) -> dict[str, Any]:
    """Train engines (run to failure): first sustained detection as fraction of life and RUL."""
    frame = pd.DataFrame(
        {"unit": rows["unit"].to_numpy(), "cycle": rows["cycle"].to_numpy(), "score": normalized}
    )
    fractions, ruls, never = [], [], 0
    for _, g in frame.groupby("unit"):
        life = int(rows.loc[g.index, RAW_RUL].iloc[0] + g["cycle"].iloc[0])  # max_cycle
        cycle = first_detection(g["cycle"].to_numpy(), g["score"].to_numpy())
        if cycle is None:
            never += 1
            continue
        fractions.append(cycle / life)
        ruls.append(life - cycle)
    n = frame["unit"].nunique()
    return {
        "consecutive_cycles": CONSECUTIVE,
        "n_engines": int(n),
        "never_flagged_fraction": float(never / n),
        "fraction_of_life": _quantiles(np.asarray(fractions)),
        "true_rul_at_detection": _quantiles(np.asarray(ruls)),
    }


def trend_by_true_rul(
    rows: pd.DataFrame, normalized: np.ndarray, truth_by_unit: pd.Series
) -> list[dict[str, Any]]:
    """Test-engine trend grouped by true RUL at cutoff (the truncation effect, 16 §10)."""
    rhos = engine_trends(rows, normalized)
    truth = truth_by_unit.reindex(rhos.index).to_numpy(dtype=float)
    out = []
    for label, lo, hi in TRUNCATION_GROUPS:
        values = rhos.to_numpy()[(truth > lo) & (truth <= hi)]
        values = values[np.isfinite(values)]
        out.append(
            {
                "true_rul_group": label,
                "n_engines_scored": int(values.size),
                "median": float(np.median(values)) if values.size else None,
            }
        )
    return out


def smooth_per_engine(rows: pd.DataFrame, scores: np.ndarray, halflife: float) -> np.ndarray:
    """EWM of raw scores within each engine, in cycle order; returned in ``rows`` order."""
    frame = pd.DataFrame(
        {
            "unit": rows["unit"].to_numpy(),
            "cycle": rows["cycle"].to_numpy(),
            "score": np.asarray(scores, dtype=float),
        }
    ).reset_index(drop=True)
    out = np.empty(len(frame))
    for _, g in frame.groupby("unit"):
        ordered = g.sort_values("cycle")
        out[ordered.index.to_numpy()] = ewm_smooth(ordered["score"].to_numpy(), halflife)
    return out


def last_cycle_values(rows: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    """Value at each engine's last cycle, engines in ascending unit order."""
    frame = pd.DataFrame(
        {"unit": rows["unit"].to_numpy(), "cycle": rows["cycle"].to_numpy(), "v": values}
    )
    return frame.sort_values(["unit", "cycle"]).groupby("unit")["v"].last().to_numpy()


def flagged_by_bucket(truth: np.ndarray, normalized_last: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for label, lo, hi in BUCKETS:
        mask = (truth > lo) & (truth <= hi)
        n = int(mask.sum())
        out.append(
            {
                "true_rul_bucket": label,
                "n": n,
                "flagged_fraction": float(np.mean(normalized_last[mask] > 1)) if n else None,
            }
        )
    return out


def run_anomaly_training(
    cmapss_dir: Path,
    model_dir: Path,
    reports_dir: Path,
    *,
    verify_checksums: bool = True,
    expected: ExpectedFigures | None = FD001_EXPECTED,
    healthy_min_rul: int | None = None,
    echo: Echo = print,
) -> dict[str, Any]:
    """Fit, evaluate and save the anomaly detector for the served model in ``model_dir``.

    ``healthy_min_rul`` defaults to the served model's ``rul_cap`` (ADR-035); tests on short
    synthetic engines may lower it.

    Raises:
        FileNotFoundError: if the served model is missing (run --select first).
        ValueError: if the data differ from the served model's training data, or no healthy rows.
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
    sensors, window, rul_cap = (
        list(meta["required_sensors"]),
        config["window_size"],
        config["rul_cap"],
    )
    seed, n_splits = config["random_seed"], config["cv_folds"]
    healthy_min = rul_cap if healthy_min_rul is None else healthy_min_rul
    features = anomaly_feature_names(sensors)

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

    labelled = add_rul(data.train, None).rename(columns={RUL_LABEL: RAW_RUL})
    rows = build_features(labelled, sensors, window, keep=[RAW_RUL])
    healthy = rows[RAW_RUL].to_numpy() >= healthy_min
    if not healthy.any():
        raise ValueError(f"no healthy reference rows (true RUL >= {healthy_min})")
    echo(
        f"Anomaly detector for served model {point.model_version}: {len(features)} rolling-mean "
        f"features; healthy reference = true RUL >= {healthy_min}: {int(healthy.sum()):,} rows "
        f"from {rows.loc[healthy, 'unit'].nunique()} engines (of {len(rows):,} eligible rows)"
    )

    def evaluate(method: str) -> dict[str, Any]:
        raw = oof_scores(rows, healthy, features, n_splits, method, seed)
        threshold = float(np.quantile(raw[healthy], THRESHOLD_QUANTILE))
        normalized = raw / threshold
        return {
            "method": method,
            "threshold": threshold,
            "raw": raw,
            "normalized": normalized,
            "trend": per_engine_trend(rows, normalized),
        }

    t0 = time.perf_counter()
    primary = evaluate("mahalanobis_ledoitwolf")
    candidates = {"mahalanobis_ledoitwolf": primary}
    median = primary["trend"]["median"]
    fallback_note = f"not needed: median train-OOF Spearman {median:.3f} >= {FALLBACK_BELOW}"
    if primary["trend"]["median"] is None or primary["trend"]["median"] < FALLBACK_BELOW:
        candidates["isolation_forest"] = evaluate("isolation_forest")
        fallback_note = "IsolationForest tried once (ADR-035)"
    chosen = max(candidates.values(), key=lambda c: c["trend"]["median"] or -np.inf)
    method, threshold, normalized = chosen["method"], chosen["threshold"], chosen["normalized"]

    false_alarm = float(np.mean(normalized[healthy] > 1.0))
    timing = detection_timing(rows, normalized)
    oof_seconds = time.perf_counter() - t0

    detector = fit_detector(rows[healthy], features, method, seed)
    detector["threshold"] = threshold
    in_sample_threshold = float(
        np.quantile(detector_scores(detector, rows[healthy]), THRESHOLD_QUANTILE)
    )

    test_rows = build_features(data.test, sensors, window)
    test_norm = detector_scores(detector, test_rows) / threshold
    test_trend = per_engine_trend(test_rows, test_norm)
    last = last_rows(test_rows)
    last_norm = detector_scores(detector, last) / threshold
    truth = data.rul[RUL_COLUMN].to_numpy(dtype=float)
    point_pred = predict_from_features(point.pipeline, last, rul_cap)
    rho_rul = _spearman_or_none(last_norm, point_pred)
    truth_by_unit = pd.Series(truth, index=np.arange(1, len(truth) + 1))
    test_raw = detector_scores(detector, test_rows)
    if not np.allclose(last_cycle_values(test_rows, test_raw) / threshold, last_norm):
        raise RuntimeError("last-cycle scores are misaligned with the RUL file")  # invariant

    # --- ADR-038: smoothed score, same OOF scores, one pre-registered half-life ------------
    halflife = float(SMOOTHING["halflife"])
    oof_smoothed = smooth_per_engine(rows, chosen["raw"], halflife)
    s_threshold = float(np.quantile(oof_smoothed[healthy], THRESHOLD_QUANTILE))
    s_norm = oof_smoothed / s_threshold
    s_trend = per_engine_trend(rows, s_norm)
    s_false_alarm = float(np.mean(s_norm[healthy] > 1.0))
    test_s_norm = smooth_per_engine(test_rows, test_raw, halflife) / s_threshold
    last_s_norm = last_cycle_values(test_rows, test_s_norm)
    trend_met = bool((s_trend["median"] or -1) > TREND_ACCEPTANCE)
    false_alarm_met = bool(s_false_alarm <= FALSE_ALARM_LIMIT)
    smoothed = {
        "threshold": s_threshold,
        "threshold_check": {
            "oof_threshold": s_threshold,
            "raw_oof_threshold": threshold,
            "smoothed_below_raw": bool(s_threshold < threshold),
        },
        "evaluation": {
            "train_oof": {
                "trend": s_trend,
                "false_alarm_rate_healthy": s_false_alarm,
                "n_healthy_rows": int(healthy.sum()),
                "detection": detection_timing(rows, s_norm),
            },
            "test": {
                "trend": per_engine_trend(test_rows, test_s_norm),
                "trend_by_true_rul": trend_by_true_rul(test_rows, test_s_norm, truth_by_unit),
                "flagged_at_last_cycle": {
                    "overall": float(np.mean(last_s_norm > 1.0)),
                    "buckets": flagged_by_bucket(truth, last_s_norm),
                },
                "spearman_with_rul_prediction": _spearman_or_none(last_s_norm, point_pred),
            },
            "acceptance": {
                "criterion": (
                    f"median per-engine Spearman(smoothed score, cycle) on train OOF > "
                    f"{TREND_ACCEPTANCE} AND false-alarm rate on healthy OOF rows <= "
                    f"{FALSE_ALARM_LIMIT}"
                ),
                "trend_value": s_trend["median"],
                "trend_met": trend_met,
                "false_alarm_value": s_false_alarm,
                "false_alarm_met": false_alarm_met,
                "met": trend_met and false_alarm_met,
            },
        },
    }

    evaluation = {
        "train_oof": {
            "trend": chosen["trend"],
            "false_alarm_rate_healthy": false_alarm,
            "n_healthy_rows": int(healthy.sum()),
            "detection": timing,
        },
        "test": {
            "trend": test_trend,
            "flagged_at_last_cycle": {
                "overall": float(np.mean(last_norm > 1.0)),
                "buckets": flagged_by_bucket(truth, last_norm),
            },
            "spearman_with_rul_prediction": rho_rul,
            "trend_by_true_rul": trend_by_true_rul(test_rows, test_raw / threshold, truth_by_unit),
        },
        "acceptance": {
            "criterion": (
                f"median per-engine Spearman(score, cycle) on train OOF > {TREND_ACCEPTANCE}"
            ),
            "value": chosen["trend"]["median"],
            "met": bool((chosen["trend"]["median"] or -1) > TREND_ACCEPTANCE),
        },
        "fallback": {
            "decision": fallback_note,
            "candidates": {
                k: {"threshold": v["threshold"], "trend": v["trend"]} for k, v in candidates.items()
            },
            "chosen": method,
        },
        "threshold_check": {
            "oof_threshold": threshold,
            "in_sample_threshold_all_healthy_rows": in_sample_threshold,
        },
    }
    _echo_report(echo, evaluation, method, threshold, oof_seconds)
    _echo_side_by_side(echo, evaluation, smoothed)

    trained_at = datetime.now(UTC).replace(microsecond=0)
    sha = _git("rev-parse", "HEAD") or "unknown"
    git_status = _git("status", "--porcelain")
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact": ANOMALY_ARTIFACT,
        "model_version": f"anomaly-{trained_at:%Y%m%d}-{sha[:7] if sha != 'unknown' else sha}",
        "method": method,
        "for_model_version": point.model_version,
        "features": features,
        "threshold": threshold,
        "threshold_quantile": THRESHOLD_QUANTILE,
        "healthy_reference": (
            f"rows with a full {window}-cycle window and true RUL >= {healthy_min}"
        ),
        "healthy_min_rul": healthy_min,
        "healthy_rows": int(healthy.sum()),
        "trained_at": trained_at.isoformat().replace("+00:00", "Z"),
        "git_sha": sha,
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
        "evaluation": evaluation,
        "smoothing": dict(SMOOTHING),
        "smoothing_enabled": SMOOTHING_ENABLED,
        "smoothed": smoothed,
        "timings_seconds": {"oof_evaluation": round(oof_seconds, 2)},
    }
    model_path, metadata_path = save_artifacts(detector, metadata, model_dir, ANOMALY_ARTIFACT)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "metrics_anomaly.json"
    report = {k: v for k, v in metadata.items() if k != "features"}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    echo(f"Saved {model_path}, {metadata_path}, {report_path}")
    echo(f"Elapsed: {time.perf_counter() - started:.1f} s")
    return metadata


def _fmt(q: dict[str, float | None], digits: int = 3) -> str:
    if q["median"] is None:
        return "n/a"
    return f"median {q['median']:.{digits}f} (IQR {q['p25']:.{digits}f}–{q['p75']:.{digits}f})"


def _echo_report(
    echo: Echo, ev: dict[str, Any], method: str, threshold: float, seconds: float
) -> None:
    tr, te = ev["train_oof"], ev["test"]
    echo(
        f"Method: {method}; OOF threshold (q{THRESHOLD_QUANTILE}) = {threshold:.4f}; "
        f"in-sample threshold on all healthy rows = "
        f"{ev['threshold_check']['in_sample_threshold_all_healthy_rows']:.4f}"
    )
    echo(f"Fallback: {ev['fallback']['decision']}")
    for label, trend in (("train OOF", tr["trend"]), ("test", te["trend"])):
        share = "n/a" if trend["share_above_0_7"] is None else f"{trend['share_above_0_7']:.3f}"
        echo(
            f"[1] Trend Spearman(score, cycle), {label}: {_fmt(trend)}; share > 0.7 = {share} "
            f"({trend['n_engines_scored']}/{trend['n_engines']} engines scored)"
        )
    echo(
        f"[2] False-alarm rate on healthy rows (train OOF): {tr['false_alarm_rate_healthy']:.4f} "
        f"of {tr['n_healthy_rows']:,} rows"
    )
    d = tr["detection"]
    echo(
        f"[3] Detection (score > threshold for {d['consecutive_cycles']} consecutive cycles), "
        f"train OOF: fraction of life {_fmt(d['fraction_of_life'])}; true RUL at detection "
        f"{_fmt(d['true_rul_at_detection'], 1)}; never flagged {d['never_flagged_fraction']:.3f}"
    )
    flags = te["flagged_at_last_cycle"]
    parts = []
    for b in flags["buckets"]:
        value = b["flagged_fraction"]
        shown = "n/a" if value is None else f"{value:.3f}"
        parts.append(f"{b['true_rul_bucket']}: {shown} (n={b['n']})")
    parts = ", ".join(parts)
    echo(
        f"[4] Test engines flagged at last cycle: overall {flags['overall']:.3f}; "
        f"by true RUL: {parts}"
    )
    rho = te["spearman_with_rul_prediction"]
    shown = "undefined (constant input)" if rho is None else f"{rho:.3f}"
    echo(f"[5] Spearman(normalized score, RUL prediction) at test last cycle: {shown}")
    acc = ev["acceptance"]
    echo(
        f"FR-019 acceptance ({acc['criterion']}): value {acc['value']:.3f} → "
        f"{'MET' if acc['met'] else 'NOT MET'}"
    )
    echo(f"Timing: OOF evaluation {seconds:.2f} s")


def _echo_side_by_side(echo: Echo, raw: dict[str, Any], smoothed: dict[str, Any]) -> None:
    """Raw (M9) vs smoothed (ADR-038) evaluation, one line per metric."""
    sm = smoothed["evaluation"]
    pairs = [(raw, "raw"), (sm, "smoothed")]

    def num(value: float | None, digits: int = 3) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    echo(
        f"Side by side (smoothing: ewm halflife={SMOOTHING['halflife']}, adjust=True). "
        f"Thresholds (OOF q0.99): raw {smoothed['threshold_check']['raw_oof_threshold']:.4f} | "
        f"smoothed {smoothed['threshold']:.4f}"
    )
    echo(f"  {'metric':<52} {'raw':>22} {'smoothed':>22}")

    def line(label: str, values: list[str]) -> None:
        echo(f"  {label:<52} {values[0]:>22} {values[1]:>22}")

    line(
        "[1] train OOF trend median (IQR)",
        [
            f"{num(e['train_oof']['trend']['median'])} "
            f"({num(e['train_oof']['trend']['p25'])}–{num(e['train_oof']['trend']['p75'])})"
            for e, _ in pairs
        ],
    )
    line(
        "[1] train OOF share > 0.7",
        [num(e["train_oof"]["trend"]["share_above_0_7"]) for e, _ in pairs],
    )
    line("[1] test trend median", [num(e["test"]["trend"]["median"]) for e, _ in pairs])
    for i, group in enumerate(TRUNCATION_GROUPS):
        line(
            f"[1] test trend median, true RUL {group[0]} (n)",
            [
                f"{num(e['test']['trend_by_true_rul'][i]['median'])} "
                f"(n={e['test']['trend_by_true_rul'][i]['n_engines_scored']})"
                for e, _ in pairs
            ],
        )
    line(
        "[2] false-alarm rate, healthy OOF rows",
        [num(e["train_oof"]["false_alarm_rate_healthy"], 4) for e, _ in pairs],
    )
    line(
        "[3] detection: fraction of life, median",
        [num(e["train_oof"]["detection"]["fraction_of_life"]["median"]) for e, _ in pairs],
    )
    line(
        "[3] detection: true RUL, median (IQR)",
        [
            f"{num(d['median'], 1)} ({num(d['p25'], 1)}–{num(d['p75'], 1)})"
            for d in (e["train_oof"]["detection"]["true_rul_at_detection"] for e, _ in pairs)
        ],
    )
    line(
        "[3] never flagged",
        [num(e["train_oof"]["detection"]["never_flagged_fraction"]) for e, _ in pairs],
    )
    line(
        "[4] test flagged at last cycle, overall",
        [num(e["test"]["flagged_at_last_cycle"]["overall"]) for e, _ in pairs],
    )
    for i, (label, _lo, _hi) in enumerate(BUCKETS):
        line(
            f"[4] test flagged, true RUL {label} (n)",
            [
                f"{num(e['test']['flagged_at_last_cycle']['buckets'][i]['flagged_fraction'])} "
                f"(n={e['test']['flagged_at_last_cycle']['buckets'][i]['n']})"
                for e, _ in pairs
            ],
        )
    line(
        "[5] Spearman with RUL prediction (test last cycle)",
        [num(e["test"]["spearman_with_rul_prediction"]) for e, _ in pairs],
    )
    acc = sm["acceptance"]
    echo(
        f"FR-021 ({acc['criterion']}): trend {num(acc['trend_value'])} "
        f"({'met' if acc['trend_met'] else 'NOT met'}), false-alarm "
        f"{num(acc['false_alarm_value'], 4)} ({'met' if acc['false_alarm_met'] else 'NOT met'})"
        f" → {'MET' if acc['met'] else 'NOT MET'}"
    )
