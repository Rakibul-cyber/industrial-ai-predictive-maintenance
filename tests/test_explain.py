"""Permutation feature importance (ADR-040): helpers, the explain step, and its outputs."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predmaint.models.explain import (
    FEATURE_TYPES,
    aggregate,
    feature_type,
    population_importance,
    rank_agreement,
    ridge_sensor_weights,
    run_explanation,
    sensor_of,
    served_rmse_scorer,
)
from predmaint.models.predict import FEATURE_IMPORTANCE_FILE, clear_model_cache
from predmaint.models.registry import make_baseline, make_primary

QUIET = {"verify_checksums": False, "expected": None, "n_jobs": 1, "echo": lambda _m: None}
SENSORS = [f"s_{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]
FEATURES = ["cycle"] + [f"{s}{suf}" for s in SENSORS for suf in ("", "_roll_mean", "_roll_std")]


def _synthetic(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    y = 60 + 25 * x["s_3_roll_mean"].to_numpy() + rng.normal(0, 1, n)  # one feature drives y
    return x, y


# --- helpers ----------------------------------------------------------------------------------
def test_feature_names_map_to_sensors_and_types() -> None:
    assert len(FEATURES) == 43
    assert sensor_of("cycle") is None and sensor_of("s_7_roll_mean") == "s_7"
    assert sensor_of("s_21_roll_std") == "s_21" and sensor_of("s_14") == "s_14"
    types = [feature_type(f) for f in FEATURES]
    assert set(types) == set(FEATURE_TYPES)
    assert {t: types.count(t) for t in FEATURE_TYPES} == {
        "cycle": 1,
        "raw sensor": 14,
        "rolling mean": 14,
        "rolling std": 14,
    }  # every one of the 43 names in exactly one type


def test_aggregate_sums_within_repeat_then_mean_std() -> None:
    features = ["cycle", "s_2", "s_2_roll_mean", "s_3"]
    importances = np.array([[1.0, 3.0], [2.0, 2.0], [5.0, 1.0], [0.5, 0.5]])
    rows = {
        r["name"]: r for r in aggregate(importances, features, {f: sensor_of(f) for f in features})
    }
    assert set(rows) == {"s_2", "s_3"}  # cycle (None) skipped
    assert rows["s_2"]["n_features"] == 2
    assert rows["s_2"]["mean"] == pytest.approx(5.0)  # per repeat: 2 + 5 = 7 and 2 + 1 = 3
    assert rows["s_2"]["std"] == pytest.approx(2.0)
    assert rows["s_3"] == {"name": "s_3", "n_features": 1, "mean": 0.5, "std": 0.0}
    ordered = aggregate(importances, features, {f: feature_type(f) for f in features})
    # rolling mean [5, 1] → 3.0; raw sensor [2.5, 2.5] → 2.5; cycle [1, 3] → 2.0
    assert [r["name"] for r in ordered] == ["rolling mean", "raw sensor", "cycle"]


def test_served_scorer_clamps_like_the_api() -> None:
    class Constant:
        def predict(self, x: pd.DataFrame) -> np.ndarray:
            return np.array([-10.0, 200.0])

    score = served_rmse_scorer(125)(Constant(), pd.DataFrame({"a": [0, 0]}), np.array([0.0, 125.0]))
    assert score == 0.0  # -10 → 0 and 200 → 125: exact after clamping


def test_permutation_ranks_the_driving_feature_first() -> None:
    x, y = _synthetic()
    for pipeline in (
        make_baseline(FEATURES).fit(x, y),
        make_primary(FEATURES, 0, {"max_iter": 50, "max_leaf_nodes": 7}).fit(x, y),
    ):
        pop = population_importance(pipeline, x, y, None, n_repeats=5, seed=0, n_jobs=1)
        assert pop["per_feature"][0]["name"] == "s_3_roll_mean"
        assert pop["per_sensor"][0]["name"] == "s_3"
        assert pop["per_feature_type"][0]["name"] == "rolling mean"
        assert len(pop["per_feature"]) == 43 and len(pop["per_sensor"]) == 14
        assert pop["per_feature"][0]["mean"] > 10 * abs(pop["per_feature"][1]["mean"])


def test_ridge_weights_and_rank_agreement() -> None:
    x, y = _synthetic()
    weights = ridge_sensor_weights(make_baseline(FEATURES).fit(x, y))
    assert weights[0]["name"] == "s_3" and len(weights) == 14  # cycle excluded
    perm = [{"name": w["name"], "mean": w["abs_coef_sum"] * 2} for w in weights]
    assert rank_agreement(weights, perm) == pytest.approx(1.0)
    reversed_perm = [{"name": w["name"], "mean": -w["abs_coef_sum"]} for w in weights]
    assert rank_agreement(weights, reversed_perm) == pytest.approx(-1.0)
    constant = [{"name": w["name"], "mean": 0.0} for w in weights]
    assert rank_agreement(weights, constant) is None  # undefined, no warning


# --- explain step ---------------------------------------------------------------------------------
def test_fixture_report_keys_and_served_version(served_model_dir: Path) -> None:
    models = served_model_dir / "models"
    report = json.loads((served_model_dir / "reports" / "feature_importance.json").read_text())
    sidecar = (models / FEATURE_IMPORTANCE_FILE).read_text()
    assert sidecar == (served_model_dir / "reports" / "feature_importance.json").read_text()
    served = json.loads((models / "model_metadata.json").read_text())
    assert report["model_version"] == report["for_model_version"] == served["model_version"]
    assert {"n_repeats", "seed", "n_jobs", "method", "populations", "ridge_comparison",
            "caveats", "timings_seconds", "data"} <= set(report)  # fmt: skip
    assert (report["n_repeats"], report["seed"]) == (10, served["config"]["random_seed"])
    for name, pop in report["populations"].items():
        assert {"n_rows", "baseline_rmse", "target", "caveat", "per_feature", "per_sensor",
                "per_feature_type"} <= set(pop), name  # fmt: skip
        assert sorted(r["name"] for r in pop["per_feature"]) == sorted(served["feature_names"])
        assert {r["name"] for r in pop["per_feature_type"]} == set(FEATURE_TYPES)
    assert report["populations"]["test"]["n_rows"] == 4  # synthetic test engines
    assert report["ridge_comparison"]["source"].endswith("baseline.joblib")
    assert set(report["ridge_comparison"]["spearman_vs_permutation_per_sensor"]) == {
        "train",
        "test",
    }


def test_explain_retrains_ridge_when_absent_and_never_writes_models(
    served_model_dir: Path, tmp_path: Path
) -> None:
    models = tmp_path / "models"
    shutil.copytree(served_model_dir / "models", models)
    for name in ("baseline.joblib", "baseline_metadata.json"):
        (models / name).unlink()
    before = {p.name: p.read_bytes() for p in models.iterdir() if p.name != FEATURE_IMPORTANCE_FILE}
    clear_model_cache()
    report = run_explanation(served_model_dir / "CMAPSSData", models, tmp_path / "r", **QUIET)
    after = {p.name: p.read_bytes() for p in models.iterdir() if p.name != FEATURE_IMPORTANCE_FILE}
    assert before == after  # no model file written or changed; no baseline left behind
    assert report["ridge_comparison"]["source"] == "retrained (temporary directory)"
    original = json.loads((served_model_dir / "reports" / "feature_importance.json").read_text())
    assert report["populations"] == original["populations"]  # deterministic, n_jobs independent
    retrained, local = (
        r["ridge_comparison"]["per_sensor"] for r in (report, original)
    )  # the baseline is deterministic: the retrained Ridge equals the local candidate
    assert [r["name"] for r in retrained] == [r["name"] for r in local]
    assert [r["abs_coef_sum"] for r in retrained] == pytest.approx(
        [r["abs_coef_sum"] for r in local]
    )


def test_explain_refuses_without_served_model(served_model_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="--select"):
        run_explanation(served_model_dir / "CMAPSSData", tmp_path, tmp_path, **QUIET)


def test_explain_refuses_other_data(served_model_dir: Path, tmp_path: Path) -> None:
    models, data = tmp_path / "models", tmp_path / "data"
    shutil.copytree(served_model_dir / "models", models)
    shutil.copytree(served_model_dir / "CMAPSSData", data)
    (data / "RUL_FD001.txt").write_text("41 \n30 \n20 \n10 \n")
    clear_model_cache()
    with pytest.raises(ValueError, match="data differ"):
        run_explanation(data, models, tmp_path, **QUIET)


def test_cli_explain_reports_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from predmaint.models.train import main

    code = main(["--explain", "--model-dir", str(tmp_path), "--data-dir", str(tmp_path)])
    assert code == 1 and "model_metadata.json" in capsys.readouterr().err
