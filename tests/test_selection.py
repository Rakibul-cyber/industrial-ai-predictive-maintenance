"""Tests for ADR-024 selection and promotion, using fabricated candidate metadata."""

import json
from pathlib import Path
from typing import Any

import joblib
import pytest

from predmaint.models.artifacts import SCHEMA_VERSION
from predmaint.models.selection import (
    IncompatibleCandidatesError,
    choose_model,
    promote,
    served_model_version,
)


def _meta(
    name: str, cv_rmse: float, cv_mae: float, test_rmse: float, test_mae: float = 10.0
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_name": name,
        "estimator": {"baseline": "ridge", "primary": "hgb"}[name],
        "model_version": f"{name}-candidate",
        "trained_at": "2026-11-08T14:50:35Z",
        "git_sha": "f4c86dfabcdef0123456789",
        "hyperparameters": {"alpha": 1.0} if name == "baseline" else {"max_iter": 100},
        "data": {"subset": "FD001", "train_sha256": "a"},
        "config": {"rul_cap": 125, "window_size": 30, "random_seed": 42, "cv_folds": 5},
        "feature_names": ["cycle", "s_2"],
        "required_sensors": ["s_2"],
        "metrics": {
            "cv": {"rmse_mean": cv_rmse, "rmse_std": 1.0, "mae_mean": cv_mae, "mae_std": 0.5},
            "test": {"rmse": test_rmse, "mae": test_mae, "nasa_score": 500.0},
            "test_clipped_truth": {"rmse": test_rmse - 1, "mae": test_mae - 1},
        },
    }


def test_primary_wins_on_cv() -> None:
    decision = choose_model(
        {
            "baseline": _meta("baseline", 19.3, 15.5, 19.2),
            "primary": _meta("primary", 17.0, 13.0, 18.0),
        }
    )
    assert decision.selected == "primary"
    assert decision.cv_ranking == ["primary", "baseline"]
    assert decision.rankings_agree
    assert "WARNING" not in decision.reason


def test_baseline_wins_on_cv() -> None:
    decision = choose_model(
        {
            "baseline": _meta("baseline", 18.0, 14.0, 17.5),
            "primary": _meta("primary", 18.5, 13.0, 19.0),
        }
    )
    assert decision.selected == "baseline"
    assert decision.rankings_agree


def test_cv_and_test_disagree_follows_cv_and_flags_it() -> None:
    decision = choose_model(
        {
            "baseline": _meta("baseline", 19.3, 15.5, 17.0),
            "primary": _meta("primary", 17.0, 13.0, 18.0),
        }
    )
    assert decision.selected == "primary"
    assert decision.test_ranking == ["baseline", "primary"]
    assert decision.rankings_agree is False
    assert "disagrees with the CV ranking" in decision.reason
    assert "CV-based choice stands" in decision.reason


def test_tie_breaks_cv_mae_then_simpler_model() -> None:
    by_mae = choose_model(
        {
            "baseline": _meta("baseline", 18.0, 14.0, 18.0),
            "primary": _meta("primary", 18.0, 13.5, 18.0),
        }
    )
    assert by_mae.selected == "primary"
    assert "CV RMSE is tied" in by_mae.reason
    full_tie = choose_model(
        {
            "baseline": _meta("baseline", 18.0, 14.0, 18.0),
            "primary": _meta("primary", 18.0, 14.0, 18.0),
        }
    )
    assert full_tie.selected == "baseline"


def test_served_model_version_format() -> None:
    assert served_model_version(_meta("primary", 1, 1, 1)) == "primary-20261108-f4c86df"
    assert served_model_version({**_meta("baseline", 1, 1, 1), "git_sha": "unknown"}) == (
        "baseline-20261108-unknown"
    )


def _write_candidates(model_dir: Path, metas: dict[str, dict[str, Any]]) -> None:
    model_dir.mkdir(parents=True)
    for name, meta in metas.items():
        joblib.dump({"stand-in model": name}, model_dir / f"{name}.joblib")
        (model_dir / f"{name}_metadata.json").write_text(json.dumps(meta))


def test_promotion_writes_served_artifact_and_reports(tmp_path: Path) -> None:
    metas = {
        "baseline": _meta("baseline", 19.3, 15.5, 17.0),
        "primary": _meta("primary", 17.0, 13.0, 18.0),
    }
    _write_candidates(tmp_path / "models", metas)
    promoted = promote(tmp_path / "models", tmp_path / "reports", echo=lambda _m: None)

    served = tmp_path / "models" / "model.joblib"
    assert served.read_bytes() == (tmp_path / "models" / "primary.joblib").read_bytes()
    on_disk = json.loads((tmp_path / "models" / "model_metadata.json").read_text())
    assert on_disk == promoted
    assert promoted["model_name"] == "primary"
    assert promoted["selected_by"] == "cv_rmse"
    assert promoted["model_version"] == "primary-20261108-f4c86df"
    assert promoted["candidate_model_version"] == "primary-candidate"
    assert promoted["health_bands"] == {"critical_below": 20, "warning_below": 60}
    comparison = promoted["comparison"]
    assert comparison["selected"] == "primary"
    assert comparison["rankings_agree"] is False
    assert set(comparison["candidates"]) == {"baseline", "primary"}

    report = json.loads((tmp_path / "reports" / "model_comparison.json").read_text())
    assert report == comparison
    markdown = (tmp_path / "reports" / "model_comparison.md").read_text()
    assert "| primary (selected) |" in markdown
    assert "| baseline |" in markdown
    assert comparison["reason"] in markdown


def test_promotion_rejects_incompatible_candidates(tmp_path: Path) -> None:
    primary = _meta("primary", 17.0, 13.0, 18.0)
    primary["config"] = {**primary["config"], "window_size": 20}
    _write_candidates(
        tmp_path / "models", {"baseline": _meta("baseline", 19, 15, 17), "primary": primary}
    )
    with pytest.raises(IncompatibleCandidatesError, match="config"):
        promote(tmp_path / "models", tmp_path / "reports", echo=lambda _m: None)
    assert not (tmp_path / "models" / "model.joblib").exists()


def test_promotion_requires_both_candidates(tmp_path: Path) -> None:
    _write_candidates(tmp_path / "models", {"baseline": _meta("baseline", 19, 15, 17)})
    with pytest.raises(FileNotFoundError, match="primary"):
        promote(tmp_path / "models", tmp_path / "reports", echo=lambda _m: None)
