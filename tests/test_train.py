"""Tests for CV grouping and end-to-end baseline training on synthetic data."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predmaint.data.download import ChecksumMismatchError
from predmaint.data.load import load_fd001
from predmaint.features.build import build_features, last_rows
from predmaint.models.artifacts import load_artifacts
from predmaint.models.predict import predict_from_features
from predmaint.models.registry import PRIMARY_SEARCH_GRID, make_model, pipeline_feature_names
from predmaint.models.train import group_folds, run_training

TrajectoryFactory = Callable[..., pd.DataFrame]

REQUIRED_METADATA = {
    "schema_version",
    "model_name",
    "estimator",
    "model_version",
    "trained_at",
    "git_sha",
    "git_dirty",
    "python_version",
    "library_versions",
    "data",
    "config",
    "required_sensors",
    "dropped_columns",
    "feature_names",
    "training_ranges",
    "health_bands",
    "hyperparameters",
    "search",
    "metrics",
}


def test_group_folds_never_share_a_unit() -> None:
    groups = np.repeat(np.arange(1, 13), 7)
    seen_val: set[int] = set()
    for train_idx, val_idx in group_folds(groups, n_splits=5):
        train_units, val_units = set(groups[train_idx]), set(groups[val_idx])
        assert train_units.isdisjoint(val_units)
        assert train_units | val_units == set(range(1, 13))
        seen_val |= val_units
    assert seen_val == set(range(1, 13))


def test_unknown_model_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        make_model("lstm", ["cycle"], seed=0)
    with pytest.raises(ValueError, match="hyperparameters"):
        make_model("primary", ["cycle"], seed=0)


def test_primary_search_grid_is_bounded() -> None:
    assert len(PRIMARY_SEARCH_GRID) == 12
    assert len({tuple(sorted(p.items())) for p in PRIMARY_SEARCH_GRID}) == 12


@pytest.fixture
def dataset(synthetic_training_dir: Path) -> Path:
    return synthetic_training_dir


def _train(dataset: Path, tmp_path: Path, model_name: str = "baseline", **kwargs: object) -> dict:
    return run_training(
        model_name,
        dataset,
        tmp_path / "models",
        tmp_path / "reports",
        verify_checksums=False,
        expected=None,
        echo=lambda _msg: None,
        **kwargs,
    )


def test_end_to_end_train_save_reload_predict(dataset: Path, tmp_path: Path) -> None:
    metadata = _train(dataset, tmp_path)

    assert set(metadata) >= REQUIRED_METADATA
    assert metadata["config"] == {
        "rul_cap": 125,
        "window_size": 30,
        "random_seed": 42,
        "cv_folds": 5,
    }
    assert "s_1" not in metadata["required_sensors"]  # constant in the synthetic data
    assert {"setting_1", "setting_2", "setting_3", "s_1"} <= set(metadata["dropped_columns"])
    assert len(metadata["feature_names"]) == 1 + 3 * len(metadata["required_sensors"])
    assert len(metadata["metrics"]["cv"]["folds"]) == 5
    for key in ("mae", "rmse", "nasa_score"):
        assert np.isfinite(metadata["metrics"]["test"][key])

    pipeline, reloaded = load_artifacts(tmp_path / "models", "baseline")
    assert reloaded == metadata
    assert pipeline_feature_names(pipeline) == metadata["feature_names"]

    data = load_fd001(dataset)
    test_rows = last_rows(build_features(data.test, metadata["required_sensors"], 30))
    pred = predict_from_features(pipeline, test_rows, 125)
    assert pred.shape == (data.test["unit"].nunique(),)
    assert ((pred >= 0) & (pred <= 125)).all()

    report = json.loads((tmp_path / "reports" / "metrics_baseline.json").read_text())
    assert report["metrics"] == metadata["metrics"]
    assert report["model_version"] == metadata["model_version"]


def test_training_is_reproducible(dataset: Path, tmp_path: Path) -> None:
    first = _train(dataset, tmp_path / "a")["metrics"]
    second = _train(dataset, tmp_path / "b")["metrics"]
    assert first == second


def test_saved_pipeline_rejects_missing_feature_columns(dataset: Path, tmp_path: Path) -> None:
    metadata = _train(dataset, tmp_path)
    pipeline, _ = load_artifacts(tmp_path / "models", "baseline")
    data = load_fd001(dataset)
    rows = last_rows(build_features(data.test, metadata["required_sensors"], 30))
    with pytest.raises(ValueError):
        pipeline.predict(rows.drop(columns=[metadata["feature_names"][1]]))


def test_checksum_gate_blocks_unverified_data(dataset: Path, tmp_path: Path) -> None:
    with pytest.raises(ChecksumMismatchError, match="docs/04"):
        run_training("baseline", dataset, tmp_path / "m", tmp_path / "r", expected=None)


def test_load_artifacts_missing_and_bad_schema(dataset: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train.py"):
        load_artifacts(tmp_path / "nothing", "baseline")
    _train(dataset, tmp_path)
    meta_path = tmp_path / "models" / "baseline_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps({**meta, "schema_version": 99}))
    with pytest.raises(ValueError, match="schema_version"):
        load_artifacts(tmp_path / "models", "baseline")


TINY_GRID = [
    {"learning_rate": 0.1, "max_iter": 20, "max_leaf_nodes": 7, "l2_regularization": 0.0},
    {"learning_rate": 0.1, "max_iter": 40, "max_leaf_nodes": 15, "l2_regularization": 1.0},
]


def test_primary_end_to_end_with_search(dataset: Path, tmp_path: Path) -> None:
    metadata = _train(dataset, tmp_path, search_grid=TINY_GRID, model_name="primary")
    search = metadata["search"]
    assert search["n_configurations"] == 2
    assert search["grid"] == TINY_GRID
    best = min(search["results"], key=lambda r: (r["rmse_mean"], r["mae_mean"]))
    assert search["selected_params"] == best["params"] == metadata["hyperparameters"]
    assert metadata["metrics"]["cv"]["rmse_mean"] == best["rmse_mean"]
    assert metadata["estimator"] == "hgb"

    pipeline, reloaded = load_artifacts(tmp_path / "models", "primary")
    assert reloaded == metadata
    assert pipeline.named_steps["model"].get_params()["max_iter"] == best["params"]["max_iter"]
    assert pipeline.named_steps["model"].get_params()["early_stopping"] is False
    data = load_fd001(dataset)
    rows = last_rows(build_features(data.test, metadata["required_sensors"], 30))
    pred = predict_from_features(pipeline, rows, 125)
    assert pred.shape == (4,)
    assert ((pred >= 0) & (pred <= 125)).all()
    assert (tmp_path / "reports" / "metrics_primary.json").is_file()


def test_primary_training_is_reproducible(dataset: Path, tmp_path: Path) -> None:
    first = _train(dataset, tmp_path / "a", search_grid=TINY_GRID, model_name="primary")
    second = _train(dataset, tmp_path / "b", search_grid=TINY_GRID, model_name="primary")
    assert first["metrics"] == second["metrics"]
    assert first["search"] == second["search"]
