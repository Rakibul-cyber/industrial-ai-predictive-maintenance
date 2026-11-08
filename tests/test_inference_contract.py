"""Tests for the inference contract (docs/05 §10) against a real promoted synthetic model."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predmaint.data.load import load_fd001
from predmaint.features.build import build_features, last_rows
from predmaint.models.predict import (
    InsufficientHistoryError,
    InvalidValuesError,
    LoadedModel,
    MissingColumnsError,
    ModelNotAvailableError,
    NonConsecutiveCyclesError,
    clear_model_cache,
    health_status,
    load_model,
    predict_from_features,
    predict_rul,
)


@pytest.fixture
def model(served_model_dir: Path) -> LoadedModel:
    clear_model_cache()
    return load_model(served_model_dir / "models")


@pytest.fixture
def test_data(served_model_dir: Path) -> pd.DataFrame:
    return load_fd001(served_model_dir / "CMAPSSData").test


def _unit(test_data: pd.DataFrame, unit: int) -> pd.DataFrame:
    return test_data[test_data["unit"] == unit].reset_index(drop=True)


def test_load_model_is_cached_and_reloadable(served_model_dir: Path) -> None:
    clear_model_cache()
    first = load_model(served_model_dir / "models")
    assert load_model(served_model_dir / "models") is first
    clear_model_cache()
    assert load_model(served_model_dir / "models") is not first


def test_load_model_uses_model_dir_env(
    served_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_model_cache()
    monkeypatch.setenv("MODEL_DIR", str(served_model_dir / "models"))
    assert load_model().model_version.startswith(("baseline-", "primary-"))


def test_missing_model_raises(tmp_path: Path) -> None:
    clear_model_cache()
    with pytest.raises(ModelNotAvailableError, match="model.joblib"):
        load_model(tmp_path)


def test_corrupt_model_raises(served_model_dir: Path, tmp_path: Path) -> None:
    clear_model_cache()
    shutil.copytree(served_model_dir / "models", tmp_path / "m")
    (tmp_path / "m" / "model.joblib").write_bytes(b"not a pickle")
    with pytest.raises(ModelNotAvailableError, match="could not load"):
        load_model(tmp_path / "m")


def test_metadata_missing_key_raises(served_model_dir: Path, tmp_path: Path) -> None:
    clear_model_cache()
    shutil.copytree(served_model_dir / "models", tmp_path / "m")
    meta_path = tmp_path / "m" / "model_metadata.json"
    meta = json.loads(meta_path.read_text())
    del meta["required_sensors"]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ModelNotAvailableError, match="required_sensors"):
        load_model(tmp_path / "m")


def test_output_schema_and_bounds(model: LoadedModel, test_data: pd.DataFrame) -> None:
    history = _unit(test_data, 2)
    result = predict_rul(history, model)
    assert isinstance(result.predicted_rul, float)
    assert 0.0 <= result.predicted_rul <= model.rul_cap
    assert result.health_status in {"critical", "warning", "healthy"}
    assert result.health_status == health_status(result.predicted_rul)
    assert result.cycles_received == len(history) == 40
    assert result.cycles_used == model.window_size == 30
    assert result.last_cycle == 40
    assert result.model_version == model.model_version
    assert isinstance(result.warnings, list)


def test_records_input_equals_dataframe_input(model: LoadedModel, test_data: pd.DataFrame) -> None:
    history = _unit(test_data, 1)
    assert predict_rul(history.to_dict("records"), model) == predict_rul(history, model)


def test_deterministic(model: LoadedModel, test_data: pd.DataFrame) -> None:
    history = _unit(test_data, 3)
    assert predict_rul(history, model) == predict_rul(history.copy(), model)


def test_only_last_window_matters(model: LoadedModel, test_data: pd.DataFrame) -> None:
    history = _unit(test_data, 4)  # 50 cycles
    recent = history.tail(30)
    tampered = history.copy()
    tampered.loc[:19, model.required_sensors] = 9_999.0  # older than the window
    full, short, changed = (predict_rul(h, model) for h in (history, recent, tampered))
    assert full.predicted_rul == short.predicted_rul == changed.predicted_rul
    assert full.cycles_received == 50 and short.cycles_received == 30


def test_matches_offline_evaluation_path(model: LoadedModel, test_data: pd.DataFrame) -> None:
    """The API seam and the offline test evaluation produce identical numbers."""
    rows = last_rows(build_features(test_data, model.required_sensors, model.window_size))
    offline = predict_from_features(model.pipeline, rows, model.rul_cap)
    online = [predict_rul(_unit(test_data, u), model).predicted_rul for u in rows["unit"]]
    np.testing.assert_array_equal(offline, np.array(online))


def test_too_few_cycles(model: LoadedModel, test_data: pd.DataFrame) -> None:
    with pytest.raises(InsufficientHistoryError, match="29 cycles"):
        predict_rul(_unit(test_data, 1).tail(29), model)
    with pytest.raises(InsufficientHistoryError):
        predict_rul([], model)


def test_missing_and_null_columns(model: LoadedModel, test_data: pd.DataFrame) -> None:
    sensor = model.required_sensors[0]
    with pytest.raises(MissingColumnsError) as missing:
        predict_rul(_unit(test_data, 1).drop(columns=[sensor, "cycle"]), model)
    assert set(missing.value.columns) == {"cycle", sensor}
    nulls = _unit(test_data, 1)
    nulls.loc[5, sensor] = np.nan
    with pytest.raises(MissingColumnsError, match=sensor):
        predict_rul(nulls, model)


@pytest.mark.parametrize("case", ["swapped", "gap", "duplicate", "descending"])
def test_non_consecutive_cycles(model: LoadedModel, test_data: pd.DataFrame, case: str) -> None:
    history = _unit(test_data, 1)
    if case == "swapped":
        history = history.iloc[[1, 0, *range(2, len(history))]]
    elif case == "gap":
        history.loc[10:, "cycle"] += 1
    elif case == "duplicate":
        history.loc[10, "cycle"] = 10
    else:
        history = history.iloc[::-1]
    with pytest.raises(NonConsecutiveCyclesError):
        predict_rul(history, model)


def test_invalid_values(model: LoadedModel, test_data: pd.DataFrame) -> None:
    history = _unit(test_data, 1)
    history.loc[3, model.required_sensors[0]] = np.inf
    with pytest.raises(InvalidValuesError, match="non-finite"):
        predict_rul(history, model)
    fractional = _unit(test_data, 1).astype({"cycle": float})
    fractional["cycle"] = fractional["cycle"] + 0.5
    with pytest.raises(InvalidValuesError, match="integers"):
        predict_rul(fractional, model)


def test_out_of_training_range_warns(model: LoadedModel, test_data: pd.DataFrame) -> None:
    history = _unit(test_data, 1)
    sensor = model.required_sensors[0]
    history.loc[len(history) - 1, sensor] = model.metadata["training_ranges"][sensor]["max"] + 50
    result = predict_rul(history, model)
    assert any(w.startswith(f"OUT_OF_TRAINING_RANGE: {sensor}") for w in result.warnings)


@pytest.mark.parametrize(
    ("rul", "status"),
    [
        (0.0, "critical"),
        (19.99, "critical"),
        (20.0, "warning"),
        (59.99, "warning"),
        (60.0, "healthy"),
        (125.0, "healthy"),
    ],
)
def test_health_status_bands(rul: float, status: str) -> None:
    assert health_status(rul) == status


def test_clamp_to_zero_and_cap() -> None:
    class Stub:
        def predict(self, _features: object) -> np.ndarray:
            return np.array([-5.0, 50.0, 200.0])

    out = predict_from_features(Stub(), pd.DataFrame(), 125)  # type: ignore[arg-type]
    np.testing.assert_array_equal(out, [0.0, 50.0, 125.0])
    np.testing.assert_array_equal(
        predict_from_features(Stub(), pd.DataFrame(), None), [0.0, 50.0, 200.0]
    )  # type: ignore[arg-type]
