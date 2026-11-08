"""Integration tests on the real FD001 files; skipped when the dataset is absent (CI)."""

from pathlib import Path

import pytest

from predmaint.config import get_settings
from predmaint.data.download import CMAPSS_SPEC, sha256_file
from predmaint.data.load import COLUMNS, load_fd001
from predmaint.data.validate import FD001_EXPECTED, validate_fd001
from predmaint.features.build import select_sensors
from predmaint.models.metrics import rmse
from predmaint.models.predict import clear_model_cache, load_model, predict_rul

pytestmark = pytest.mark.data


def test_fd001_checksums_match_docs(real_cmapss_dir: Path) -> None:
    for name, expected in CMAPSS_SPEC.verified_files.items():
        assert sha256_file(real_cmapss_dir / name) == expected, name


def test_fd001_passes_full_validation(real_cmapss_dir: Path) -> None:
    validate_fd001(load_fd001(real_cmapss_dir))


def test_fd001_recorded_figures(real_cmapss_dir: Path) -> None:
    data = load_fd001(real_cmapss_dir)
    assert data.train.shape == (FD001_EXPECTED.train_rows, len(COLUMNS))
    assert data.test.shape == (FD001_EXPECTED.test_rows, len(COLUMNS))
    assert data.rul.shape == (FD001_EXPECTED.rul_rows, 1)
    assert int(data.test.groupby("unit")["cycle"].max().min()) == 31
    assert (int(data.rul["rul"].min()), int(data.rul["rul"].max())) == (7, 145)


def test_fd001_sensor_selection_matches_docs(real_cmapss_dir: Path) -> None:
    """Locks the verified kept/dropped list recorded in docs/04 §5 (ADR-005)."""
    kept = select_sensors(load_fd001(real_cmapss_dir).train)
    assert kept == [f"s_{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]


def test_promoted_model_test_rmse_matches_metadata(real_cmapss_dir: Path) -> None:
    """Recompute the served model's official-test RMSE through the inference contract."""
    model_dir = get_settings().model_dir
    if not (model_dir / "model.joblib").is_file():
        pytest.skip(f"no promoted model in {model_dir} (run scripts/train.py --select)")
    clear_model_cache()
    model = load_model(model_dir)
    data = load_fd001(real_cmapss_dir)
    predictions = [
        predict_rul(data.test[data.test["unit"] == unit], model).predicted_rul
        for unit in range(1, len(data.rul) + 1)
    ]
    recomputed = rmse(data.rul["rul"].to_numpy(dtype=float), predictions)
    assert recomputed == pytest.approx(model.metadata["metrics"]["test"]["rmse"], abs=1e-9)


@pytest.mark.parametrize(
    "example", ["predict_request.json", "predict_request_min.json", "predict_request_late.json"]
)
def test_api_reproduces_committed_example(example: str) -> None:
    """POST a committed demo payload through the API; it must match the recorded sidecar."""
    import json

    from fastapi.testclient import TestClient

    from predmaint.api.main import create_app

    model_path = get_settings().model_path
    if not model_path.is_file():
        pytest.skip(f"no served model at {model_path} (run scripts/train.py --select)")
    examples = Path("examples")
    payload = json.loads((examples / example).read_text())
    expected = json.loads((examples / "expected_predictions.json").read_text())[example]
    clear_model_cache()
    with TestClient(create_app(model_path=model_path)) as client:
        body = client.post("/predict", json=payload).json()
    keys = ("unit_id", "cycles_received", "last_cycle", "predicted_rul", "health_status")
    for key in (*keys, "rul_lower", "rul_upper", "interval_level"):
        assert body[key] == expected[key], key
    with TestClient(create_app(model_path=model_path)) as client:
        scored = client.post("/anomaly", json=payload).json()
    assert scored["unit_id"] == expected["unit_id"]
    assert scored["last_cycle"] == expected["last_cycle"]
    for key, value in expected["anomaly"].items():
        assert scored[key] == value, key


@pytest.mark.parametrize("unit", [1, 39, 69])  # 192, 128 (shortest), 362 (longest) cycles
def test_dashboard_true_rul_matches_training_labels(real_cmapss_dir: Path, unit: int) -> None:
    """The dashboard's capped true-RUL series equals the training labels (add_rul) at each
    sampled cycle; it ends at 0 and starts at min(RUL_CAP, max_cycle - WINDOW_SIZE).
    add_rul is imported here only — the dashboard itself never imports training code."""
    from predmaint.config import RUL_CAP, WINDOW_SIZE
    from predmaint.dashboard.helpers import engine_history, trajectory_cycles, true_rul_series
    from predmaint.data.labels import add_rul

    data = load_fd001(real_cmapss_dir)
    history = engine_history(data, "train", unit)
    cycles = trajectory_cycles(history.last_cycle, WINDOW_SIZE)
    series = true_rul_series(history, cycles, RUL_CAP)
    labels = add_rul(data.train[data.train["unit"] == unit], RUL_CAP).set_index("cycle")["rul"]
    assert series["true_rul_capped"].tolist() == [int(labels[c]) for c in cycles]
    assert series["true_rul_capped"].iloc[-1] == 0
    assert series["true_rul_capped"].iloc[0] == min(RUL_CAP, history.last_cycle - WINDOW_SIZE)


def test_committed_intervals_reproduce_recorded_test_coverage(real_cmapss_dir: Path) -> None:
    """Recompute the official-test interval evaluation from the committed artifacts."""
    import numpy as np

    from predmaint.features.build import build_features, last_rows
    from predmaint.models.intervals import summarize_intervals
    from predmaint.models.predict import (
        interval_path_for,
        load_interval_model,
        order_bounds,
        predict_from_features,
    )

    model_path = get_settings().model_path
    if not interval_path_for(model_path).is_file():
        pytest.skip("no interval artifact (run scripts/train.py --model intervals)")
    clear_model_cache()
    model, intervals = load_model(model_path), load_interval_model(interval_path_for(model_path))
    data = load_fd001(real_cmapss_dir)
    last = last_rows(build_features(data.test, model.required_sensors, model.window_size))
    point = predict_from_features(model.pipeline, last, model.rul_cap)
    lo, hi, n = order_bounds(
        point, intervals.lower.predict(last), intervals.upper.predict(last), 125
    )
    summary = summarize_intervals(data.rul["rul"].to_numpy(dtype=float), lo, hi, 125, n)
    assert summary == intervals.metadata["evaluation"]["test"]
    assert np.all(lo <= point) and np.all(point <= hi)


def test_committed_anomaly_reproduces_recorded_evaluation(
    real_cmapss_dir: Path, tmp_path: Path
) -> None:
    """Retraining the detector (seconds) reproduces the committed threshold and evaluation."""
    import json
    import shutil

    from predmaint.models.anomaly import run_anomaly_training
    from predmaint.models.predict import anomaly_path_for, load_anomaly_model

    model_path = get_settings().model_path
    if not anomaly_path_for(model_path).is_file():
        pytest.skip("no anomaly artifact (run scripts/train.py --model anomaly)")
    clear_model_cache()
    committed = load_anomaly_model(anomaly_path_for(model_path)).metadata
    models = tmp_path / "models"
    models.mkdir()
    for name in ("model.joblib", "model_metadata.json"):
        shutil.copyfile(model_path.parent / name, models / name)
    fresh = run_anomaly_training(
        real_cmapss_dir, models, tmp_path / "reports", echo=lambda _m: None
    )
    assert fresh["threshold"] == committed["threshold"]
    assert fresh["for_model_version"] == committed["for_model_version"]
    assert json.loads(json.dumps(fresh["evaluation"])) == committed["evaluation"]
