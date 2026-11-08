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
