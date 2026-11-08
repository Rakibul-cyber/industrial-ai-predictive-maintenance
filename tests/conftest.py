"""Shared fixtures: synthetic C-MAPSS-shaped data (no NASA data is copied; docs/09 §2)."""

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predmaint.config import get_settings
from predmaint.data.load import COLUMNS, SENSOR_COLUMNS, SETTING_COLUMNS

_GATED_MARKERS = {"network": "RUN_NETWORK_TESTS", "docker": "RUN_DOCKER_TESTS"}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``network``/``docker`` tests unless their opt-in env var is ``1``."""
    for item in items:
        for marker, env_var in _GATED_MARKERS.items():
            if marker in item.keywords and os.environ.get(env_var) != "1":
                item.add_marker(pytest.mark.skip(reason=f"set {env_var}=1 to run"))


def make_trajectories(cycles_per_unit: list[int], seed: int = 0) -> pd.DataFrame:
    """Build a 26-column frame: units 1..n, cycles 1..len, a constant and trending sensors."""
    rng = np.random.default_rng(seed)
    frames = []
    for unit, n_cycles in enumerate(cycles_per_unit, start=1):
        cycle = np.arange(1, n_cycles + 1)
        data: dict[str, np.ndarray] = {
            "unit": np.full(n_cycles, unit, dtype=np.int64),
            "cycle": cycle.astype(np.int64),
        }
        for column in SETTING_COLUMNS:
            data[column] = rng.normal(0.0, 0.001, n_cycles).round(4)
        for i, column in enumerate(SENSOR_COLUMNS):
            if i == 0:  # constant sensor, like a fixed-condition reading
                data[column] = np.full(n_cycles, 518.67)
            else:
                trend = 0.05 * i * cycle / n_cycles
                data[column] = (100.0 + i + trend + rng.normal(0.0, 0.01, n_cycles)).round(4)
        frames.append(pd.DataFrame(data))
    return pd.DataFrame(pd.concat(frames, ignore_index=True)[COLUMNS])


def write_raw_trajectories(frame: pd.DataFrame, path: Path) -> None:
    """Write in the raw C-MAPSS format: single spaces and two trailing spaces per line."""
    lines = []
    for row in frame.itertuples(index=False):
        values = [str(int(row[0])), str(int(row[1]))] + [f"{v:.4f}" for v in row[2:]]
        lines.append(" ".join(values) + "  ")
    path.write_text("\n".join(lines) + "\n")


def write_raw_rul(values: list[int], path: Path) -> None:
    """Write a RUL file: one integer per line followed by a single trailing space."""
    path.write_text("".join(f"{v} \n" for v in values))


@pytest.fixture
def trajectories() -> Callable[..., pd.DataFrame]:
    """Factory fixture exposing :func:`make_trajectories` to tests."""
    return make_trajectories


@pytest.fixture
def synthetic_cmapss_dir(tmp_path: Path) -> Path:
    """A directory with small synthetic ``train/test/RUL_FD001.txt`` files."""
    directory = tmp_path / "CMAPSSData"
    directory.mkdir()
    write_raw_trajectories(make_trajectories([40, 45, 50], seed=1), directory / "train_FD001.txt")
    write_raw_trajectories(make_trajectories([35, 32, 38], seed=2), directory / "test_FD001.txt")
    write_raw_rul([10, 20, 30], directory / "RUL_FD001.txt")
    return directory


@pytest.fixture
def real_cmapss_dir() -> Path:
    """The real FD001 directory under ``$DATA_DIR``; skips the test when absent."""
    directory = get_settings().cmapss_dir
    if not (directory / "train_FD001.txt").is_file():
        pytest.skip(f"real FD001 data not found in {directory}")
    return directory
