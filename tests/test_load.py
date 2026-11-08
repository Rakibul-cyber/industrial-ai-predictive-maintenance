"""Tests for predmaint.data.load on synthetic raw-format files."""

from pathlib import Path

import pandas as pd
import pytest

from predmaint.data.load import (
    COLUMNS,
    DataValidationError,
    load_fd001,
    read_rul,
    read_trajectories,
)


def test_trailing_double_space_trap_is_real(synthetic_cmapss_dir: Path) -> None:
    """A naive single-space parse yields 28 columns; this is why the loader uses \\s+."""
    path = synthetic_cmapss_dir / "train_FD001.txt"
    assert path.read_text().splitlines()[0].endswith("  ")
    assert pd.read_csv(path, sep=" ", header=None).shape[1] == 28


def test_read_trajectories_returns_26_named_columns(synthetic_cmapss_dir: Path) -> None:
    frame = read_trajectories(synthetic_cmapss_dir / "train_FD001.txt")
    assert list(frame.columns) == COLUMNS
    assert frame.shape == (40 + 45 + 50, 26)
    assert frame["unit"].dtype == "int64"
    assert frame["cycle"].dtype == "int64"
    assert frame.isna().sum().sum() == 0


def test_read_rul(synthetic_cmapss_dir: Path) -> None:
    rul = read_rul(synthetic_cmapss_dir / "RUL_FD001.txt")
    assert list(rul.columns) == ["rul"]
    assert rul["rul"].tolist() == [10, 20, 30]


def test_wrong_column_count_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "train_FD001.txt"
    path.write_text(" ".join(["1"] * 25) + "  \n")
    with pytest.raises(DataValidationError, match="expected 26 columns, found 25"):
        read_trajectories(path)


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "train_FD001.txt"
    path.write_text("")
    with pytest.raises(DataValidationError, match="empty"):
        read_trajectories(path)


def test_multi_column_rul_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "RUL_FD001.txt"
    path.write_text("1 2\n")
    with pytest.raises(DataValidationError, match="expected 1 columns, found 2"):
        read_rul(path)


def test_load_fd001(synthetic_cmapss_dir: Path) -> None:
    data = load_fd001(synthetic_cmapss_dir)
    assert data.train["unit"].nunique() == 3
    assert data.test["unit"].nunique() == 3
    assert len(data.rul) == 3


def test_load_fd001_missing_files_points_to_download_script(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_fd001(tmp_path)
