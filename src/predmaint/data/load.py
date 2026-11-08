"""Parse the raw C-MAPSS text files into named DataFrames (docs/04 §4).

The files are whitespace-delimited with no header, and every train/test line ends with two
trailing spaces. ``sep=" "`` would therefore yield 28 columns; ``sep=r"\\s+"`` yields 26.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from predmaint.config import DATASET_SUBSET

ID_COLUMNS: list[str] = ["unit", "cycle"]
SETTING_COLUMNS: list[str] = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS: list[str] = [f"s_{i}" for i in range(1, 22)]
COLUMNS: list[str] = ID_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS
RUL_COLUMN = "rul"


class DataValidationError(ValueError):
    """Raised when a data file or frame violates the data contract (docs/04 §6)."""


@dataclass(frozen=True)
class FD001Data:
    """The three FD001 tables: run-to-failure train, truncated test, and test RUL truth."""

    train: pd.DataFrame
    test: pd.DataFrame
    rul: pd.DataFrame


def _read_whitespace(path: Path, expected_columns: int) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, sep=r"\s+", header=None)
    except EmptyDataError as exc:
        raise DataValidationError(f"{path.name}: file is empty") from exc
    if frame.shape[1] != expected_columns:
        raise DataValidationError(
            f"{path.name}: expected {expected_columns} columns, found {frame.shape[1]}"
        )
    return frame


def read_trajectories(path: Path) -> pd.DataFrame:
    """Read a ``train_FD00x.txt`` / ``test_FD00x.txt`` file with the 26 named columns.

    Raises:
        FileNotFoundError: if the file does not exist.
        DataValidationError: if the file is empty or does not have exactly 26 columns.
    """
    frame = _read_whitespace(path, len(COLUMNS))
    frame.columns = COLUMNS
    for column in ID_COLUMNS:
        if pd.api.types.is_integer_dtype(frame[column]):
            frame[column] = frame[column].astype("int64")
    return frame


def read_rul(path: Path) -> pd.DataFrame:
    """Read a ``RUL_FD00x.txt`` file into a one-column frame named ``rul``.

    Raises:
        FileNotFoundError: if the file does not exist.
        DataValidationError: if the file is empty or has more than one column.
    """
    frame = _read_whitespace(path, 1)
    frame.columns = [RUL_COLUMN]
    return frame


def load_fd001(cmapss_dir: Path) -> FD001Data:
    """Load train/test/RUL for FD001 from ``cmapss_dir`` (no validation is applied here).

    Raises:
        FileNotFoundError: if a file is missing (run ``python scripts/download_data.py``).
        DataValidationError: if a file has the wrong number of columns.
    """
    paths = {kind: cmapss_dir / f"{kind}_{DATASET_SUBSET}.txt" for kind in ("train", "test", "RUL")}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset file(s): {', '.join(missing)}. "
            "Run `python scripts/download_data.py` first."
        )
    return FD001Data(
        train=read_trajectories(paths["train"]),
        test=read_trajectories(paths["test"]),
        rul=read_rul(paths["RUL"]),
    )
