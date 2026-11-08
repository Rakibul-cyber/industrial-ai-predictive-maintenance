"""Data-contract checks for C-MAPSS frames (docs/04 §6).

Run against the real dataset with ``python -m predmaint.data.validate``.
"""

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from predmaint.config import get_settings
from predmaint.data.load import (
    COLUMNS,
    RUL_COLUMN,
    DataValidationError,
    FD001Data,
    load_fd001,
)

__all__ = [
    "FD001_EXPECTED",
    "DataValidationError",
    "ExpectedFigures",
    "validate_fd001",
    "validate_rul",
    "validate_trajectories",
]


@dataclass(frozen=True)
class ExpectedFigures:
    """Dataset-level figures recorded in docs/04 §5."""

    train_rows: int
    test_rows: int
    train_units: int
    test_units: int
    rul_rows: int
    min_test_cycles: int


FD001_EXPECTED = ExpectedFigures(
    train_rows=20631,
    test_rows=13096,
    train_units=100,
    test_units=100,
    rul_rows=100,
    min_test_cycles=31,
)


def validate_trajectories(frame: pd.DataFrame, name: str) -> None:
    """Check a train/test frame against checks 1-4 of docs/04 §6.

    Raises:
        DataValidationError: naming ``name`` and the first violated rule.
    """
    if list(frame.columns) != COLUMNS:
        raise DataValidationError(f"{name}: columns {list(frame.columns)} != expected {COLUMNS}")
    if frame.empty:
        raise DataValidationError(f"{name}: no rows")
    null_columns = frame.columns[frame.isna().any()].tolist()
    if null_columns:
        raise DataValidationError(f"{name}: null values in columns {null_columns}")
    non_numeric = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise DataValidationError(f"{name}: non-numeric columns {non_numeric}")
    non_finite = frame.columns[~np.isfinite(frame.to_numpy(dtype=float)).all(axis=0)].tolist()
    if non_finite:
        raise DataValidationError(f"{name}: non-finite values in columns {non_finite}")
    for column in ("unit", "cycle"):
        if not pd.api.types.is_integer_dtype(frame[column]):
            raise DataValidationError(f"{name}: column {column!r} must be integer")
        if frame[column].min() < 1:
            raise DataValidationError(f"{name}: column {column!r} has values < 1")

    cycles = frame.groupby("unit", sort=False)["cycle"]
    first = cycles.transform("first")
    bad_start = frame.loc[first != 1, "unit"].unique()
    if len(bad_start):
        raise DataValidationError(f"{name}: unit(s) {bad_start[:5].tolist()} do not start at 1")
    step = cycles.diff().dropna()
    bad_step = frame.loc[step.index[step != 1], "unit"].unique()
    if len(bad_step):
        raise DataValidationError(
            f"{name}: unit(s) {bad_step[:5].tolist()} have cycles not increasing by exactly 1"
        )


def validate_rul(rul: pd.DataFrame, test: pd.DataFrame) -> None:
    """Check the RUL truth against the test frame (checks 1, 2, 5, 6 of docs/04 §6).

    Raises:
        DataValidationError: describing the first violated rule.
    """
    if list(rul.columns) != [RUL_COLUMN]:
        raise DataValidationError(f"RUL: columns {list(rul.columns)} != expected [{RUL_COLUMN!r}]")
    values = rul[RUL_COLUMN]
    if values.isna().any():
        raise DataValidationError("RUL: null values")
    if not pd.api.types.is_integer_dtype(values):
        raise DataValidationError("RUL: values must be integers")
    if (values < 0).any():
        raise DataValidationError("RUL: negative values")
    units = sorted(test["unit"].unique().tolist())
    if units != list(range(1, len(units) + 1)):
        raise DataValidationError("test: unit ids are not contiguous 1..n")
    if len(values) != len(units):
        raise DataValidationError(f"RUL: {len(values)} values for {len(units)} test units")


def validate_fd001(data: FD001Data, expected: ExpectedFigures = FD001_EXPECTED) -> None:
    """Run all contract checks plus the dataset-level figures in ``expected``.

    Raises:
        DataValidationError: describing the first violated rule.
    """
    validate_trajectories(data.train, "train")
    validate_trajectories(data.test, "test")
    validate_rul(data.rul, data.test)
    actual = {
        "train_rows": len(data.train),
        "test_rows": len(data.test),
        "train_units": data.train["unit"].nunique(),
        "test_units": data.test["unit"].nunique(),
        "rul_rows": len(data.rul),
    }
    for key, value in actual.items():
        if value != getattr(expected, key):
            raise DataValidationError(f"{key}: expected {getattr(expected, key)}, found {value}")
    min_test_cycles = int(data.test.groupby("unit")["cycle"].max().min())
    if min_test_cycles < expected.min_test_cycles:
        raise DataValidationError(
            f"shortest test unit has {min_test_cycles} cycles, "
            f"expected >= {expected.min_test_cycles}"
        )


def main() -> int:
    """Load FD001 from ``$DATA_DIR`` and validate it; print a summary. Returns an exit code."""
    cmapss_dir = get_settings().cmapss_dir
    try:
        data = load_fd001(cmapss_dir)
        validate_fd001(data)
    except (FileNotFoundError, DataValidationError) as exc:
        print(f"FD001 validation FAILED: {exc}", file=sys.stderr)
        return 1
    test_lengths = data.test.groupby("unit")["cycle"].max()
    print(f"FD001 validation passed ({cmapss_dir})")
    print(
        f"  train: {data.train.shape[0]} rows x {data.train.shape[1]} cols, "
        f"{data.train['unit'].nunique()} units"
    )
    print(
        f"  test:  {data.test.shape[0]} rows x {data.test.shape[1]} cols, "
        f"{data.test['unit'].nunique()} units, cycles/unit min {test_lengths.min()}"
    )
    print(
        f"  RUL:   {len(data.rul)} values, range {data.rul[RUL_COLUMN].min()}-"
        f"{data.rul[RUL_COLUMN].max()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
