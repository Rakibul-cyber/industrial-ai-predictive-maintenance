"""Tests for predmaint.data.validate: good data passes, each broken case fails."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from predmaint.data.load import FD001Data
from predmaint.data.validate import (
    DataValidationError,
    ExpectedFigures,
    validate_fd001,
    validate_rul,
    validate_trajectories,
)

TrajectoryFactory = Callable[..., pd.DataFrame]


@pytest.fixture
def frame(trajectories: TrajectoryFactory) -> pd.DataFrame:
    return trajectories([5, 6, 4])


@pytest.fixture
def rul() -> pd.DataFrame:
    return pd.DataFrame({"rul": [3, 2, 1]})


def test_good_data_passes(frame: pd.DataFrame, rul: pd.DataFrame) -> None:
    validate_trajectories(frame, "train")
    validate_rul(rul, frame)


def _set(row: int, column: str, value: object) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def mutate(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if isinstance(value, float) and column in ("unit", "cycle"):
            df[column] = df[column].astype(float)
        df.loc[row, column] = value
        return df

    return mutate


BROKEN_TRAJECTORIES = {
    "wrong column names": (lambda df: df.rename(columns={"s_21": "s_22"}), "columns"),
    "missing column": (lambda df: df.drop(columns=["s_21"]), "columns"),
    "null value": (_set(3, "s_5", np.nan), "null values"),
    "infinite value": (_set(3, "s_5", np.inf), "non-finite"),
    "non-numeric column": (lambda df: df.assign(s_2="x"), "non-numeric"),
    "float cycle": (_set(0, "cycle", 1.5), "'cycle' must be integer"),
    "unit below 1": (lambda df: df.assign(unit=df["unit"] - 1), "'unit' has values < 1"),
    "cycle not starting at 1": (_set(5, "cycle", 2), "do not start at 1"),
    "gap in cycles": (_set(2, "cycle", 4), "not increasing by exactly 1"),
    "duplicate cycle": (_set(2, "cycle", 2), "not increasing by exactly 1"),
    "empty frame": (lambda df: df.iloc[0:0], "no rows"),
}


@pytest.mark.parametrize("case", list(BROKEN_TRAJECTORIES))
def test_broken_trajectories_fail(frame: pd.DataFrame, case: str) -> None:
    mutate, message = BROKEN_TRAJECTORIES[case]
    with pytest.raises(DataValidationError, match=message):
        validate_trajectories(mutate(frame), "train")


BROKEN_RUL = {
    "wrong column": (lambda r: r.rename(columns={"rul": "y"}), "columns"),
    "null": (lambda r: r.assign(rul=[np.nan, 2.0, 1.0]), "null values"),
    "non-integer": (lambda r: r.astype(float), "must be integers"),
    "negative": (lambda r: r.assign(rul=[3, -1, 1]), "negative"),
    "count mismatch": (lambda r: r.iloc[:2], "2 values for 3 test units"),
}


@pytest.mark.parametrize("case", list(BROKEN_RUL))
def test_broken_rul_fails(frame: pd.DataFrame, rul: pd.DataFrame, case: str) -> None:
    mutate, message = BROKEN_RUL[case]
    with pytest.raises(DataValidationError, match=message):
        validate_rul(mutate(rul), frame)


def test_non_contiguous_test_units_fail(frame: pd.DataFrame, rul: pd.DataFrame) -> None:
    test = frame.assign(unit=frame["unit"].replace({3: 4}))
    with pytest.raises(DataValidationError, match="not contiguous"):
        validate_rul(rul, test)


@pytest.fixture
def fd_data(trajectories: TrajectoryFactory) -> FD001Data:
    return FD001Data(
        train=trajectories([40, 45]),
        test=trajectories([35, 31]),
        rul=pd.DataFrame({"rul": [5, 9]}),
    )


MATCHING = ExpectedFigures(
    train_rows=85, test_rows=66, train_units=2, test_units=2, rul_rows=2, min_test_cycles=31
)


def test_validate_fd001_passes_with_matching_figures(fd_data: FD001Data) -> None:
    validate_fd001(fd_data, MATCHING)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("train_rows", 86, "train_rows: expected 86, found 85"),
        ("test_rows", 1, "test_rows"),
        ("train_units", 3, "train_units"),
        ("test_units", 3, "test_units"),
        ("min_test_cycles", 32, "shortest test unit has 31 cycles"),
    ],
)
def test_validate_fd001_fails_on_figure_mismatch(
    fd_data: FD001Data, field: str, value: int, message: str
) -> None:
    expected = ExpectedFigures(**{**MATCHING.__dict__, field: value})
    with pytest.raises(DataValidationError, match=message):
        validate_fd001(fd_data, expected)


def test_validate_fd001_default_figures_reject_synthetic_data(fd_data: FD001Data) -> None:
    with pytest.raises(DataValidationError, match="expected 20631"):
        validate_fd001(fd_data)
