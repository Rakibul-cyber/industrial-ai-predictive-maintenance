"""Tests for RUL labelling (FR-004)."""

from collections.abc import Callable

import pandas as pd
import pytest

from predmaint.data.labels import add_rul


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"unit": [1] * 5 + [2] * 3, "cycle": [1, 2, 3, 4, 5, 1, 2, 3]})


def test_rul_is_max_cycle_minus_cycle_per_unit() -> None:
    out = add_rul(_frame(), rul_cap=None)
    assert out["rul"].tolist() == [4, 3, 2, 1, 0, 2, 1, 0]
    assert out["rul"].dtype == "int64"


def test_cap_is_applied() -> None:
    assert add_rul(_frame(), rul_cap=2)["rul"].tolist() == [2, 2, 2, 1, 0, 2, 1, 0]


def test_default_cap_is_125(trajectories: Callable[..., pd.DataFrame]) -> None:
    out = add_rul(trajectories([200]))
    assert out["rul"].max() == 125
    assert out["rul"].iloc[-1] == 0
    assert out.loc[out["cycle"] == 200 - 125, "rul"].item() == 125


def test_input_is_not_mutated() -> None:
    frame = _frame()
    add_rul(frame)
    assert "rul" not in frame.columns


@pytest.mark.parametrize("cap", [0, -5])
def test_invalid_cap_rejected(cap: int) -> None:
    with pytest.raises(ValueError, match="rul_cap"):
        add_rul(_frame(), rul_cap=cap)
