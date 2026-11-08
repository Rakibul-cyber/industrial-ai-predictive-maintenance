"""Tests for the shared feature function (FR-005): values, causality, isolation, eligibility."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from predmaint.features.build import (
    build_features,
    feature_names,
    last_rows,
    select_sensors,
)

TrajectoryFactory = Callable[..., pd.DataFrame]
SENSORS = ["s_2", "s_3"]
W = 5


@pytest.fixture
def frame(trajectories: TrajectoryFactory) -> pd.DataFrame:
    return trajectories([12, 9, 4], seed=3)  # unit 3 is shorter than the window


def test_feature_names_order_and_count() -> None:
    assert feature_names(SENSORS) == [
        "cycle",
        "s_2",
        "s_2_roll_mean",
        "s_2_roll_std",
        "s_3",
        "s_3_roll_mean",
        "s_3_roll_std",
    ]
    assert len(feature_names([f"s_{i}" for i in range(14)])) == 1 + 3 * 14


def test_row_count_and_columns(frame: pd.DataFrame) -> None:
    out = build_features(frame, SENSORS, W)
    assert list(out.columns) == ["unit", *feature_names(SENSORS)]
    # (12 - 5 + 1) + (9 - 5 + 1) + 0 rows; unit 3 has no full window
    assert len(out) == 8 + 5
    assert out.groupby("unit")["cycle"].min().to_dict() == {1: W, 2: W}
    assert not out.isna().any().any()


def test_values_match_hand_computation(frame: pd.DataFrame) -> None:
    out = build_features(frame, SENSORS, W)
    unit1 = frame[frame["unit"] == 1].set_index("cycle")
    row = out[(out["unit"] == 1) & (out["cycle"] == 9)].iloc[0]
    window = unit1.loc[5:9, "s_3"].to_numpy()
    assert row["s_3"] == unit1.loc[9, "s_3"]
    assert row["s_3_roll_mean"] == pytest.approx(window.mean(), abs=1e-12)
    assert row["s_3_roll_std"] == pytest.approx(window.std(ddof=1), abs=1e-12)


def test_per_engine_isolation(frame: pd.DataFrame) -> None:
    """Unit 1's last values never enter unit 2's windows, and vice versa."""
    shifted = frame.copy()
    shifted.loc[shifted["unit"] == 1, SENSORS] += 1_000.0
    base = build_features(frame, SENSORS, W)
    out = build_features(shifted, SENSORS, W)
    unit2 = out["unit"] == 2
    pd.testing.assert_frame_equal(out[unit2], base[unit2])
    alone = build_features(frame[frame["unit"] == 2], SENSORS, W)
    pd.testing.assert_frame_equal(base[unit2].reset_index(drop=True), alone)


@pytest.mark.parametrize("t", [W, 7, 12])
def test_truncation_invariance(frame: pd.DataFrame, t: int) -> None:
    """The row for cycle t is identical with or without the cycles after t."""
    full = build_features(frame, SENSORS, W)
    truncated = build_features(frame[(frame["unit"] != 1) | (frame["cycle"] <= t)], SENSORS, W)
    pick = (full["unit"] == 1) & (full["cycle"] == t)
    pick_t = (truncated["unit"] == 1) & (truncated["cycle"] == t)
    np.testing.assert_array_equal(full[pick].to_numpy(), truncated[pick_t].to_numpy())


def test_only_the_trailing_window_matters(frame: pd.DataFrame) -> None:
    """Dropping history older than the window (as an API client may) changes nothing."""
    full = build_features(frame, SENSORS, W)
    recent = frame[(frame["unit"] == 1) & (frame["cycle"] > 12 - W)]
    out = build_features(recent, SENSORS, W)
    assert len(out) == 1
    np.testing.assert_array_equal(out.to_numpy(), full[full["unit"] == 1].tail(1).to_numpy())


def test_row_order_of_input_does_not_matter(frame: pd.DataFrame) -> None:
    shuffled = frame.sample(frac=1.0, random_state=0)
    pd.testing.assert_frame_equal(
        build_features(shuffled, SENSORS, W), build_features(frame, SENSORS, W)
    )


def test_keep_columns_pass_through(frame: pd.DataFrame) -> None:
    labelled = frame.assign(rul=np.arange(len(frame)))
    out = build_features(labelled, SENSORS, W, keep=["rul"])
    source = labelled.set_index(["unit", "cycle"])["rul"]
    assert out["rul"].tolist() == [
        source[(u, c)] for u, c in zip(out["unit"], out["cycle"], strict=True)
    ]


def test_missing_column_and_bad_window_rejected(frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="s_2"):
        build_features(frame.drop(columns=["s_2"]), SENSORS, W)
    with pytest.raises(ValueError, match="window_size"):
        build_features(frame, SENSORS, 1)


def test_no_eligible_rows_gives_empty_frame(frame: pd.DataFrame) -> None:
    out = build_features(frame[frame["unit"] == 3], SENSORS, W)
    assert out.empty
    assert list(out.columns) == ["unit", *feature_names(SENSORS)]


def test_last_rows_picks_each_units_last_cycle(frame: pd.DataFrame) -> None:
    last = last_rows(build_features(frame, SENSORS, W))
    assert last[["unit", "cycle"]].values.tolist() == [[1, 12], [2, 9]]


def test_sensor_selection_drops_constant_and_two_valued(
    trajectories: TrajectoryFactory,
) -> None:
    train = trajectories([20, 20])
    train["s_6"] = np.where(train["cycle"] % 2 == 0, 21.60, 21.61)  # 2 unique values
    train["s_10"] = 1.3  # constant
    train["s_16"] = 0.03 + 1e-18 * np.arange(len(train))  # numerically "constant"
    kept = select_sensors(train)
    assert "s_1" not in kept  # constant in the fixture
    assert {"s_6", "s_10"}.isdisjoint(kept)
    assert {"s_2", "s_3", "s_21"} <= set(kept)
    assert not any(c.startswith("setting_") for c in kept)
