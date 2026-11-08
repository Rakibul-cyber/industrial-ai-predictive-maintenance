"""Feature engineering: the single code path for training, evaluation and inference (docs/05 §4).

Each feature row describes one cycle of one engine using only that cycle and the
``window_size - 1`` cycles before it. Window statistics are computed independently per window
(not with a running accumulator), so the feature row for cycle *t* is bit-for-bit identical
whether or not cycles after *t* — or cycles older than the window — are present. Rows without a
full trailing window (``min_periods = window_size``) are dropped; for histories that start at
cycle 1 this is exactly ``cycle < window_size``.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from predmaint.config import WINDOW_SIZE
from predmaint.data.load import SENSOR_COLUMNS

#: ADR-005: a sensor with at most this many distinct training values is treated as constant.
SENSOR_MAX_UNIQUE_TO_DROP = 2


def sensor_stats(train: pd.DataFrame) -> pd.DataFrame:
    """Per-sensor unique count and std on ``train`` (for reporting the selection rule)."""
    return pd.DataFrame(
        {
            "nunique": [int(train[s].nunique()) for s in SENSOR_COLUMNS],
            "std": [float(train[s].std()) for s in SENSOR_COLUMNS],
        },
        index=SENSOR_COLUMNS,
    )


def select_sensors(
    train: pd.DataFrame, max_unique_to_drop: int = SENSOR_MAX_UNIQUE_TO_DROP
) -> list[str]:
    """Return the sensors to keep: those with more than ``max_unique_to_drop`` distinct values.

    Must be called on training data only. Operational settings are never candidates (ADR-005).
    """
    return [s for s in SENSOR_COLUMNS if train[s].nunique() > max_unique_to_drop]


def feature_names(sensors: Sequence[str]) -> list[str]:
    """Model input columns, in order: ``cycle`` then raw/rolling-mean/rolling-std per sensor."""
    names = ["cycle"]
    for sensor in sensors:
        names += [sensor, f"{sensor}_roll_mean", f"{sensor}_roll_std"]
    return names


def build_features(
    frame: pd.DataFrame,
    sensors: Sequence[str],
    window_size: int = WINDOW_SIZE,
    keep: Sequence[str] = (),
) -> pd.DataFrame:
    """Build one feature row per cycle that has a full trailing window, per engine.

    Args:
        frame: rows for one or more engines with ``unit``, ``cycle`` and every name in
            ``sensors``. Rows are sorted by (unit, cycle) internally; cycles are assumed to be
            consecutive per unit (enforced by ``data.validate`` / the API).
        sensors: kept sensors, in the order recorded at training time.
        window_size: trailing window length (>= 2, because the std uses ``ddof=1``).
        keep: extra columns (e.g. the ``rul`` label) copied through unchanged.

    Returns:
        DataFrame with columns ``["unit", *feature_names(sensors), *keep]`` and a fresh index.

    Raises:
        KeyError: if a required column is missing.
        ValueError: if ``window_size < 2``.
    """
    if window_size < 2:
        raise ValueError(f"window_size must be >= 2, got {window_size}")
    required = ["unit", "cycle", *sensors, *keep]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise KeyError(f"missing columns for feature building: {missing}")

    ordered = frame.sort_values(["unit", "cycle"], kind="stable").reset_index(drop=True)
    values = ordered[list(sensors)].to_numpy(dtype=float)
    units = ordered["unit"].to_numpy()
    starts = np.flatnonzero(np.r_[True, units[1:] != units[:-1]])
    ends = np.r_[starts[1:], len(units)]

    rows, means, stds = [], [], []
    for start, end in zip(starts, ends, strict=True):
        if end - start < window_size:
            continue
        windows = sliding_window_view(values[start:end], window_size, axis=0)
        means.append(windows.mean(axis=-1))
        stds.append(windows.std(axis=-1, ddof=1))
        rows.append(np.arange(start + window_size - 1, end))

    columns = ["unit", *feature_names(sensors), *keep]
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype=float) for c in columns})
    index = np.concatenate(rows)
    mean_block = np.concatenate(means)
    std_block = np.concatenate(stds)

    data: dict[str, np.ndarray] = {
        "unit": units[index],
        "cycle": ordered["cycle"].to_numpy()[index],
    }
    for j, sensor in enumerate(sensors):
        data[sensor] = values[index, j]
        data[f"{sensor}_roll_mean"] = mean_block[:, j]
        data[f"{sensor}_roll_std"] = std_block[:, j]
    for column in keep:
        data[column] = ordered[column].to_numpy()[index]
    return pd.DataFrame(data, columns=columns)


def last_rows(features: pd.DataFrame) -> pd.DataFrame:
    """Return the feature row of each unit's last available cycle, sorted by unit."""
    idx = features.groupby("unit")["cycle"].idxmax()
    return features.loc[idx.sort_index().to_numpy()].reset_index(drop=True)
