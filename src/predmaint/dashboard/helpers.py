"""Pure helpers for the dashboard: ground truth, request bodies, trajectories, chart data.

No model, feature or inference code here — predictions always come from the API.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from predmaint.data.load import RUL_COLUMN, SENSOR_COLUMNS, SETTING_COLUMNS, FD001Data

#: Trajectory sampling step in cycles (RUL trajectory section).
TRAJECTORY_STEP = 5
DATASETS = ("test", "train")


def true_rul(rul_at_last_cycle: int, last_cycle: int, n: int) -> int:
    """True RUL after observing ``n`` cycles.

    Train engines run to failure (``rul_at_last_cycle = 0`` → ``max_cycle - n``); test engines
    stop early (``RUL_FD001 + (last_cycle - n)``).
    """
    if not 1 <= n <= last_cycle:
        raise ValueError(f"n must be in 1..{last_cycle}, got {n}")
    return int(rul_at_last_cycle + (last_cycle - n))


@dataclass(frozen=True)
class EngineHistory:
    """One engine's raw cycles plus the ground truth needed to score it."""

    dataset: str
    unit: int
    frame: pd.DataFrame
    rul_at_last_cycle: int

    @property
    def last_cycle(self) -> int:
        return int(self.frame["cycle"].max())

    def true_rul(self, n: int) -> int:
        return true_rul(self.rul_at_last_cycle, self.last_cycle, n)


def units(data: FD001Data, dataset: str) -> list[int]:
    frame = data.train if dataset == "train" else data.test
    return sorted(int(u) for u in frame["unit"].unique())


def engine_history(data: FD001Data, dataset: str, unit: int) -> EngineHistory:
    """Select one engine; ``rul_at_last_cycle`` is 0 for train and ``RUL_FD001[unit]`` for test."""
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    frame = data.train if dataset == "train" else data.test
    engine = frame[frame["unit"] == unit].sort_values("cycle").reset_index(drop=True)
    if engine.empty:
        raise ValueError(f"no {dataset} engine {unit}")
    rul = 0 if dataset == "train" else int(data.rul[RUL_COLUMN].iloc[unit - 1])
    return EngineHistory(dataset, unit, engine, rul)


def engine_payload(history: EngineHistory, n: int) -> dict[str, Any]:
    """``POST /predict`` body with the engine's first ``n`` cycles (docs/06 §3.1)."""
    columns = ["cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS]
    rows = history.frame[history.frame["cycle"] <= n][columns]
    records = [
        {"cycle": int(r["cycle"]), **{c: float(r[c]) for c in columns[1:]}}
        for r in rows.to_dict("records")
    ]
    return {"unit_id": f"{history.dataset}-unit-{history.unit}", "records": records}


def trajectory_cycles(last_cycle: int, window: int, step: int = TRAJECTORY_STEP) -> list[int]:
    """Cycles at which to query the API: window, window+step, …, always ending at last_cycle."""
    if step < 1:
        raise ValueError("step must be >= 1")
    if last_cycle < window:
        return []
    cycles = list(range(window, last_cycle + 1, step))
    if cycles[-1] != last_cycle:
        cycles.append(last_cycle)
    return cycles


def true_rul_series(history: EngineHistory, cycles: list[int], rul_cap: int | None) -> pd.DataFrame:
    """True RUL at each cycle, raw and capped like the training target (``min(rul, cap)``)."""
    raw = np.array([history.true_rul(n) for n in cycles], dtype=int)
    capped = raw if rul_cap is None else np.minimum(raw, rul_cap)
    return pd.DataFrame({"cycle": cycles, "true_rul": raw, "true_rul_capped": capped})


class TrajectoryError(RuntimeError):
    """A /predict call failed while computing a trajectory; carries the error envelope."""

    def __init__(self, cycle: int, body: Mapping[str, Any]) -> None:
        self.cycle, self.body = cycle, dict(body)
        super().__init__(f"/predict failed at cycle {cycle}: {self.body}")


def compute_trajectory(
    history: EngineHistory,
    predict: Callable[[dict[str, Any]], Any],
    window: int,
    rul_cap: int | None,
    step: int = TRAJECTORY_STEP,
) -> pd.DataFrame:
    """Call ``predict`` (an ``ApiClient.predict``) every ``step`` cycles.

    Returns columns ``cycle, predicted_rul, health_status, true_rul, true_rul_capped``.

    Raises:
        TrajectoryError: on the first failed call (so callers never cache partial results).
    """
    cycles = trajectory_cycles(history.last_cycle, window, step)
    predicted, status = [], []
    for n in cycles:
        result = predict(engine_payload(history, n))
        if not result.ok:
            raise TrajectoryError(n, result.body)
        predicted.append(float(result.body["predicted_rul"]))
        status.append(result.body["health_status"])
    out = true_rul_series(history, cycles, rul_cap)
    out.insert(1, "predicted_rul", predicted)
    out.insert(2, "health_status", status)
    return out


def sensor_chart_frame(history: EngineHistory, sensors: list[str]) -> pd.DataFrame:
    """Long-format sensor readings, standardised per sensor over the engine (display only)."""
    frame = history.frame[["cycle", *sensors]]
    values = frame[sensors]
    std = values.std(ddof=0).replace(0, 1.0)
    scaled = (values - values.mean()) / std
    scaled.insert(0, "cycle", frame["cycle"])
    return scaled.melt(id_vars="cycle", var_name="sensor", value_name="standardised value")
