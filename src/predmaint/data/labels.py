"""RUL training labels (docs/05 §5, ADR-004)."""

import pandas as pd

from predmaint.config import RUL_CAP

RUL_LABEL = "rul"


def add_rul(frame: pd.DataFrame, rul_cap: int | None = RUL_CAP) -> pd.DataFrame:
    """Return a copy of a run-to-failure frame with an integer ``rul`` column.

    ``rul = max_cycle(unit) - cycle`` (0 at the failure cycle), capped at ``rul_cap``;
    ``rul_cap=None`` disables the cap. Only valid for trajectories that end at failure
    (the training set), never for truncated test trajectories.

    Raises:
        ValueError: if ``rul_cap`` is not a positive integer or ``None``.
    """
    if rul_cap is not None and rul_cap <= 0:
        raise ValueError(f"rul_cap must be positive or None, got {rul_cap}")
    out = frame.copy()
    rul = out.groupby("unit")["cycle"].transform("max") - out["cycle"]
    if rul_cap is not None:
        rul = rul.clip(upper=rul_cap)
    out[RUL_LABEL] = rul.astype("int64")
    return out
