"""Evaluation metrics (docs/05 §7). MAE and RMSE are primary; the NASA score is secondary."""

import numpy as np
from numpy.typing import ArrayLike

#: ADR-009: early predictions (d < 0) use divisor 13, late predictions (d >= 0) divisor 10.
NASA_EARLY_DIVISOR = 13.0
NASA_LATE_DIVISOR = 10.0


def _errors(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape or true.ndim != 1:
        raise ValueError(f"expected two 1-D arrays of equal length, got {true.shape}, {pred.shape}")
    if true.size == 0:
        raise ValueError("cannot compute a metric on empty arrays")
    return pred - true


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(_errors(y_true, y_pred))))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean(_errors(y_true, y_pred) ** 2)))


def nasa_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """PHM08 asymmetric score, summed over units; ``d = predicted - true`` (ADR-009).

    Late predictions (``d >= 0``) are penalised more heavily than early ones.
    """
    d = _errors(y_true, y_pred)
    penalties = np.where(
        d < 0, np.exp(-d / NASA_EARLY_DIVISOR) - 1.0, np.exp(d / NASA_LATE_DIVISOR) - 1.0
    )
    return float(np.sum(penalties))
