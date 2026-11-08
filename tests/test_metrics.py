"""Tests for metric functions against hand-computed values (docs/05 §7, ADR-009)."""

import math
from collections.abc import Callable

import pytest

from predmaint.models.metrics import mae, nasa_score, rmse


def test_mae_and_rmse_hand_computed() -> None:
    y_true = [10.0, 20.0, 30.0, 40.0]
    y_pred = [12.0, 18.0, 33.0, 40.0]  # errors +2, -2, +3, 0
    assert mae(y_true, y_pred) == pytest.approx(7 / 4)
    assert rmse(y_true, y_pred) == pytest.approx(math.sqrt((4 + 4 + 9 + 0) / 4))


def test_nasa_score_known_values() -> None:
    assert nasa_score([50.0], [60.0]) == pytest.approx(math.e - 1)  # late by 10 → e^(10/10) - 1
    assert nasa_score([50.0], [37.0]) == pytest.approx(math.e - 1)  # early by 13 → e^(13/13) - 1
    assert nasa_score([50.0], [50.0]) == 0.0


def test_nasa_score_penalises_late_more_than_early() -> None:
    late = nasa_score([50.0], [70.0])
    early = nasa_score([50.0], [30.0])
    assert late == pytest.approx(math.exp(2.0) - 1)
    assert early == pytest.approx(math.exp(20 / 13) - 1)
    assert late > early


def test_nasa_score_sums_over_units() -> None:
    assert nasa_score([50.0, 50.0], [60.0, 37.0]) == pytest.approx(2 * (math.e - 1))


@pytest.mark.parametrize("metric", [mae, rmse, nasa_score])
def test_shape_mismatch_and_empty_rejected(metric: Callable[..., float]) -> None:
    with pytest.raises(ValueError, match="equal length"):
        metric([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="empty"):
        metric([], [])
