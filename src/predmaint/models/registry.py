"""Model pipeline factories (docs/05 §2-3).

Every pipeline starts with a ``ColumnTransformer`` that selects exactly the training-time
feature columns, so the saved artifact itself carries the kept-sensor feature list and rejects
inputs that lack those columns. Scaling (where used) is fitted inside the pipeline, i.e. on the
training rows of each fold only.
"""

from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES: tuple[str, ...] = ("baseline", "primary")

#: Short estimator tags used in candidate ``model_version`` strings (docs/05 §9).
ESTIMATOR_TAGS: dict[str, str] = {"baseline": "ridge", "primary": "hgb"}

BASELINE_PARAMS: dict[str, Any] = {"alpha": 1.0}

#: ADR-026: the single bounded search for the primary model — 3 x 2 x 2 = 12 configurations.
#: (learning_rate, max_iter) move together so every pair has a comparable learning budget.
#: Internal early stopping stays off: its validation split is row-wise and would mix cycles
#: of the same engine (ADR-007).
PRIMARY_SEARCH_GRID: list[dict[str, Any]] = [
    {
        "learning_rate": lr,
        "max_iter": iters,
        "max_leaf_nodes": leaves,
        "l2_regularization": l2,
    }
    for (lr, iters), leaves, l2 in product(
        [(0.1, 100), (0.05, 200), (0.03, 400)], [15, 31], [0.0, 1.0]
    )
]


def _select(columns: Sequence[str]) -> ColumnTransformer:
    return ColumnTransformer([("features", "passthrough", list(columns))], remainder="drop")


def make_baseline(columns: Sequence[str]) -> Pipeline:
    """ADR-006: select features → StandardScaler → Ridge(alpha=1.0)."""
    return Pipeline(
        [
            ("select", _select(columns)),
            ("scaler", StandardScaler()),
            ("model", Ridge(**BASELINE_PARAMS)),
        ]
    )


def make_primary(columns: Sequence[str], seed: int, params: Mapping[str, Any]) -> Pipeline:
    """ADR-007/026: select features → HistGradientBoostingRegressor (no scaler: trees are
    invariant to monotonic feature scaling)."""
    model = HistGradientBoostingRegressor(early_stopping=False, random_state=seed, **params)
    return Pipeline([("select", _select(columns)), ("model", model)])


def make_quantile_model(
    columns: Sequence[str], seed: int, params: Mapping[str, Any], quantile: float
) -> Pipeline:
    """ADR-034: same pipeline as the primary model, with ``loss="quantile"`` at ``quantile``."""
    model = HistGradientBoostingRegressor(
        loss="quantile", quantile=quantile, early_stopping=False, random_state=seed, **params
    )
    return Pipeline([("select", _select(columns)), ("model", model)])


def make_model(
    name: str, columns: Sequence[str], seed: int, params: Mapping[str, Any] | None = None
) -> Pipeline:
    """Build an unfitted pipeline for candidate ``name``.

    Raises:
        ValueError: for an unknown model name, or ``primary`` without ``params``.
    """
    if name == "baseline":
        return make_baseline(columns)
    if name == "primary":
        if params is None:
            raise ValueError("primary model needs hyperparameters (see PRIMARY_SEARCH_GRID)")
        return make_primary(columns, seed, params)
    raise ValueError(f"unknown model {name!r}; available: {', '.join(MODEL_NAMES)}")


def pipeline_feature_names(pipeline: Pipeline) -> list[str]:
    """Return the feature columns stored in a pipeline built by this module."""
    select: ColumnTransformer = pipeline.named_steps["select"]
    return list(select.transformers[0][2])
