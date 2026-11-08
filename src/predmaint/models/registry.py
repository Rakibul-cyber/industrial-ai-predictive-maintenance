"""Model pipeline factories (docs/05 §2-3).

Every pipeline starts with a ``ColumnTransformer`` that selects exactly the training-time
feature columns, so the saved artifact itself carries the kept-sensor feature list and rejects
inputs that lack those columns. Scaling (where used) is fitted inside the pipeline, i.e. on the
training rows of each fold only.
"""

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#: Candidates implemented so far. The primary model is added in Milestone 3.
MODEL_NAMES: tuple[str, ...] = ("baseline",)

#: Short estimator tags used in ``model_version`` (docs/05 §9).
ESTIMATOR_TAGS: dict[str, str] = {"baseline": "ridge"}


def _select(columns: Sequence[str]) -> ColumnTransformer:
    return ColumnTransformer([("features", "passthrough", list(columns))], remainder="drop")


def make_baseline(columns: Sequence[str]) -> Pipeline:
    """ADR-006: select features → StandardScaler → Ridge(alpha=1.0)."""
    return Pipeline(
        [("select", _select(columns)), ("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]
    )


def make_model(name: str, columns: Sequence[str], seed: int) -> Pipeline:
    """Build an unfitted pipeline for candidate ``name``.

    ``seed`` is accepted for every candidate so stochastic models receive the configured seed;
    Ridge is deterministic and does not use it.

    Raises:
        ValueError: for an unknown model name.
    """
    if name == "baseline":
        return make_baseline(columns)
    raise ValueError(f"unknown model {name!r}; available: {', '.join(MODEL_NAMES)}")


def pipeline_feature_names(pipeline: Pipeline) -> list[str]:
    """Return the feature columns stored in a pipeline built by this module."""
    select: ColumnTransformer = pipeline.named_steps["select"]
    return list(select.transformers[0][2])
