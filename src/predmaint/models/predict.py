"""Prediction from engineered features (docs/05 §10).

Milestone 2 provides the clamped prediction step shared by evaluation and, from Milestone 3,
the full inference contract (``load_model`` / ``predict_rul`` / ``health_band``).
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def predict_from_features(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Predict RUL for feature rows and clamp at 0 (ADR-018)."""
    return np.maximum(pipeline.predict(features), 0.0)
