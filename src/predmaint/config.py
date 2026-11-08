"""Central configuration (docs/05 §4, docs/10 §3).

Model constants below are the single source shared by training, feature building and
inference. Runtime settings (paths, URLs, log level) come from environment variables.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

# --- Model constants (ADR-004, ADR-005, ADR-025) --------------------------------------------
DATASET_SUBSET: Final = "FD001"
WINDOW_SIZE: Final = 30
RUL_CAP: Final = 125
RANDOM_SEED: Final = 42
CV_FOLDS: Final = 5
#: ADR-025: predicted RUL < 20 → "critical"; 20 ≤ RUL < 60 → "warning"; RUL ≥ 60 → "healthy".
HEALTH_BANDS: Final = MappingProxyType({"critical_below": 20, "warning_below": 60})

# --- Environment-driven settings -------------------------------------------------------------
_DEFAULTS: Final = MappingProxyType(
    {
        "DATA_DIR": "data",
        "MODEL_DIR": "models",
        "LOG_LEVEL": "INFO",
        "API_BASE_URL": "http://localhost:8000",
        "API_TIMEOUT_SECONDS": "10",
    }
)


@dataclass(frozen=True)
class Settings:
    """Runtime settings resolved from the environment."""

    data_dir: Path
    model_dir: Path
    log_level: str
    api_base_url: str
    api_timeout_seconds: float

    @property
    def raw_dir(self) -> Path:
        """Directory for the downloaded archive (``$DATA_DIR/raw``)."""
        return self.data_dir / "raw"

    @property
    def cmapss_dir(self) -> Path:
        """Directory holding the extracted C-MAPSS text files."""
        return self.raw_dir / "CMAPSSData"


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or _DEFAULTS[name]


def get_settings() -> Settings:
    """Read settings from environment variables, falling back to documented defaults.

    Raises:
        ValueError: if ``API_TIMEOUT_SECONDS`` is not a positive number.
    """
    raw_timeout = _env("API_TIMEOUT_SECONDS")
    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise ValueError(f"API_TIMEOUT_SECONDS must be a number, got {raw_timeout!r}") from exc
    if timeout <= 0:
        raise ValueError(f"API_TIMEOUT_SECONDS must be positive, got {timeout}")
    return Settings(
        data_dir=Path(_env("DATA_DIR")),
        model_dir=Path(_env("MODEL_DIR")),
        log_level=_env("LOG_LEVEL").upper(),
        api_base_url=_env("API_BASE_URL").rstrip("/"),
        api_timeout_seconds=timeout,
    )
