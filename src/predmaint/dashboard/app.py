"""Streamlit dashboard for the RUL API (docs/02 §6).

Run: ``streamlit run src/predmaint/dashboard/app.py``. Every prediction comes from the API via
``api_client``; raw FD001 data is read only for display and for building request bodies.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from predmaint.config import HEALTH_BANDS, RUL_CAP, WINDOW_SIZE, get_settings
from predmaint.dashboard.api_client import ApiClient, ApiResult
from predmaint.dashboard.helpers import (
    DATASETS,
    TRAJECTORY_STEP,
    TrajectoryError,
    compute_trajectory,
    engine_history,
    engine_payload,
    sensor_chart_frame,
    units,
)
from predmaint.data.load import FD001Data, load_fd001

STATUS_COLORS = {"critical": "red", "warning": "orange", "healthy": "green"}
DOWNLOAD_HINT = "Run `python scripts/download_data.py` to download FD001 (≈ 12 MB)."

st.set_page_config(page_title="Predictive Maintenance — RUL", page_icon="🛠️", layout="wide")
settings = get_settings()
client = ApiClient(settings.api_base_url, settings.api_timeout_seconds)


@st.cache_data(show_spinner=False)
def load_data(cmapss_dir: str) -> FD001Data | None:
    """Raw FD001 for display; ``None`` when the dataset is not downloaded."""
    try:
        return load_fd001(Path(cmapss_dir))
    except (FileNotFoundError, ValueError):
        return None


@st.cache_data(show_spinner="Calling /predict along the engine's life…")
def trajectory(
    cmapss_dir: str, dataset: str, unit: int, step: int, model_version: str, base_url: str
) -> pd.DataFrame:
    """Predicted vs true RUL every ``step`` cycles; cached per (dataset, unit, step, model)."""
    data = load_data(cmapss_dir)
    if data is None:
        raise FileNotFoundError(cmapss_dir)
    history = engine_history(data, dataset, unit)
    api = ApiClient(base_url, settings.api_timeout_seconds)
    return compute_trajectory(history, api.predict, WINDOW_SIZE, RUL_CAP, step)


def show_error(result: ApiResult) -> None:
    error = result.error or {}
    st.error(f"{error.get('code', 'ERROR')}: {error.get('message', '')}")
    st.json(result.body, expanded=False)


def status_badge(status: str) -> None:
    st.badge(status.upper(), color=STATUS_COLORS.get(status, "gray"))  # type: ignore[arg-type]


# --- 1. Status bar ------------------------------------------------------------------------
st.title("Predictive Maintenance — Remaining Useful Life")
health = client.health()
model_loaded = bool(health.ok and health.body.get("model_loaded"))
model_version = health.body.get("model_version") if health.ok else None
col_api, col_model, col_version = st.columns(3)
col_api.metric("API reachable", "yes" if health.reachable else "no")
col_model.metric("Model loaded", "yes" if model_loaded else "no")
col_version.metric("Model version", model_version or "—")
if not health.reachable:
    st.error(
        f"API unreachable at {settings.api_base_url}. Start it with `python -m predmaint.api.main`."
    )
elif not model_loaded:
    st.warning("API is up but no model is loaded (degraded). Predictions are disabled.")

info = client.model_info() if model_loaded else None
model_meta: dict[str, Any] = info.body if info is not None and info.ok else {}
sensors: list[str] = model_meta.get("required_sensors", [])
bands: dict[str, Any] = model_meta.get("health_bands", dict(HEALTH_BANDS))

cmapss_dir = str(settings.cmapss_dir)
data = load_data(cmapss_dir)

with st.sidebar:
    st.header("Engine")
    if data is None:
        st.info(f"Dataset not found in `{cmapss_dir}`. {DOWNLOAD_HINT}")
        dataset, unit = None, None
    else:
        dataset = st.radio("Dataset", DATASETS, horizontal=True)
        unit = st.selectbox("Unit", units(data, dataset))
    st.caption(f"API: {settings.api_base_url}")

history = engine_history(data, dataset, unit) if data is not None and unit is not None else None
explorer, trajectory_tab, upload, model_tab = st.tabs(
    ["Engine explorer", "RUL trajectory", "Upload", "Model info"]
)

# --- 2. Engine explorer -------------------------------------------------------------------
with explorer:
    if history is None:
        st.info(f"The engine explorer needs the raw dataset. {DOWNLOAD_HINT}")
    elif history.last_cycle < WINDOW_SIZE:
        st.warning(f"Engine has {history.last_cycle} cycles; at least {WINDOW_SIZE} are needed.")
    else:
        n = st.slider(
            "Cycles observed so far",
            min_value=WINDOW_SIZE,
            max_value=history.last_cycle,
            value=history.last_cycle,
        )
        truth = history.true_rul(n)
        left, mid, right = st.columns(3)
        if model_loaded:
            result = client.predict(engine_payload(history, n))
            if result.ok:
                left.metric("Predicted RUL (cycles)", f"{result.body['predicted_rul']:.1f}")
                with mid:
                    st.caption("Health status")
                    status_badge(result.body["health_status"])
                right.metric("True RUL (cycles)", truth)
                st.caption(
                    f"cycles used: {result.body['cycles_used']} of {result.body['cycles_received']}"
                    f" sent · last cycle {result.body['last_cycle']}"
                    f" · model {result.body['model_version']}"
                )
            else:
                show_error(result)
        else:
            left.metric("Predicted RUL (cycles)", "—")
            right.metric("True RUL (cycles)", truth)
            st.info("Prediction disabled: the API is unreachable or has no model.")
        if sensors:
            st.subheader("Kept sensors (standardised per sensor), current cycle marked")
            st.vega_lite_chart(
                sensor_chart_frame(history, sensors),
                {
                    "layer": [
                        {
                            "mark": {"type": "line", "strokeWidth": 1},
                            "encoding": {
                                "x": {"field": "cycle", "type": "quantitative"},
                                "y": {"field": "standardised value", "type": "quantitative"},
                                "color": {"field": "sensor", "type": "nominal"},
                            },
                        },
                        {
                            "mark": {"type": "rule", "color": "black", "strokeDash": [4, 4]},
                            "encoding": {"x": {"datum": n, "type": "quantitative"}},
                        },
                    ]
                },
                width="stretch",
            )
        else:
            st.caption("Sensor chart needs the kept-sensor list from /model/info.")

# --- 3. RUL trajectory --------------------------------------------------------------------
with trajectory_tab:
    if history is None:
        st.info(f"The RUL trajectory needs the raw dataset. {DOWNLOAD_HINT}")
    else:
        key = (history.dataset, history.unit, TRAJECTORY_STEP, model_version)
        st.write(
            f"Calls `/predict` every {TRAJECTORY_STEP} cycles from cycle {WINDOW_SIZE} to "
            f"{history.last_cycle} for {history.dataset} engine {history.unit}."
        )
        if st.button("Compute RUL trajectory", disabled=not model_loaded):
            st.session_state["trajectory_key"] = key
        if st.session_state.get("trajectory_key") == key and model_loaded:
            try:
                frame = trajectory(cmapss_dir, *key[:3], str(model_version), settings.api_base_url)
            except TrajectoryError as exc:
                st.error(f"/predict failed at cycle {exc.cycle}")
                st.json(exc.body, expanded=False)
            else:
                long = frame.melt(
                    id_vars="cycle",
                    value_vars=["predicted_rul", "true_rul_capped"],
                    var_name="series",
                    value_name="RUL (cycles)",
                )
                thresholds = pd.DataFrame(
                    {
                        "RUL (cycles)": [bands["critical_below"], bands["warning_below"]],
                        "band": ["critical below", "warning below"],
                    }
                )
                st.vega_lite_chart(
                    long,
                    {
                        "layer": [
                            {
                                "mark": {"type": "line", "point": True},
                                "encoding": {
                                    "x": {"field": "cycle", "type": "quantitative"},
                                    "y": {"field": "RUL (cycles)", "type": "quantitative"},
                                    "color": {"field": "series", "type": "nominal"},
                                },
                            },
                            {
                                "data": {"values": thresholds.to_dict("records")},
                                "mark": {"type": "rule", "strokeDash": [6, 4], "color": "gray"},
                                "encoding": {
                                    "y": {"field": "RUL (cycles)", "type": "quantitative"}
                                },
                            },
                        ]
                    },
                    width="stretch",
                )
                st.caption(
                    f"true_rul_capped = min(true RUL, {RUL_CAP}), the training target. Dashed: "
                    f"health bands (critical < {bands['critical_below']}, "
                    f"warning < {bands['warning_below']})."
                )
                with st.expander("Trajectory table"):
                    st.dataframe(frame, hide_index=True)
        elif not model_loaded:
            st.info("Trajectory disabled: the API is unreachable or has no model.")

# --- 4. Upload ----------------------------------------------------------------------------
with upload:
    st.write("Upload a `/predict` request body (docs/06 format), e.g. a file from `examples/`.")
    uploaded = st.file_uploader("Request JSON", type=["json"])
    if uploaded is not None:
        try:
            payload = json.loads(uploaded.getvalue())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            st.error(f"Not valid JSON: {exc}")
        else:
            if st.button("Send to /predict", disabled=not model_loaded):
                result = client.predict(payload)
                st.caption(f"HTTP {result.status_code}")
                if result.ok:
                    status_badge(result.body["health_status"])
                st.json(result.body)

# --- 5. Model info ------------------------------------------------------------------------
with model_tab:
    if info is None:
        st.info("Model info is available once the API reports a loaded model.")
    elif not info.ok:
        show_error(info)
    else:
        m = info.body["metrics"]
        st.table(
            pd.DataFrame(
                {
                    "value": {
                        "Model": info.body["model_name"],
                        "Version": info.body["model_version"],
                        "Selected by": info.body["selected_by"],
                        "Trained at": info.body["trained_at"],
                        "Git SHA": info.body["git_sha"][:12],
                        "Data": f"{info.body['data_subset']} (train sha256 "
                        f"{info.body['train_sha256'][:12]}…)",
                        "Window size / RUL cap": f"{info.body['window_size']} / "
                        f"{info.body['rul_cap']} cycles",
                        "Features": f"{info.body['n_features']} "
                        f"({len(info.body['required_sensors'])} sensors)",
                        "Health bands": f"critical < {bands['critical_below']} ≤ warning < "
                        f"{bands['warning_below']} ≤ healthy",
                        "scikit-learn version match": str(info.body["sklearn_version_match"]),
                    }
                }
            )
        )
        st.subheader("Metrics")
        st.table(
            pd.DataFrame(
                {
                    "RMSE": [
                        f"{m['cv']['rmse_mean']:.3f} ± {m['cv']['rmse_std']:.3f}",
                        f"{m['test']['rmse']:.3f}",
                        f"{m['test_clipped_truth']['rmse']:.3f}",
                    ],
                    "MAE": [
                        f"{m['cv']['mae_mean']:.3f} ± {m['cv']['mae_std']:.3f}",
                        f"{m['test']['mae']:.3f}",
                        f"{m['test_clipped_truth']['mae']:.3f}",
                    ],
                    "NASA score": ["—", f"{m['test']['nasa_score']:.1f}", "—"],
                },
                index=[
                    "CV (5-fold GroupKFold, mean ± std)",
                    "Official test (unclipped truth)",
                    "Official test (truth clipped at cap)",
                ],
            )
        )
