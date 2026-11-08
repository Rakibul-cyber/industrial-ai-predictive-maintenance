# 05 — ML Design

## 1. Objective

Regress **RUL in cycles** for an FD001 engine from its observed sensor history, evaluated at the last observed cycle of each official test unit. Primary metrics: MAE, RMSE (lower is better). Classification accuracy is not used.

## 2. Baseline model (ADR-006)

`Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])`. Interpretable, fast, and sets the bar the primary model must beat.

## 3. Primary model (ADR-007)

`HistGradientBoostingRegressor(early_stopping=False, random_state=42)`, all other hyperparameters at scikit-learn 1.9.1 defaults (no scaler). Hyperparameters are fixed a priori — no tuning in the MVP. `early_stopping=False` because the estimator's internal validation split would be row-wise, mixing cycles of the same engine.

## 4. Feature strategy (ADR-005)

Config defaults: `WINDOW_SIZE = 30`, `RUL_CAP = 125`. Both constants are defined **once**, in `predmaint/config.py`, and imported by training (`models/train.py`), feature building (`features/build.py`), and inference (`models/predict.py` / API); no module re-declares them. Training writes the values used into metadata (`config.window_size`, `config.rul_cap`), and inference uses the metadata values so a served model always matches how it was trained. Lower-case `window_size` / `rul_cap` below refer to these values.

1. **Candidate columns:** `s_1 … s_21`. `setting_1..3` are excluded (FD001 has one operating condition per readme).
2. **Sensor drop rule (train only):** drop a sensor if it has **≤ 2 unique values** in `train_FD001`. The kept list is computed in M2 on real data, printed with each sensor's std and unique count, and recorded in [04](04_DATASET_SPECIFICATION.md) and metadata (`required_sensors`, `dropped_columns`). If inspection shows the rule misclassifies a sensor, the rule is amended through [15](15_CHANGE_CONTROL.md) — not silently.
3. **Per kept sensor `s`, per row (computed within each unit, trailing only):**
   - `s` — raw value at the current cycle
   - `s_roll_mean` — mean of the last `window_size` rows (inclusive of current)
   - `s_roll_std` — sample std (`ddof=1`) of the last `window_size` rows
4. **`cycle`** — the current cycle number (observed engine age; available at inference).
5. **Row eligibility:** a row gets features only if `cycle ≥ window_size` (full window). Training drops earlier rows (≈ 29 × 100 = 2,900 of 20,631); inference requires ≥ `window_size` records. Train and serve therefore see identical feature semantics.
6. **Feature order** = `["cycle"] + [f for s in required_sensors for f in (s, f"{s}_roll_mean", f"{s}_roll_std")]`, stored in metadata as `feature_names` and enforced at inference.

All of this is implemented once in `predmaint/features/build.py` and reused by training, evaluation, and the API.

## 5. Training procedure (`scripts/train.py`)

1. Verify FD001 checksums (04 §3) → load → validate.
2. Labels: `RUL = max_cycle − cycle`, clipped at `rul_cap`.
3. Sensor selection on train; build features; drop ineligible rows.
4. CV: `GroupKFold(n_splits=5)` with `groups=unit`, for each candidate → per-fold MAE/RMSE, mean and std.
5. Fit each candidate on all eligible training rows.
6. Test: build features over each test unit's full observed history, take its last row (all 100 units have ≥ 31 cycles ≥ `window_size`), predict, clamp at 0, compare with `RUL_FD001`.
7. Selection (§8) → save artifact + metadata (§9). Print a metrics table to stdout.

## 6. Reproducibility controls

- `random_seed = 42` in config, passed to every estimator with a `random_state`.
- GroupKFold is deterministic (no shuffling).
- Data identity: SHA-256 of FD001 files recorded in metadata; training aborts on mismatch.
- Code identity: `git rev-parse HEAD` recorded (`"unknown"` if unavailable, e.g. no commits / no `.git`).
- Environment: Python and key library versions recorded.
- NFR-001: two runs must give identical test metrics (to 1e-9).

## 7. Metrics

For predictions `ŷ` and truths `y` over *n* units, `d = ŷ − y`:

- **MAE** = mean(|d|); **RMSE** = sqrt(mean(d²)) — primary, used for selection.
- **NASA score** (secondary and informational only — **never used for model selection**, §8; lower is better): `s = Σ (exp(−d/13) − 1)` for `d < 0` (early), `Σ (exp(d/10) − 1)` for `d ≥ 0` (late).

**Evidence for the NASA score (verified from the paper bundled in the archive, eq. 11 and §VII):** `d = estimated RUL − true RUL`; "late predictions were more heavily penalized than early predictions"; `a1 = 10`, `a2 = 13`. The printed equation pairs `a1` with `d < 0` and `a2` with `d ≥ 0`, which contradicts the text and Figure 9 (steeper for positive error). We follow the **stated intent** (late divisor 10, early divisor 13) — ADR-009. Unit tests pin known values (`d = 10 → e − 1`, `d = −13 → e − 1`).

Reported metric sets:

- `cv`: per-fold and mean ± std MAE/RMSE (clipped training labels, eligible rows).
- `test`: MAE, RMSE, NASA score vs **unclipped** `RUL_FD001` (primary).
- `test_clipped_truth`: MAE/RMSE vs `min(RUL_FD001, rul_cap)` — for comparability with literature that clips test truth; not used for selection.

No metric value is assumed or targeted in advance; actual values are recorded in [14](14_PROJECT_STATUS.md) after M3.

## 8. Model selection rule (ADR-010)

The primary model is selected **only if** its `test.mae < baseline test.mae` **and** `test.rmse < baseline test.rmse`. Otherwise the baseline is shipped. The NASA score and CV metrics play no part in this decision. The decision and both candidates' metrics are written to metadata (`selected`, `selection_reason`).

## 9. Artifact format and metadata schema (ADR-011)

- `models/model.joblib` — the fitted sklearn `Pipeline` of the selected candidate (`joblib.dump`).
- `models/model_metadata.json`:

```json
{
  "schema_version": 1,
  "model_version": "hgb-20261110T120000Z",
  "model_name": "hist_gradient_boosting",
  "trained_at": "2026-11-10T12:00:00Z",
  "git_sha": "<40-hex or 'unknown'>",
  "python_version": "3.13.3",
  "library_versions": {"scikit-learn": "1.9.1", "pandas": "3.1.1", "numpy": "2.5.4", "joblib": "1.6.0"},
  "data": {
    "subset": "FD001",
    "train_sha256": "963b5e22…",
    "test_sha256": "3cda7109…",
    "rul_sha256": "a19c8ec9…"
  },
  "config": {"rul_cap": 125, "window_size": 30, "random_seed": 42, "cv_folds": 5},
  "required_sensors": ["s_2", "…"],
  "dropped_columns": ["setting_1", "setting_2", "setting_3", "…"],
  "feature_names": ["cycle", "s_2", "s_2_roll_mean", "s_2_roll_std", "…"],
  "training_ranges": {"s_2": {"min": 0.0, "max": 0.0}},
  "health_bands": {"critical_below": 30, "warning_below": 80},
  "metrics": {
    "baseline": {"cv": {"mae_mean": 0.0, "mae_std": 0.0, "rmse_mean": 0.0, "rmse_std": 0.0, "folds": []},
                 "test": {"mae": 0.0, "rmse": 0.0, "nasa_score": 0.0},
                 "test_clipped_truth": {"mae": 0.0, "rmse": 0.0}},
    "primary":  {"cv": {}, "test": {}, "test_clipped_truth": {}}
  },
  "selected": "primary",
  "selection_reason": "primary improved test MAE and RMSE over baseline"
}
```

Numbers and the `s_2` entry above illustrate **shape only**; real values are written by training. `model_version` = `{"ridge"|"hgb"}-{trained_at as YYYYMMDDTHHMMSSZ}`. `training_ranges` covers each required sensor's raw min/max on train.

## 10. Inference contract (`predmaint/models/predict.py`)

```
load_model(model_dir: Path) -> LoadedModel            # pipeline + metadata; raises ModelUnavailableError
predict_rul(history: pd.DataFrame, model: LoadedModel) -> PredictionResult
health_band(rul: float, bands: dict) -> Literal["CRITICAL", "WARNING", "HEALTHY"]
```

- `history`: one engine; columns `cycle` + every name in `required_sensors` (extra columns ignored); ≥ `window_size` rows; cycles consecutive (+1) after sorting.
- Raises `InsufficientCyclesError`, `MissingSensorsError`, `NonConsecutiveCyclesError` (all subclass `InputContractError`) — mapped 1:1 to API error codes (06 §6).
- `PredictionResult`: `predicted_rul` (float, clamped at ≥ 0, ADR-018), `health_band`, `last_cycle`, `cycles_used` (= `window_size`), `warnings` (list of out-of-training-range messages over the used window).
- Health band (ADR-017): `CRITICAL` if RUL < 30; `WARNING` if 30 ≤ RUL < 80; else `HEALTHY`.
- The API and the offline test evaluation call this same function; the contract test asserts identical outputs.

## 11. Limitations

- FD001 only: one operating condition and one fault mode; the model is not valid for other conditions.
- Simulated data; no claim about real engines.
- Predictions are effectively bounded near `rul_cap` (125); true test RULs up to 145 will be under-predicted by design.
- Selecting between two pre-specified candidates on the test set introduces a small selection bias; CV metrics are reported alongside to expose it.
- Rolling features need ≥ 30 cycles of history; younger engines cannot be scored.
- No uncertainty estimate; health bands are illustrative thresholds, not engineering limits.
- joblib artifacts are tied to the scikit-learn version (mismatch warning, 02 §7).
