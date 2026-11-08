# 05 — ML Design

## 1. Objective

Regress **RUL in cycles** for an FD001 engine from its observed sensor history, evaluated at the last observed cycle of each official test unit. Primary metrics: MAE, RMSE (lower is better). Classification accuracy is not used.

## 2. Baseline model (ADR-006)

`Pipeline([("select", ColumnTransformer([("features", "passthrough", feature_names)], remainder="drop")), ("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])`. Interpretable, fast, and sets the bar the primary model must beat. The leading `select` step (ADR-023) stores the training-time feature list — and therefore the kept sensors — inside the artifact and rejects inputs missing any of those columns; the scaler is fitted inside the pipeline, i.e. on training rows only (per CV fold, and on the full training set for the final fit).

## 3. Primary model (ADR-007, ADR-026)

`Pipeline([("select", <same ColumnTransformer>), ("model", HistGradientBoostingRegressor(early_stopping=False, random_state=42, **params))])` on the identical 43 features, training-row filter and seed as the baseline. No scaler: tree splits are invariant to monotonic feature scaling. Internal early stopping stays off because its validation split is row-wise and would mix cycles of the same engine; the iteration budget is a grid parameter instead.

**One bounded search (ADR-026), no second round:** 12 configurations = (learning_rate, max_iter) ∈ {(0.1, 100), (0.05, 200), (0.03, 400)} × max_leaf_nodes ∈ {15, 31} × l2_regularization ∈ {0.0, 1.0}. Each one is scored by the same 5-fold GroupKFold CV; the winner has the lowest CV RMSE mean (tie-break CV MAE mean). The test set is never consulted. The grid, every configuration's CV result and the winner are stored in metadata/report `search`. Results are in §7b.

## 4. Feature strategy (ADR-005)

Config defaults: `WINDOW_SIZE = 30`, `RUL_CAP = 125`. Both constants are defined **once**, in `predmaint/config.py`, and imported by training (`models/train.py`), feature building (`features/build.py`), and inference (`models/predict.py` / API); no module re-declares them. Training writes the values used into metadata (`config.window_size`, `config.rul_cap`), and inference uses the metadata values so a served model always matches how it was trained. Lower-case `window_size` / `rul_cap` below refer to these values.

1. **Candidate columns:** `s_1 … s_21`. `setting_1..3` are excluded (FD001 has one operating condition per readme).
2. **Sensor drop rule (train only):** drop a sensor if it has **≤ 2 unique values** in `train_FD001`. The kept list is computed in M2 on real data, printed with each sensor's std and unique count, and recorded in [04](04_DATASET_SPECIFICATION.md) and metadata (`required_sensors`, `dropped_columns`). If inspection shows the rule misclassifies a sensor, the rule is amended through [15](15_CHANGE_CONTROL.md) — not silently. **Confirmed on real data in M2** (04 §5a): drops `s_1, s_5, s_6, s_10, s_16, s_18, s_19`, keeps 14 sensors → 43 features.
3. **Per kept sensor `s`, per row (computed within each unit, trailing only):**
   - `s` — raw value at the current cycle
   - `s_roll_mean` — mean of the last `window_size` rows (inclusive of current)
   - `s_roll_std` — sample std (`ddof=1`) of the last `window_size` rows

   Semantics equal pandas `rolling(window_size, min_periods=window_size)` per unit. The implementation computes each window independently (NumPy `sliding_window_view`), not with a running accumulator, so a cycle's feature row is bit-identical whether or not later cycles — or cycles older than the window — are present (tested).
4. **`cycle`** — the current cycle number (observed engine age; available at inference).
5. **Row eligibility:** a row gets features only if a full trailing window of `window_size` rows of the same unit exists (`min_periods = window_size`); for histories starting at cycle 1 this is exactly `cycle ≥ window_size`. Training drops earlier rows (≈ 29 × 100 = 2,900 of 20,631); inference requires ≥ `window_size` records. Train and serve therefore see identical feature semantics.
6. **Feature order** = `["cycle"] + [f for s in required_sensors for f in (s, f"{s}_roll_mean", f"{s}_roll_std")]`, stored in metadata as `feature_names` and enforced at inference.

All of this is implemented once in `predmaint/features/build.py` and reused by training, evaluation, and the API.

## 5. Training procedure (`scripts/train.py --model <name>`)

1. Verify FD001 checksums (04 §3) → load → validate.
2. Labels: `RUL = max_cycle − cycle`, clipped at `rul_cap`.
3. Sensor selection on train; build features; drop ineligible rows.
4. CV: `GroupKFold(n_splits=5)` with `groups=unit`, for each candidate → per-fold MAE/RMSE, mean and std.
5. Fit each candidate on all eligible training rows.
6. Test: build features over each test unit's full observed history, take its last row (all 100 units have ≥ 31 cycles ≥ `window_size`), predict, clamp at 0, compare with `RUL_FD001`.
7. Save the candidate's artifact, metadata and committed report (§9); print the dropped-sensor list, per-fold CV, and test metrics. Selection (§8) and the served artifact follow in M3.

## 6. Reproducibility controls

- `random_seed = 42` in config, passed to every estimator with a `random_state`.
- GroupKFold is deterministic (no shuffling).
- Data identity: SHA-256 of FD001 files recorded in metadata; training aborts on mismatch.
- Code identity: `git rev-parse HEAD` recorded (`"unknown"` if unavailable, e.g. no commits / no `.git`).
- Environment: Python and key library versions recorded.
- NFR-001: two runs must give identical test metrics (to 1e-9).

## 7. Metrics

For predictions `ŷ` and truths `y` over *n* units, `d = ŷ − y`:

- **MAE** = mean(|d|); **RMSE** = sqrt(mean(d²)) — primary metrics. Model selection uses **CV** RMSE (tie-break CV MAE) only (ADR-024, §8).
- **NASA score** (secondary and informational only — **never used for model selection**, §8; lower is better): `s = Σ (exp(−d/13) − 1)` for `d < 0` (early), `Σ (exp(d/10) − 1)` for `d ≥ 0` (late).

**Evidence for the NASA score (verified from the paper bundled in the archive, eq. 11 and §VII):** `d = estimated RUL − true RUL`; "late predictions were more heavily penalized than early predictions"; `a1 = 10`, `a2 = 13`. The printed equation pairs `a1` with `d < 0` and `a2` with `d ≥ 0`, which contradicts the text and Figure 9 (steeper for positive error). We follow the **stated intent** (late divisor 10, early divisor 13) — ADR-009. Unit tests pin known values (`d = 10 → e − 1`, `d = −13 → e − 1`).

Reported metric sets:

- `cv`: per-fold and mean ± std MAE/RMSE (clipped training labels, eligible rows).
- `test`: MAE, RMSE, NASA score vs **unclipped** `RUL_FD001` (primary).
- `test_clipped_truth`: MAE/RMSE vs `min(RUL_FD001, rul_cap)` — for comparability with literature that clips test truth; not used for selection.

All predictions (CV folds, test, API) pass through the same clamp to `[0, rul_cap]` (ADR-027).

No metric value is assumed or targeted in advance; actual values are recorded in [14](14_PROJECT_STATUS.md).

### 7a. Measured results — 2026-11-08 (Milestone 3)

Provenance: code = commit `f4c86df` plus the uncommitted Milestone 3 working tree (`git_dirty: true`), Python 3.13.3, scikit-learn 1.9.1, data SHA-256 per 04 §3. Sources: [reports/metrics_baseline.json](../reports/metrics_baseline.json) (`ridge-20261108T150138Z`, trained 15:01:38Z), [reports/metrics_primary.json](../reports/metrics_primary.json) (`hgb-20261108T150441Z`, trained 15:04:41Z), [reports/model_comparison.md](../reports/model_comparison.md).

| Model | CV RMSE (mean ± std) | CV MAE (mean ± std) | **Test RMSE** | **Test MAE** | NASA score | Test RMSE / MAE, truth clipped at 125 |
|---|---|---|---|---|---|---|
| Baseline — Ridge(alpha=1.0) | 19.152 ± 1.182 | 15.112 ± 1.156 | 19.144 | 15.347 | 590.6 | 17.845 / 14.277 |
| **Primary — HGB (selected)** | **16.559 ± 0.695** | **11.843 ± 0.676** | **16.610** | **12.567** | 441.5 | 15.333 / 11.497 |

CV: 5-fold GroupKFold by engine, capped labels, std ddof=1. Test: official FD001 test set, last observed cycle of each of 100 engines, unclipped truth. Per-fold CV RMSE — baseline: 20.810, 19.111, 19.309, 17.476, 19.054; primary: 17.479, 16.769, 15.898, 15.826, 16.823. Training rows: 17,731 of 20,631.

**Baseline change since M2 (disclosed):** the M2 run (`ridge-20261108T145035Z`) clamped predictions at 0 only and reported CV RMSE 19.287 ± 1.181, test RMSE 19.205, test MAE 15.403, NASA 595.2. The shared clamp is now `[0, 125]` (ADR-027) and also applies inside CV. One test prediction was 130.58, which accounts for the small improvement. The model itself is unchanged.

**Training cost (NFR-002) on smaller machines:** the primary run measured 80 s wall but ≈ 582 CPU-seconds (≈ 9.7 CPU-minutes: 434 s user + 148 s sys) on an 8-core Apple M1 (4 performance + 4 efficiency cores, `sysctl hw.ncpu` = 8), because HistGradientBoosting uses all cores via OpenMP. **Unmeasured estimate:** wall time should scale roughly with that CPU budget divided by available cores (on the order of 3–5 minutes on 2–4 cores), so NFR-002's < 5-minute wall-time target is not guaranteed on small machines. The baseline is negligible (≈ 1.4 s).

Reproducibility (NFR-001): retraining the primary gave identical metrics and search results and a byte-identical `primary.joblib`. Primary training wall time 76.6–80 s on the Apple-Silicon dev machine (`time`: 434 s user + 148 s sys CPU, i.e. ≈ 9.7 CPU-minutes across cores, because HGB is multi-threaded).

### 7b. Primary search results (ADR-026)

| learning_rate | max_iter | max_leaf_nodes | l2_regularization | CV RMSE (mean ± std) | CV MAE |
|---|---|---|---|---|---|
| 0.1 | 100 | 15 | 0.0 | 16.788 ± 0.614 | 11.962 |
| 0.1 | 100 | 15 | 1.0 | 16.717 ± 0.679 | 11.915 |
| 0.1 | 100 | 31 | 0.0 | 16.740 ± 0.636 | 11.808 |
| 0.1 | 100 | 31 | 1.0 | 16.717 ± 0.653 | 11.826 |
| **0.05** | **200** | **15** | **0.0** | **16.559 ± 0.695** | **11.843** |
| 0.05 | 200 | 15 | 1.0 | 16.688 ± 0.686 | 11.958 |
| 0.05 | 200 | 31 | 0.0 | 16.828 ± 0.679 | 11.840 |
| 0.05 | 200 | 31 | 1.0 | 16.726 ± 0.646 | 11.789 |
| 0.03 | 400 | 15 | 0.0 | 16.642 ± 0.634 | 11.849 |
| 0.03 | 400 | 15 | 1.0 | 16.629 ± 0.652 | 11.843 |
| 0.03 | 400 | 31 | 0.0 | 16.683 ± 0.611 | 11.742 |
| 0.03 | 400 | 31 | 1.0 | 16.798 ± 0.634 | 11.815 |

The spread across all 12 configurations is small (CV RMSE 16.56–16.83, within one fold-std), so the choice of configuration matters far less than the move from Ridge to boosted trees.

### 7c. Selection outcome

**Primary (HGB) was promoted** as `primary-20261108-f4c86df`: its CV RMSE (16.559) is lower than the baseline's (19.152). The official test ranking agrees (16.610 vs 19.144), so no disagreement was flagged.

## 8. Model selection rule (ADR-024, supersedes ADR-010)

`python scripts/train.py --select` reads both candidates' metadata. It requires both to share the same data hashes, config, feature names and kept sensors, then picks the candidate with the **lowest 5-fold GroupKFold CV RMSE mean**. Tie-breaks: CV MAE mean, then the simpler model (baseline). Official test metrics — unclipped truth primary, clipped secondary — and the NASA score are reported for both but **never drive the choice**. If the test ranking disagrees with the CV ranking, the selection statement says so explicitly and the CV choice stands.

## 9. Artifact format and metadata schema (ADR-011, ADR-023)

**Per-candidate artifacts:** `scripts/train.py --model <name>` writes

- `models/<name>.joblib` — the fitted sklearn `Pipeline` (select → [scaler] → estimator), gitignored;
- `models/<name>_metadata.json` — gitignored; fields as in the example below, minus the promotion fields (`candidate_model_version`, `selected_by`, `promoted_at`, `comparison`); `model_version` = `{"ridge"|"hgb"}-{trained_at as YYYYMMDDTHHMMSSZ}`;
- `reports/metrics_<name>.json` — **committed**; a subset (identity, provenance, config, sensors, hyperparameters, search, metrics).

**Served artifact:** `scripts/train.py --select` writes the files the API loads:

- `models/model.joblib` — a byte copy of the winning candidate's joblib;
- `models/model_metadata.json` — the winner's metadata plus promotion fields;
- `reports/model_comparison.json` and `reports/model_comparison.md` — committed comparison table and selection statement.

```json
{
  "schema_version": 1,
  "model_name": "primary",
  "estimator": "hgb",
  "model_version": "primary-20261108-f4c86df",
  "candidate_model_version": "hgb-20261108T150441Z",
  "selected_by": "cv_rmse",
  "trained_at": "2026-11-08T15:04:41Z",
  "promoted_at": "2026-11-08T15:04:43Z",
  "git_sha": "<40-hex or 'unknown'>",
  "git_dirty": true,
  "python_version": "3.13.3",
  "library_versions": {"scikit-learn": "1.9.1", "pandas": "3.1.1", "numpy": "2.5.4", "joblib": "1.6.0"},
  "data": {"subset": "FD001", "train_sha256": "963b5e22…", "test_sha256": "3cda7109…", "rul_sha256": "a19c8ec9…"},
  "config": {"rul_cap": 125, "window_size": 30, "random_seed": 42, "cv_folds": 5},
  "required_sensors": ["s_2", "…"],
  "dropped_columns": ["setting_1", "setting_2", "setting_3", "s_1", "…"],
  "feature_names": ["cycle", "s_2", "s_2_roll_mean", "s_2_roll_std", "…"],
  "training_ranges": {"s_2": {"min": 641.21, "max": 644.53}},
  "health_bands": {"critical_below": 20, "warning_below": 60},
  "hyperparameters": {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 15, "l2_regularization": 0.0},
  "search": {"scored_by": "cv_rmse_mean (tie-break cv_mae_mean)", "n_configurations": 12, "grid": ["…"], "results": ["…"], "selected_params": {"…": "…"}},
  "metrics": {"cv": {"mae_mean": 0.0, "mae_std": 0.0, "rmse_mean": 0.0, "rmse_std": 0.0, "folds": ["…"]},
              "test": {"mae": 0.0, "rmse": 0.0, "nasa_score": 0.0},
              "test_clipped_truth": {"mae": 0.0, "rmse": 0.0}, "n_train_rows": 17731, "n_test_units": 100},
  "comparison": {"selected": "primary", "selected_by": "cv_rmse", "tie_break": "cv_mae",
                 "cv_ranking": ["primary", "baseline"], "test_ranking": ["primary", "baseline"],
                 "rankings_agree": true, "reason": "Selected primary because …",
                 "candidates": {"baseline": {"model_version": "…", "cv": {}, "test": {}, "test_clipped_truth": {}}, "primary": {}}}
}
```

Metric numbers and `"…"` illustrate **shape only**; the real values are in §7a and the files themselves. Served `model_version` = `<name>-<YYYYMMDD of training>-<short git SHA>`.

## 10. Inference contract (`predmaint/models/predict.py`)

```
load_model(model_dir: Path | None = None) -> LoadedModel     # default $MODEL_DIR; cached per directory
clear_model_cache() -> None
predict_rul(cycles: pd.DataFrame | Sequence[Mapping], model: LoadedModel,
            intervals: IntervalModel | None = None) -> PredictionResult
load_interval_model(path: Path | None = None) -> IntervalModel   # default: next to $MODEL_PATH (M8)
order_bounds(point, lower, upper, rul_cap, conformal_q=None) -> (lower, upper, n_reordered)  # Q applied first (M10)
load_anomaly_model(path: Path | None = None) -> AnomalyModel     # default: next to $MODEL_PATH (M9)
score_anomaly(cycles, model: LoadedModel, anomaly: AnomalyModel,
              include_series: bool = False) -> AnomalyResult    # M9, ADR-035
health_status(rul: float) -> Literal["critical", "warning", "healthy"]
predict_from_features(pipeline, features, rul_cap) -> np.ndarray   # shared clamp, also used offline
```

- `load_model` loads `model.joblib` + `model_metadata.json` once per directory. Missing or unloadable files (corrupt pickle/JSON, schema mismatch, missing metadata keys) raise `ModelNotAvailableError`. Failures are not cached, so a model promoted afterwards is picked up.
- `cycles`: one engine, records in **chronological order**. Columns `cycle` + every name in `required_sensors`; extra columns are ignored.
- Checks, in order: ≥ `window_size` records (`InsufficientHistoryError`); `cycle` and required sensors present and non-null (`MissingColumnsError`, with `.columns`); values finite and cycles positive integers (`InvalidValuesError`); cycles strictly increasing by exactly 1 as given (`NonConsecutiveCyclesError` — covers out-of-order, duplicates, gaps). All except `ModelNotAvailableError` subclass `InputContractError(ValueError)`. HTTP mapping is in 06 §6.
- Only the **last `window_size` records** are used. They pass through the shared `features.build.build_features` to produce one feature row; extra older history cannot change the result (tested).
- `PredictionResult`: `predicted_rul` (float, clamped to `[0, rul_cap]`, ADR-027), `health_status` (ADR-025), `cycles_received`, `cycles_used` (= `window_size`), `last_cycle`, `model_version`, `warnings` (`OUT_OF_TRAINING_RANGE` messages for required sensors in the used window).
- **Intervals (M8, ADR-034):** with a matching `IntervalModel` (`for_model_version` == served `model_version` and identical `feature_names`), `PredictionResult` also carries `rul_lower`, `rul_upper` (clamped to [0, `rul_cap`], ordered around the point) and `interval_level` (0.8). Since M10 the raw quantile predictions are first widened by the metadata's `conformal_q` (CQR, ADR-037) inside `order_bounds`, and `interval_calibrated` reports whether Q was applied (metadata without Q → v1.1 bounds, `false`). Without an interval model, or on a mismatch (logged warning), these fields are `None`. The point prediction is byte-identical in all cases (tested).
- **Anomaly score (M9, ADR-035):** `score_anomaly` runs the **same validation function** as `predict_rul` (identical exceptions and messages, tested), builds features over the whole validated request and scores every full-window row with the detector. `AnomalyResult`: `anomaly_score` (Mahalanobis distance of the last row), `normalized_score` (= score / threshold), `threshold`, `is_anomalous` (normalised > 1), `last_cycle`, `cycles_received`, `model_version`, and optional `series` (`cycles_received − window_size + 1` entries). A detector whose `for_model_version` differs from the served model, or whose features the served model does not produce, raises `AnomalyNotAvailableError`. The point model and interval code paths are not touched.
- Health status (ADR-025, `config.HEALTH_BANDS`): `critical` if RUL < 20; `warning` if 20 ≤ RUL < 60; `healthy` if RUL ≥ 60.
- The offline test evaluation and `predict_rul` produce bit-identical predictions (contract test; real-data test recomputes the served model's test RMSE through `predict_rul`).

### 10a. Prediction intervals (M8, ADR-034) — measured 2026-11-08T18:55:47Z

Two `HistGradientBoostingRegressor(loss="quantile")` models (q = 0.1, 0.9 → nominal 80 %) with the served model's features, train-row filter, seed and hyperparameters (`learning_rate=0.05, max_iter=200, max_leaf_nodes=15, l2_regularization=0.0`), trained by `scripts/train.py --model intervals` after `--select`, on code commit `8d893ff` plus the uncommitted M8 working tree. There is no search, and the point model is untouched (SHA-256 unchanged).

| | Coverage (unclipped) | Coverage (clipped) | Mean width | Reordered |
|---|---|---|---|---|
| 5-fold GroupKFold out-of-fold (17,731 rows) | 0.509 | 0.737 | 37.2 | 1,023 |
| Official test, last cycle (100 engines) | 0.740 | 0.820 | 42.1 | 8 |

Coverage for true RUL > 125 is 0 against unclipped truth by construction (cap); bucket tables and interpretation in [16 §9](16_EVALUATION_REPORT.md). Source: `reports/metrics_intervals.json`.

**Conformal calibration (M10, ADR-037), measured 2026-11-08T20:37:53Z.** Split-conformal CQR on the same out-of-fold predictions, α = 0.2, target = capped RUL. E = max(lo − y, y − hi) on raw bounds; Q = ⌈(n+1)(1−α)⌉-th smallest E. Engines split at random (seed 42) into 50 calibration / 50 evaluation engines. No model was retrained; `model_intervals.joblib` is byte-identical.

| | Q | Coverage (clipped) | Coverage (unclipped) | Mean width |
|---|---|---|---|---|
| Evaluation half (8,921 rows), before | — | 0.734 | 0.504 | 37.2 |
| Evaluation half, after (**FR-020**) | Q_eval 1.320 | **0.777** | 0.536 | 39.2 |
| Official test, before → after | Q_final 1.742 (shipped) | 0.820 → 0.850 | 0.740 → 0.750 | 42.1 → 44.7 |

FR-020 (0.80 ± 0.03) is **met at the lower edge**. Rows of one engine are correlated, so the conformal guarantee is approximate (16 §9d).

### 10b. Anomaly detection (M9, ADR-035) — measured 2026-11-08T19:33:06Z

`scripts/train.py --model anomaly` (after `--select`; refuses without `models/model_metadata.json`; ~2 s wall) fits `StandardScaler` → `LedoitWolf` on the **healthy reference**: training rows with a full 30-cycle window and true (uncapped) RUL ≥ 125, i.e. 5,329 rows from 90 of 100 engines (an engine needs ≥ 155 cycles). Features: the 14 kept sensors' `*_roll_mean` columns from `features.build`. Score = Mahalanobis distance; threshold = 99th percentile of **out-of-fold** healthy-row scores (5-fold GroupKFold by engine) = 5.4036 (in-sample would be 5.1510). The IsolationForest fallback (only if the median train-OOF trend were < 0.5) was not triggered.

| Measure | Value |
|---|---|
| Median per-engine Spearman(score, cycle), train OOF / test | **0.680** (target > 0.7, **not met**) / 0.213 |
| False-alarm rate on healthy rows (train OOF) | 0.0101 |
| First sustained detection (3 cycles), train OOF | median 0.758 of life, true RUL 49.0; never flagged 0.020 |
| Test engines flagged at last cycle | 0.250 overall; 0.750 for true RUL ≤ 20, 0.000 for > 125 |
| Spearman(normalised score, predicted RUL), test last cycle | −0.613 |

Full tables and what the evaluation does and does not prove: [16 §10](16_EVALUATION_REPORT.md). Source: `reports/metrics_anomaly.json`.

## 11. Limitations

- FD001 only: one operating condition and one fault mode; the model is not valid for other conditions.
- Simulated data; no claim about real engines.
- Predictions are clamped to `rul_cap` (125); true test RULs up to 145 are under-predicted by design.
- The primary configuration was chosen by CV on the training set only, and selection is by CV, so the official test set stays a held-out report. The 12-configuration search still slightly favours the primary's CV estimate (best-of-12); its test RMSE (16.610) is close to its CV RMSE (16.559).
- Rolling features need ≥ 30 cycles of history; younger engines cannot be scored.
- Uncertainty: the 80 % intervals are conformally calibrated since M10 (held-out-engine coverage 0.777 against the capped target, still slightly below nominal; per-bucket coverage uneven, 21–60 at 0.687) and cannot cover true RUL > 125; health bands are illustrative thresholds, not engineering limits.
- The M9 anomaly score is unsupervised: it measures distance from a population healthy reference, has no fault labels to validate against, and missed its trend target (0.680 ≤ 0.7).
- joblib artifacts are tied to the scikit-learn version (mismatch warning, 02 §7).
