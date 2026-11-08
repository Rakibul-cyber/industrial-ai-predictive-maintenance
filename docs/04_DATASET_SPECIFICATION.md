# 04 — Dataset Specification

Dataset: **NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set**, subset **FD001**.
Reference (from the archive's `readme.txt`): A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08, Denver CO, Oct 2008.

## 1. Source and download (Verified 2026-11-08)

| Item | Value | Status |
|---|---|---|
| Landing page | https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ | HTTP 200; page contains `href` to the archive below. The legacy `https://ti.arc.nasa.gov/c/6/` 301-redirects here. |
| **Archive URL (used)** | `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip` | HTTP 200, `Content-Length: 12429152`, `Last-Modified: Sun, 18 Sep 2022 03:42:08 GMT`. Downloaded. |
| Catalog entry | https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data | HTTP 200; links `https://data.nasa.gov/docs/legacy/CMAPSSData.zip`. |
| Alternate official copy | `https://data.nasa.gov/docs/legacy/CMAPSSData.zip` | **Failed**: 302 → `403 Forbidden`. Not usable. |
| Old Socrata link | `https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip` | **Failed**: 404. |
| Third-party mirrors (e.g. Kaggle) | Not checked. | **UNVERIFIED** — fallback only; would require its own licence review and checksum comparison against §3 (R1). |

**Procedure (implemented in M1 as `scripts/download_data.py`):**

1. Download the archive URL to `data/raw/cmapss_outer.zip`; verify SHA-256.
2. The archive is **nested**: it contains one folder `6. Turbofan Engine Degradation Simulation Data Set/` holding `CMAPSSData.zip`. Extract only that member; verify its SHA-256.
3. Extract the inner zip to `data/raw/CMAPSSData/`, allowing only the 14 expected member names (zip-slip guard); verify FD001 file checksums.

## 2. Archive contents (Verified)

Inner `CMAPSSData.zip` (14 files): `readme.txt`, `Damage Propagation Modeling.pdf`, and `train_FD00{1..4}.txt`, `test_FD00{1..4}.txt`, `RUL_FD00{1..4}.txt`. Only FD001 is used (ADR-001).

## 3. Checksums (SHA-256, Verified)

| File | SHA-256 |
|---|---|
| `cmapss_outer.zip` (S3 archive) | `c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2` |
| `CMAPSSData.zip` (inner) | `74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f` |
| `train_FD001.txt` | `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8` |
| `test_FD001.txt` | `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851` |
| `RUL_FD001.txt` | `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca` |

`model_metadata.json` records the three FD001 hashes (05 §9).

## 4. Observed file format (Verified by inspection)

- ASCII, LF line endings, **no header**.
- Fields separated by single spaces; **every train/test line ends with two trailing spaces**. Consequence: `pd.read_csv(sep=" ")` produces **28** columns (cols 26 and 27 all-NaN). The loader **must** use `sep=r"\s+"`, which yields exactly 26 columns.
- `readme.txt` says "26 columns" but its list ends with "sensor measurement 26" — internally inconsistent. The data itself has 26 columns = 2 ids + 3 settings + **21** sensors, matching the paper (Table 2 lists 21 sensor outputs).

| Col # | Name used in code | Meaning (readme) | Observed dtype |
|---|---|---|---|
| 1 | `unit` | unit number | int |
| 2 | `cycle` | time, in cycles | int |
| 3–5 | `setting_1`, `setting_2`, `setting_3` | operational settings | float |
| 6–26 | `s_1` … `s_21` | sensor measurements 1–21 | float (`s_17`, `s_18` parse as int) |

Sensor names are kept **generic** (ADR-020). The paper's Table 2 lists 21 physical sensors (T2, T24, …, W32), but neither the readme nor the paper states that column order equals table order → mapping is **UNVERIFIED** and not used.

## 5. Observed FD001 statistics (Verified, pandas 3.1.1)

| Property | train_FD001 | test_FD001 | RUL_FD001 |
|---|---|---|---|
| Rows × cols | 20631 × 26 | 13096 × 26 | 100 × 1 |
| Nulls | 0 | 0 | 0 |
| Units | 100 (ids 1–100, contiguous) | 100 (ids 1–100, contiguous) | one value per test unit (row *i* ↔ unit *i*) |
| Cycles start at 1, strictly +1 per unit | yes | yes | — |
| Rows sorted by (unit, cycle) | yes | yes | — |
| Cycles per unit (min / median / max) | 128 / 199 / 362 | **31** / 133.5 / 303 | values min 7, max 145, mean 75.52 |

Readme (Verified): FD001 = 100 train / 100 test trajectories, one condition (sea level), one fault mode (HPC degradation). Train trajectories run to failure; test trajectories stop before failure; `RUL_FD001` gives the true remaining cycles **after the last observed test cycle**.

**Design consequence:** the shortest test unit has 31 cycles, so `window_size = 30` lets every test unit be evaluated with a full window through the same API contract (ADR-005).

## 6. Data quality checks (implemented in `data/validate.py`, M1)

1. Train/test have exactly 26 columns; RUL has exactly 1.
2. No nulls; all values finite.
3. `unit`, `cycle` are integers ≥ 1.
4. Per unit: cycles start at 1 and increase by exactly 1.
5. `len(RUL) == test.unit.nunique()` and test unit ids are `1..n`.
6. RUL values are integers ≥ 0.
7. FD001-specific (real-data tests): shapes in §5; 100 units each.

Value-range checks: physical ranges are not documented by the source, so no hard-coded ranges. Instead, per-sensor **training min/max** are stored in metadata (05 §9) and the API emits `OUT_OF_TRAINING_RANGE` warnings (06).

## 7. Target definition

- **Train:** `RUL(unit, cycle) = max_cycle(unit) − cycle` (0 at the failure cycle), then `min(RUL, rul_cap)` with `rul_cap = 125` (ADR-004).
- **Test:** truth is `RUL_FD001[i]` at the **last observed cycle** of test unit *i*; not clipped for the primary metrics (05 §7).

## 8. Train / validation / test strategy

- Model selection & diagnostics: 5-fold **GroupKFold by `unit`** on `train_FD001` (no unit in both train and validation folds).
- Final evaluation: official `test_FD001` + `RUL_FD001`, one prediction per unit at its last cycle (100 predictions).
- The test set is never used for fitting, feature selection, or hyperparameter choice. It is used once per candidate for the ADR-010 selection between exactly two pre-specified models (limitation noted in 05 §11).

## 9. Leakage risks and controls

| Risk | Control |
|---|---|
| Same engine in train and validation fold | GroupKFold by `unit`; unit test asserts disjoint groups. |
| Future cycles in rolling features | Trailing windows only (`groupby(unit).rolling(window)`); causality unit test. |
| Scaler fitted on validation/test | Scaler lives inside the sklearn `Pipeline`, fitted per fold / on train only. |
| Sensor selection using test data | Selection computed on `train_FD001` only. |
| Using the test unit's (unknown) end of life | Test features use only observed rows; no test row is labelled from test data. |
| HGB internal early-stopping split mixes engines row-wise | `early_stopping=False` (ADR-007). |

## 10. Preprocessing rules

- Parse with `sep=r"\s+"`, `header=None`, explicit names (§4); cast `unit`, `cycle` to int.
- Exclude `setting_1..3` from features (single operating condition; ADR-005).
- Sensor drop rule and rolling features: [05](05_ML_DESIGN.md) §4.
- No imputation (no nulls exist; validation rejects nulls).

## 11. Licensing and redistribution

- data.nasa.gov catalog page states: **"License not specified"** (observed in page HTML).
- Usage terms on the PCoE repository page: **UNVERIFIED** (not inspected beyond the link). Resolved in M1 (task M1-T8).
- Policy regardless of outcome: raw data is **never committed** (`data/*` gitignored); users download it from NASA via the script; README cites the Saxena et al. (2008) reference and NASA PCoE as source.
- Test fixtures are **synthetic** (generated in `tests/conftest.py`), not excerpts of NASA files.

## 12. Verified vs UNVERIFIED summary

**Verified:** archive URL + size + checksums; nested-zip structure; 14 member files; FD001 shapes, nulls, unit ids, cycle monotonicity, test min length 31, RUL range; whitespace/trailing-space format; readme semantics of RUL; NASA scoring formula text (05 §7).
**UNVERIFIED:** dataset licence/usage terms (M1-T8, R1); third-party mirror availability (R1); column → physical-sensor-name mapping (ADR-020, not needed); which sensors are near-constant (determined in M2, M2-T1).
