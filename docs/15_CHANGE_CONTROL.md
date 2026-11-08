# 15 — Change Control

Applies once the architecture is frozen (approval of M0). Approved decisions in [13](13_DECISION_LOG.md) and the contracts in 04, 05, 06, 07 are **never reversed or altered silently** — not in code, not in docs.

## 1. Permitted reasons for change

1. **Evidence contradicts a documented assumption** (e.g. a command output, data inspection, or library behaviour differs from what a doc states).
2. **Defect** that blocks an acceptance criterion or causes wrong results, leakage, or a crash.
3. **External change** (URL moved, package yanked, security advisory, deprecation).
4. **Explicit user request** (including scope additions from the out-of-scope list in 00 §6).

"Nicer design", unrequested refactors, or extra features are not permitted reasons.

## 2. Required change record

Append to the "Change records" section below **before** implementing, and get user approval unless the change is a pure defect fix inside the currently authorized milestone that touches no contract (04/05/06/07/13).

| Field | Content |
|---|---|
| ID / date | `CR-NNN`, YYYY-MM-DD |
| Reason | One of §1, one sentence |
| Evidence | Exact command(s) and actual output excerpt, or file/line reference |
| Impact | Requirements, ADRs, milestones, metrics, or API contract affected |
| Smallest fix | The minimal change that resolves it; alternatives considered |
| Affected docs | Every doc to update (always 14; 13 if a decision changes) |
| Regression tests | Tests added/changed to lock in the fix |
| Approval | "Approved by user on YYYY-MM-DD" or "Defect fix within authorized milestone" |

## 3. Procedure

1. Stop implementation of the affected part.
2. Write the change record with evidence.
3. If a decision changes: add a new ADR that **supersedes** the old one; mark the old one `Superseded by ADR-xxx` (do not delete its text).
4. Obtain approval (per §2).
5. Implement the smallest fix + regression test; update all affected docs in the same working-tree change.
6. Re-run the current milestone's verification commands; record outputs in 14.

## Change records

### CR-001 — Dataset download fails TLS verification with stdlib urllib (2026-11-08)

| Field | Content |
|---|---|
| Reason | Defect (§1.2) found during M1 verification. |
| Evidence | `python scripts/download_data.py` → `ERROR: Download failed from https://phm-datasets.s3.amazonaws.com/...: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate>`. `ssl.get_default_verify_paths().cafile` → `None` (python.org macOS build; `Install Certificates.command` not run). Same URL via `requests.head(...)` → `200 12429152` using the certifi bundle. `curl` worked in M0 because it uses the system store. |
| Impact | FR-001 on any python.org macOS install. No contract, ADR, or dependency change (`requests` already pinned, docs/03). |
| Smallest fix | `_download` uses `requests.get(stream=True)` + `raise_for_status()` instead of `urllib`. Rejected: asking users to run `Install Certificates.command` (machine-specific, breaks reproducibility); importing `certifi` directly (transitive, unpinned). |
| Affected docs | 03 (requests row), 09 §3 (download tests use local HTTP server), 14. |
| Regression tests | `tests/test_download.py` now serves archives over a local HTTP server (exercises the `requests` path, incl. HTTP 404 → `DownloadError`); `RUN_NETWORK_TESTS=1` live test passes. |
| Approval | Defect fix within authorized milestone (M1). |

### CR-002 — Baseline training and evaluation moved from M3 into M2 (2026-11-08)

| Field | Content |
|---|---|
| Reason | Explicit user request (§1.4), in the Milestone 2 authorization message. |
| Evidence | That message specifies M2 = preprocessing, features, baseline training, CV, official-test evaluation, `models/baseline.joblib` + `models/baseline_metadata.json`, committed `reports/metrics_baseline.json`, and the kept-sensor list stored in the preprocessing artifact. Doc 08 had placed training in M3. |
| Impact | 08 (M2/M3 scope and traceability), 01 (FR-006, FR-008, NFR-007 milestones), 05 §2/§4/§5/§7a/§9, 07 (`reports/`, per-candidate files), 09 §4, 13 (ADR-023). Served-model API contract unchanged. |
| Smallest fix | Re-scope milestones; per-candidate artifact naming alongside the unchanged served artifact; built-in `ColumnTransformer` select step to carry the feature list. Entrypoint kept as `scripts/train.py --model baseline` (the name in 08, permitted by the request). |
| Affected docs | 01, 05, 07, 08, 09, 13, 14. |
| Regression tests | `tests/test_train.py` (end-to-end save/reload/predict, pipeline feature list == metadata, missing feature column rejected), `tests/test_real_data.py` (sensor list). |
| Approval | Requested by the user on 2026-11-08 (M2 authorization). |

### CR-003 — Milestone 3 amendments: CV-based selection, health bands, bounded search, prediction cap (2026-11-08)

| Field | Content |
|---|---|
| Reason | Explicit user request (§1.4) in the Milestone 3 authorization. |
| Evidence | That message: (1) choose the served model by 5-fold GroupKFold CV RMSE (tie-break CV MAE); report test metrics without letting them drive the choice; flag CV/test ranking disagreement. (2) Health bands ≥ 60 healthy, 20–59 warning, < 20 critical, as constants in `config.py`. (3) One bounded HGB search of ≤ ~12 configurations scored by CV RMSE. (4) `predicted_rul` non-negative and capped at `RUL_CAP`. Measured effect of (4) on the baseline: one test prediction 130.58 > 125; test RMSE 19.205 → 19.144. |
| Impact | ADR-010 → ADR-024; ADR-017 → ADR-025; ADR-007 amended by ADR-026; ADR-018 → ADR-027. API contract: `health_band` → `health_status` (lowercase), `cycles_received` added, `selected_over_baseline` → `selected_by`, error class names, cycle order checked as sent. 05, 06, 00, 01, 02, 04, 07, 09, 11, 12, CLAUDE.md. Baseline metrics re-measured. |
| Smallest fix | New ADRs superseding the old ones (old text kept); selection in `models/selection.py`; shared clamp gains an upper bound; inference contract in `models/predict.py` (doc 07's location). |
| Affected docs | 00, 01, 02, 04, 05, 06, 07, 08, 09, 11, 12, 13, 14, CLAUDE.md, README. |
| Regression tests | `tests/test_selection.py`, `tests/test_inference_contract.py`, primary tests in `tests/test_train.py`, `tests/test_real_data.py` promoted-model RMSE. |
| Approval | Requested by the user on 2026-11-08 (M3 authorization). |

### CR-004 — Milestone 4 API amendments (2026-11-08)

| Field | Content |
|---|---|
| Reason | Explicit user request (§1.4) in the Milestone 4 authorization. |
| Evidence | That message: `/health` → 200 with `model_loaded: false` when `MODEL_PATH` points nowhere; env vars `MODEL_PATH`, `API_HOST`, `API_PORT`, `LOG_LEVEL`; `/health` fields incl. `api_version`; `/model/info` incl. `n_features`; committed real-data examples from FD001 test engine 1 with a sidecar of expected predictions. |
| Impact | ADR-028 (amends ADR-022: `/health` status code), ADR-029 (exception to ADR-003's no-redistribution policy). Additions not contradicting 06: `MODEL_PATH`/`API_HOST`/`API_PORT` env vars (10 §3), `n_features` + `api_version` fields. Planned compose healthcheck now reads `model_loaded`. The request keeps 06's field names (`unit_id`, `records`) rather than `engine_id`, because 06 is not silent on them. |
| Smallest fix | New ADRs; `Settings.model_path/api_host/api_port`; `load_model` accepts a `.joblib` path; docs updated. |
| Affected docs | 01, 02, 04, 05, 06, 07, 08, 09, 10, 11, 13, 14, README, `.env.example`. |
| Regression tests | `tests/test_api.py` (health with/without model, `MODEL_PATH` env), `tests/test_real_data.py` (committed examples vs sidecar). |
| Approval | Requested by the user on 2026-11-08 (M4 authorization). |

### CR-005 — Milestone 5 dashboard specifics (2026-11-08)

| Field | Content |
|---|---|
| Reason | Explicit user request (§1.4) in the Milestone 5 authorization. |
| Evidence | That message: thin `api_client.py` returning typed results (not exceptions); five sections (status, explorer, trajectory, upload, model info); dashboard may import only the data loader and config; late-life example `examples/predict_request_late.json`. 08 had planned `dashboard/client.py` raising `ApiUnavailableError`/`ApiError`. |
| Impact | 07/08 file names (`api_client.py`, `helpers.py`); 02 §6 flow; 09 M5 tests; the ADR-029 excerpt grows to two test engines (1 and 34). No ADR change; no new dependency (charts use Streamlit's built-in `st.vega_lite_chart`; no Altair import). |
| Smallest fix | Follow the requested module name and result type; keep display logic in `helpers.py` so it is unit-testable. |
| Affected docs | 02, 06, 07, 08, 09, 10, 14, README, `.env.example`. |
| Regression tests | `tests/test_dashboard_client.py`, `tests/test_dashboard_helpers.py`, `tests/test_dashboard_app.py`, `tests/test_real_data.py`. |
| Approval | Requested by the user on 2026-11-08 (M5 authorization). |

### CR-006 — README Option B retrained the model and broke the artifact tests (2026-11-08, M7 fresh-clone run)

| Field | Content |
|---|---|
| Reason | Documentation defect (§1.2) found by the M7 fresh-clone verification. |
| Evidence | Fresh clone of `ccf68a7`, README Option B followed literally (it included the three `train.py` commands): `pytest -q` → `3 failed, 184 passed`, all in `tests/test_model_artifact.py::test_committed_artifact_reproduces_sidecar`, with `assert 'primary-20261108-ccf68a7' == 'primary-20261108-f4c86df'`. The retrained `model.joblib` was byte-identical to the committed one; only `model_version` (which embeds the git SHA) changed, while `examples/expected_predictions.json` still recorded the old version. Doc 10 §6's retraining procedure had the same gap. |
| Impact | README Option B and doc 10 §6 only. No code, model or contract change. The test is correct: it detects a sidecar that is out of sync with the committed model. |
| Smallest fix | README: Option B no longer retrains; a separate optional "Retrain" block adds the three `make_request_payload.py --expected` commands that refresh the sidecar. Doc 10 §6: same step, plus `examples/expected_predictions.json` in the commit list. |
| Affected docs | README, 10, 14. |
| Regression tests | Existing `tests/test_model_artifact.py` (unchanged). Re-run in the fresh clone: corrected Option B → `187 passed, 2 skipped`; retrain + sidecar refresh → `187 passed, 2 skipped`. |
| Approval | Documentation defect fix within the authorized milestone (M7). |

### CR-007 — Project title overstated scope: remove the unimplemented "Anomaly Detection" claim (2026-11-08)

| Field | Content |
|---|---|
| Reason | Credibility defect (§1.2) and explicit user request (§1.4): the title "Industrial AI — Predictive Maintenance & Anomaly Detection Platform" names a capability that is not implemented. Anomaly detection was an optional extension in doc 00 and was never authorized. |
| Evidence | `git grep -n -i "anomaly"` on `32fdae0` → 10 occurrences, none in `src/`, `tests/`, `scripts/`, the API or requirements (01). The only title occurrence is `docs/00_PROJECT_CHARTER.md:3`, plus a justification of that title at `docs/00:44`. The code titles (FastAPI "Predictive Maintenance — RUL API", Streamlit "Predictive Maintenance — RUL", `pyproject.toml` description) contain neither the old title nor an anomaly claim. |
| Impact | Naming only. No behaviour, API contract, model, test or dependency change. |
| Smallest fix | Rename to "Industrial AI — Predictive Maintenance Platform" in doc 00 (the only remaining old-title occurrence; README and doc 17 already use the new title). Rewrite doc 00's justification as a planned extension labelled NOT IMPLEMENTED. Label CLAUDE.md's candidate list NOT IMPLEMENTED. Keep the honest "planned/not implemented" mentions in README, doc 12, doc 13 (historical ADR-001 text) and doc 16 §8. Code strings unchanged because none overstated scope. ADR-033 records the rename. |
| Affected docs | 00, 13, 14, 15, CLAUDE.md. |
| Regression tests | Full `pytest -q`, `ruff check .`, `ruff format --check .`; no test change needed (`tests/test_api.py` asserts only `"RUL" in title`). |
| Approval | Requested by the user on 2026-11-08 (CR-007). |
