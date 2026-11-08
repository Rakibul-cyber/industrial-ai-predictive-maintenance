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
