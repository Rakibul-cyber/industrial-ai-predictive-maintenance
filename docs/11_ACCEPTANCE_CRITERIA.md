# 11 — Acceptance Criteria (Final Definition of Done)

Checked in M7. Each item needs **evidence** (command + actual output summary, recorded in [14](14_PROJECT_STATUS.md)). An item without evidence is not done.

| # | Criterion | Evidence required | Req. |
|---|---|---|---|
| AC-01 | **Reproducible data acquisition:** fresh clone → `python scripts/download_data.py` downloads and verifies all checksums in 04 §3. | Command output; `shasum` output. | FR-001 |
| AC-02 | **Data contract holds:** validation passes on real FD001; real-data tests run (not skipped). | `pytest -v -m data` output. | FR-002, FR-003 |
| AC-03 | **Training runs:** `python scripts/train.py --model baseline`, `--model primary`, `--select` each exit 0; the primary run (incl. the 12-config search) completes in < 5 min wall time. | `time` output. | FR-006–FR-009, NFR-002 |
| AC-04 | **Real metrics:** baseline and primary CV + test MAE/RMSE/NASA score come from that run and are copied verbatim into README and 14. | `model_metadata.json` excerpt. | FR-008 |
| AC-05 | **Reproducible metrics:** second training run yields identical metrics. | "identical" output from 08 M3 command. | NFR-001 |
| AC-06 | **Selection rule applied** exactly as ADR-024: the served model has the lower CV RMSE (tie-break CV MAE); test metrics for both candidates are reported; any CV/test ranking disagreement is stated in `comparison.reason` and `reports/model_comparison.md`. | `model_metadata.json` `comparison` excerpt. | FR-009 |
| AC-07 | **Loadable artifacts:** `load_model(models/)` succeeds in a new process; contract tests pass. | Test output. | FR-009 |
| AC-08 | **API returns real predictions:** `/health` 200, `/model/info` matches metadata, `/predict` for test unit 1 (last 30 cycles) returns the same value as offline `predict_rul`. | curl outputs + contract test. | FR-010–FR-013 |
| AC-09 | **API validation:** every error code in 06 §6 covered by a passing test. | `pytest tests/test_api.py -v`. | FR-013 |
| AC-10 | **Latency:** p95 `/predict` < 200 ms (30 cycles). | 08 M4 latency command output. | NFR-003 |
| AC-11 | **Dashboard shows real API results:** prediction shown for a unit equals curl result for the same input; API-down shows error, no crash. | Manual check notes + screenshot path. | FR-014–FR-016 |
| AC-12 | **Tests pass:** full suite incl. `data` and `docker` locally; CI green on `main`. | pytest summary; CI run URL. | 09 |
| AC-13 | **Coverage** ≥ 80 % (`src/predmaint`, excl. `dashboard/app.py`). | `--cov-report=term-missing` total line. | NFR-005 |
| AC-14 | **Lint/format clean.** | `ruff check .`, `ruff format --check .`. | NFR-004 |
| AC-15 | **Docker starts:** `docker compose up -d --build` → `api` healthy, dashboard HTTP 200, smoke test passes. | `docker compose ps`, smoke test output. | FR-017 |
| AC-16 | **Pinned, consistent deps:** fresh venv from `requirements-dev.txt`, `pip check` clean; versions equal 03. | `pip check`, `pip freeze` excerpt. | NFR-008 |
| AC-17 | **Docs match code:** every command in README, 08, 10 executed from a clean clone; endpoints/fields in 06 match OpenAPI (`/openapi.json`); folder tree in 07 matches `git ls-files`. | Consistency checklist in 14. | NFR-009 |
| AC-18 | **No critical defects:** no open defect that causes a crash, wrong prediction path, data leakage, or failing acceptance item. | Open-issues list in 14 (empty or non-critical only). | — |
| AC-19 | **No placeholder text:** the placeholder grep in 08 M7 returns `clean`. | grep output. | — |
| AC-20 | **Licence/attribution:** README cites the dataset source and reference; no raw data in git. | `git ls-files data/` shows only `.gitkeep`. | 04 §11 |
| AC-21 | **No-artifact behaviour (ADR-022):** with an empty `MODEL_DIR`, `/health`, `/predict`, `/model/info` return HTTP 503 with the standard error body (`MODEL_UNAVAILABLE`). | `test_api.py` output + curl against a container started without artifacts. | FR-010, FR-013 |
| AC-22 | **Demo path (ADR-022):** from a clean clone, `python scripts/download_data.py` → `python scripts/train.py` → `docker compose up` yields a working demo (or `docker compose up` alone if M6-T4 decided to commit the artifact). | Command log in 14. | FR-017 |
