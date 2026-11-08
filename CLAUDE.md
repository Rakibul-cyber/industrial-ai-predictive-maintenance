# CLAUDE.md — Standing Instructions

## Project purpose

Portfolio-grade, end-to-end ML system: RUL regression on NASA C-MAPSS **FD001**, served by a FastAPI API and a Streamlit dashboard, tested, containerised, and documented with honest, reproducible metrics.

## Approved architecture (summary — details in docs/)

- Modular monolith: package `src/predmaint/` with `data/`, `features/`, `models/`, `api/`, `dashboard/`; thin `scripts/`.
- Python 3.13; exact pins in `requirements*.txt` (docs/03).
- Target: `RUL = max_cycle − cycle`, clipped at 125 (configurable). Features: kept sensors (≤ 2 unique values dropped), trailing rolling mean/std over 30 cycles, plus `cycle`; settings excluded; only rows with `cycle ≥ 30`.
- Baseline `StandardScaler + Ridge`; primary `HistGradientBoostingRegressor` from one bounded 12-config search (ADR-026). 5-fold GroupKFold by unit; official test at last cycle; MAE/RMSE primary, NASA score secondary. Served model = lower CV RMSE (ADR-024); test metrics reported, never used for selection. Predictions clamped to [0, RUL_CAP] (ADR-027); health status critical < 20 ≤ warning < 60 ≤ healthy (ADR-025).
- Artifacts: `models/model.joblib` + `models/model_metadata.json`.
- API: `GET /health`, `POST /predict`, `GET /model/info`, error envelope `{"error": {code, message, details}}`.
- Dashboard calls the API over HTTP (`API_BASE_URL`), never imports `predmaint.models`.
- One Docker image, compose services `api` (8000) and `dashboard` (8501); served model committed and baked into the image (ADR-030/032); `./data` bind-mounted, FD001 downloaded by the dashboard on first start (ADR-031).
- ADR status: architecture frozen 2026-11-08 — ADR-001…022 *Accepted* (docs/13). Changes only via docs/15.

## Source of truth and session-start reading order

1. `CLAUDE.md` (this file)
2. `docs/14_PROJECT_STATUS.md` — current milestone, blockers, next authorized action
3. `docs/08_IMPLEMENTATION_PLAN.md` — the authorized milestone's files, tasks, commands, acceptance criteria
4. Design docs relevant to the task: 04 (data), 05 (ML), 06 (API), 07 (structure), 09 (tests), 10 (deployment), 13 (decisions), 15 (change control)

If code and docs disagree, stop and raise it via docs/15 — do not pick one silently.

## Development principles

- **Plan once, implement once:** follow 08; do not redesign during implementation.
- **No hallucinations:** never invent columns, metrics, library behaviour, URLs, or results. Report only what a command actually printed. Mark anything unconfirmed **UNVERIFIED**.
- **Minimal changes:** touch only files listed for the milestone (plus docs that must be updated).
- **No scope creep:** out-of-scope list in docs/00 §6. New ideas go to the user, not into the code.

## Coding standards

- `ruff check .` and `ruff format --check .` must pass.
- Type hints on all public functions; concise docstrings stating inputs, outputs, and raised errors.
- No logic in `scripts/`; no notebooks as source of truth (exploration results go into docs as tables).
- Config via `predmaint/config.py` + env vars; no hard-coded absolute paths or secrets.
- Feature computation exists in exactly one place (`features/build.py`), used by training, evaluation, and API.

## Warnings policy

- `DeprecationWarning` is an error in the test suite (`filterwarnings = ["error::DeprecationWarning"]` in `pyproject.toml`).
- If a warning originates in project code: fix the cause.
- If it originates inside a third-party library: add a narrowly scoped `filterwarnings` entry (module + message pattern) in `pyproject.toml`, with a comment naming the library, the warning, and why it is out of our control.
- Never disable warnings globally or per test without that comment.

## Scope restrictions

FD001 only. No anomaly detection, other subsets, deep learning, hyperparameter search, MLflow, cloud deploy, auth, or drift monitoring unless approved through docs/15.

## Testing requirements

- Each milestone must pass the tests listed for it in docs/09 §4, plus all previously passing tests.
- `data` tests must run (not skip) locally for M1, M3, M7 sign-off; `docker` tests for M6, M7.
- Leakage tests (grouped folds, causal features) and the API==offline contract test are mandatory.

## Git safety rules

- **Never commit or push without explicit user authorization** for that specific commit/push.
- Never force-push, rewrite history, or change branches/remotes without explicit instruction.
- Never modify, delete, or reformat files unrelated to the authorized milestone.
- Never commit `data/` contents or `*.joblib` artifacts.

## Change control

Any deviation from an approved decision or contract follows docs/15 (change record with evidence, superseding ADR, approval). Approved decisions are never reversed silently.

## Milestone execution rules

- Implement **only** the milestone the user has authorized; do not start the next one.
- Stop when the milestone's acceptance criteria in 08 are met.
- Report actual command outputs (summarised faithfully), including failures.
- If blocked, stop and report the blocker with evidence rather than working around it.

## Documentation update rules

- After every milestone: update `docs/14_PROJECT_STATUS.md` (status, evidence, measured results, blockers, next action).
- Every new or changed decision: `docs/13_DECISION_LOG.md`.
- Keep each doc to its single purpose; reference other docs instead of duplicating content.

## Definition of Done (per milestone)

1. All files listed for the milestone in 08 exist; no unlisted files created.
2. Verification commands in 08 executed; outputs recorded in 14.
3. Required tests (09 §4) pass; ruff clean.
4. Acceptance criteria in 08 met; for M7, every item of docs/11 checked with evidence.
5. Docs updated (14 always; 13/others as affected); no placeholder text.
6. Work left uncommitted unless the user authorized a commit.
