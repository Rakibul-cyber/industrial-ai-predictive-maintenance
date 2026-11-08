"""Model selection and promotion (ADR-024).

The served model is the candidate with the lowest 5-fold GroupKFold CV RMSE (tie-break: CV MAE,
then the simpler model in ``CANDIDATE_ORDER``). Official test metrics are reported for every
candidate but never drive the choice; a test ranking that disagrees with the CV ranking is
flagged explicitly.
"""

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from predmaint.config import HEALTH_BANDS
from predmaint.models.artifacts import artifact_paths, load_artifacts
from predmaint.models.predict import SERVED_ARTIFACT, clear_model_cache

#: Candidates in order of simplicity; the last-resort tie-break prefers the earlier one.
CANDIDATE_ORDER: tuple[str, ...] = ("baseline", "primary")
#: Metadata that must match across candidates for the comparison to be meaningful.
_COMPATIBILITY_KEYS = ("data", "config", "feature_names", "required_sensors")


class IncompatibleCandidatesError(ValueError):
    """Candidates were trained on different data, features or configuration."""


@dataclass(frozen=True)
class SelectionDecision:
    """Outcome of :func:`choose_model`."""

    selected: str
    cv_ranking: list[str]
    test_ranking: list[str]
    rankings_agree: bool
    reason: str


def _cv(meta: Mapping[str, Any]) -> Mapping[str, float]:
    return meta["metrics"]["cv"]


def _test(meta: Mapping[str, Any]) -> Mapping[str, float]:
    return meta["metrics"]["test"]


def check_compatible(candidates: Mapping[str, Mapping[str, Any]]) -> None:
    """Raise :class:`IncompatibleCandidatesError` unless all candidates share data/features."""
    names = list(candidates)
    reference = candidates[names[0]]
    for name in names[1:]:
        for key in _COMPATIBILITY_KEYS:
            if candidates[name].get(key) != reference.get(key):
                raise IncompatibleCandidatesError(
                    f"{name!r} and {names[0]!r} differ in {key!r}; retrain both on the same data"
                )


def choose_model(candidates: Mapping[str, Mapping[str, Any]]) -> SelectionDecision:
    """Apply ADR-024 to candidate metadata dicts keyed by candidate name."""
    if len(candidates) < 2:
        raise ValueError("selection needs at least two candidates")
    order = {name: i for i, name in enumerate(CANDIDATE_ORDER)}

    def cv_key(name: str) -> tuple[float, float, int]:
        cv = _cv(candidates[name])
        return (cv["rmse_mean"], cv["mae_mean"], order.get(name, len(order)))

    def test_key(name: str) -> tuple[float, float, int]:
        test = _test(candidates[name])
        return (test["rmse"], test["mae"], order.get(name, len(order)))

    cv_ranking = sorted(candidates, key=cv_key)
    test_ranking = sorted(candidates, key=test_key)
    winner, runner_up = cv_ranking[0], cv_ranking[1]
    agree = cv_ranking[0] == test_ranking[0]

    w_cv, r_cv = _cv(candidates[winner]), _cv(candidates[runner_up])
    if w_cv["rmse_mean"] < r_cv["rmse_mean"]:
        basis = (
            f"its 5-fold GroupKFold CV RMSE ({w_cv['rmse_mean']:.3f}) is lower than "
            f"{runner_up}'s ({r_cv['rmse_mean']:.3f})"
        )
    elif w_cv["mae_mean"] < r_cv["mae_mean"]:
        basis = (
            f"CV RMSE is tied ({w_cv['rmse_mean']:.3f}) and its CV MAE "
            f"({w_cv['mae_mean']:.3f}) is lower than {runner_up}'s ({r_cv['mae_mean']:.3f})"
        )
    else:
        basis = "CV RMSE and CV MAE are tied, so the simpler candidate is kept"
    reason = (
        f"Selected {winner} because {basis} (ADR-024). Official test metrics are reported but "
        "do not drive the choice."
    )
    if agree:
        reason += " The official test ranking agrees with the CV ranking."
    else:
        t_w, t_r = _test(candidates[test_ranking[0]]), _test(candidates[winner])
        reason += (
            f" WARNING: the official test ranking disagrees with the CV ranking — "
            f"{test_ranking[0]} has the lower test RMSE ({t_w['rmse']:.3f} vs "
            f"{t_r['rmse']:.3f}); per ADR-024 the CV-based choice stands."
        )
    return SelectionDecision(winner, cv_ranking, test_ranking, agree, reason)


def served_model_version(meta: Mapping[str, Any]) -> str:
    """``<name>-<YYYYMMDD of training>-<short git sha>`` (``unknown`` if no SHA)."""
    date = str(meta["trained_at"])[:10].replace("-", "")
    sha = str(meta.get("git_sha") or "unknown")
    return f"{meta['model_name']}-{date}-{sha[:7] if sha != 'unknown' else sha}"


def _comparison(
    candidates: Mapping[str, Mapping[str, Any]], decision: SelectionDecision
) -> dict[str, Any]:
    return {
        "selected": decision.selected,
        "selected_by": "cv_rmse",
        "tie_break": "cv_mae",
        "cv_ranking": decision.cv_ranking,
        "test_ranking": decision.test_ranking,
        "rankings_agree": decision.rankings_agree,
        "reason": decision.reason,
        "candidates": {
            name: {
                "model_version": meta["model_version"],
                "estimator": meta.get("estimator"),
                "hyperparameters": meta.get("hyperparameters"),
                "cv": {k: v for k, v in _cv(meta).items() if k != "folds"},
                "test": dict(_test(meta)),
                "test_clipped_truth": dict(meta["metrics"]["test_clipped_truth"]),
            }
            for name, meta in candidates.items()
        },
    }


def comparison_markdown(comparison: Mapping[str, Any]) -> str:
    """Human-readable comparison table and selection statement."""
    lines = [
        "# Model comparison",
        "",
        "| Model | Version | CV RMSE (mean ± std) | CV MAE (mean ± std) | Test RMSE | Test MAE "
        "| NASA score | Test RMSE (truth clipped) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, c in comparison["candidates"].items():
        cv, test, clipped = c["cv"], c["test"], c["test_clipped_truth"]
        marker = " (selected)" if name == comparison["selected"] else ""
        lines.append(
            f"| {name}{marker} | `{c['model_version']}` | {cv['rmse_mean']:.3f} ± "
            f"{cv['rmse_std']:.3f} | {cv['mae_mean']:.3f} ± {cv['mae_std']:.3f} | "
            f"{test['rmse']:.3f} | {test['mae']:.3f} | {test['nasa_score']:.1f} | "
            f"{clipped['rmse']:.3f} |"
        )
    lines += [
        "",
        "CV: 5-fold GroupKFold by engine on the training set (capped labels). Test: official "
        "FD001 test set, last observed cycle of each of the 100 engines, unclipped truth unless "
        "stated. Predictions clamped to [0, RUL_CAP] (ADR-027).",
        "",
        comparison["reason"],
        "",
    ]
    return "\n".join(lines)


def promote(
    model_dir: Path,
    reports_dir: Path,
    names: Sequence[str] = CANDIDATE_ORDER,
    echo: Any = print,
) -> dict[str, Any]:
    """Select among trained candidates and write the served artifact plus comparison reports.

    Writes ``model.joblib`` (byte copy of the winner), ``model_metadata.json``,
    ``reports/model_comparison.json`` and ``reports/model_comparison.md``.

    Raises:
        FileNotFoundError: if a candidate's artifacts are missing.
        IncompatibleCandidatesError: if candidates were trained on different inputs.
    """
    candidates = {name: load_artifacts(model_dir, name)[1] for name in names}
    check_compatible(candidates)
    decision = choose_model(candidates)
    comparison = _comparison(candidates, decision)
    winner_meta = candidates[decision.selected]

    source_model, _ = artifact_paths(model_dir, decision.selected)
    served_model, served_meta = artifact_paths(model_dir, SERVED_ARTIFACT)
    shutil.copyfile(source_model, served_model)
    promoted = {
        **winner_meta,
        "model_version": served_model_version(winner_meta),
        "candidate_model_version": winner_meta["model_version"],
        "selected_by": "cv_rmse",
        "promoted_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "health_bands": dict(HEALTH_BANDS),
        "comparison": comparison,
    }
    served_meta.write_text(json.dumps(promoted, indent=2) + "\n", encoding="utf-8")
    clear_model_cache()

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "model_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    markdown = comparison_markdown(comparison)
    (reports_dir / "model_comparison.md").write_text(markdown, encoding="utf-8")
    echo(markdown)
    echo(f"Promoted {decision.selected} → {served_model}, {served_meta}")
    return promoted
