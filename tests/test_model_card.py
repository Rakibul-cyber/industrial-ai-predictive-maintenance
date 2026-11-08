"""docs/18_MODEL_CARD.md: every numeric table cites existing source files (ADR-040)."""

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "docs" / "18_MODEL_CARD.md"


def _tables(lines: list[str]) -> list[tuple[int, int]]:
    """(first, last) line index of each Markdown table."""
    spans, start = [], None
    for i, line in enumerate([*lines, ""]):
        if line.startswith("|") and start is None:
            start = i
        elif not line.startswith("|") and start is not None:
            spans.append((start, i - 1))
            start = None
    return spans


def test_every_numeric_table_names_existing_sources() -> None:
    lines = CARD.read_text(encoding="utf-8").splitlines()
    numeric = [
        (a, b)
        for a, b in _tables(lines)
        if any(re.search(r"\d", line) for line in lines[a + 2 : b + 1])
    ]
    assert (
        len(numeric) >= 8
    )  # model, hashes, data, point, intervals, anomaly, residuals, importance
    for a, b in numeric:
        following = next(line for line in lines[b + 1 :] if line.strip())
        assert following.startswith("Source: "), f"table at line {a + 1} has no Source line"
        paths = re.findall(r"`([^`]+)`", following)
        assert paths, f"Source line after table at line {a + 1} names no file"
        for path in paths:
            assert (ROOT / path).is_file(), f"{path} (cited after table at line {a + 1}) missing"


def test_artifact_hash_table_matches_files() -> None:
    """The card is regenerated when an artifact changes; a stale hash fails here."""
    rows = re.findall(r"^\| `(models/[^`]+)` \| `([0-9a-f]{64})` \|$", CARD.read_text(), re.M)
    assert len(rows) == 7
    for path, digest in rows:
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
