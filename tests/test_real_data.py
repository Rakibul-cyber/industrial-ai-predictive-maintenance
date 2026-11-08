"""Integration tests on the real FD001 files; skipped when the dataset is absent (CI)."""

from pathlib import Path

import pytest

from predmaint.data.download import CMAPSS_SPEC, sha256_file
from predmaint.data.load import COLUMNS, load_fd001
from predmaint.data.validate import FD001_EXPECTED, validate_fd001

pytestmark = pytest.mark.data


def test_fd001_checksums_match_docs(real_cmapss_dir: Path) -> None:
    for name, expected in CMAPSS_SPEC.verified_files.items():
        assert sha256_file(real_cmapss_dir / name) == expected, name


def test_fd001_passes_full_validation(real_cmapss_dir: Path) -> None:
    validate_fd001(load_fd001(real_cmapss_dir))


def test_fd001_recorded_figures(real_cmapss_dir: Path) -> None:
    data = load_fd001(real_cmapss_dir)
    assert data.train.shape == (FD001_EXPECTED.train_rows, len(COLUMNS))
    assert data.test.shape == (FD001_EXPECTED.test_rows, len(COLUMNS))
    assert data.rul.shape == (FD001_EXPECTED.rul_rows, 1)
    assert int(data.test.groupby("unit")["cycle"].max().min()) == 31
    assert (int(data.rul["rul"].min()), int(data.rul["rul"].max())) == (7, 145)
