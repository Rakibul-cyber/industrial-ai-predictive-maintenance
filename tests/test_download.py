"""Tests for predmaint.data.download against a local stdlib HTTP server (no internet)."""

import functools
import hashlib
import io
import threading
import zipfile
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import MappingProxyType

import pytest

from predmaint.data.download import (
    CMAPSS_SPEC,
    OUTER_ZIP_NAME,
    ChecksumMismatchError,
    DatasetSpec,
    DownloadError,
    ensure_dataset,
    sha256_file,
)

INNER_MEMBER = "folder/inner.zip"
GOOD_FILES = {"train_FD001.txt": b"1 1 0.1  \n", "RUL_FD001.txt": b"7 \n"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence per-request logging."""


@pytest.fixture
def server(tmp_path: Path) -> Iterator[tuple[Path, str]]:
    """Serve ``tmp_path / "srv"`` over HTTP on a free localhost port."""
    root = tmp_path / "srv"
    root.mkdir()
    handler = functools.partial(_QuietHandler, directory=str(root))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    yield root, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _make_archive(
    server: tuple[Path, str], inner_members: dict[str, bytes]
) -> tuple[Path, DatasetSpec]:
    """Publish a nested archive like the NASA one and return a spec that matches it."""
    root, base_url = server
    inner = _zip_bytes(inner_members)
    outer = _zip_bytes({INNER_MEMBER: inner})
    outer_path = root / "archive.zip"
    outer_path.write_bytes(outer)
    spec = DatasetSpec(
        url=f"{base_url}/archive.zip",
        outer_sha256=_sha(outer),
        inner_member=INNER_MEMBER,
        inner_sha256=_sha(inner),
        allowed_members=frozenset(GOOD_FILES),
        verified_files=MappingProxyType({n: _sha(c) for n, c in GOOD_FILES.items()}),
    )
    return outer_path, spec


def test_downloads_verifies_and_extracts(tmp_path: Path, server: tuple[Path, str]) -> None:
    _, spec = _make_archive(server, GOOD_FILES)
    result = ensure_dataset(tmp_path / "raw", spec)
    assert result.action == "downloaded"
    for name, content in GOOD_FILES.items():
        assert (tmp_path / "raw" / "CMAPSSData" / name).read_bytes() == content
    assert (tmp_path / "raw" / OUTER_ZIP_NAME).is_file()
    assert not (tmp_path / "raw" / f"{OUTER_ZIP_NAME}.part").exists()


def test_second_run_skips_without_touching_the_source(
    tmp_path: Path, server: tuple[Path, str]
) -> None:
    source, spec = _make_archive(server, GOOD_FILES)
    ensure_dataset(tmp_path / "raw", spec)
    source.unlink()  # any download attempt would now fail
    assert ensure_dataset(tmp_path / "raw", spec).action == "skipped"


def test_corrupted_file_is_reextracted_from_cached_archive(
    tmp_path: Path, server: tuple[Path, str]
) -> None:
    source, spec = _make_archive(server, GOOD_FILES)
    ensure_dataset(tmp_path / "raw", spec)
    target = tmp_path / "raw" / "CMAPSSData" / "train_FD001.txt"
    target.write_bytes(b"tampered")
    source.unlink()
    assert ensure_dataset(tmp_path / "raw", spec).action == "extracted"
    assert target.read_bytes() == GOOD_FILES["train_FD001.txt"]


def test_archive_checksum_mismatch_fails_and_keeps_nothing(
    tmp_path: Path, server: tuple[Path, str]
) -> None:
    _, spec = _make_archive(server, GOOD_FILES)
    bad_spec = DatasetSpec(**{**spec.__dict__, "outer_sha256": "0" * 64})
    with pytest.raises(ChecksumMismatchError, match="Archive checksum mismatch"):
        ensure_dataset(tmp_path / "raw", bad_spec)
    assert list((tmp_path / "raw").iterdir()) == []


def test_inner_checksum_mismatch_fails(tmp_path: Path, server: tuple[Path, str]) -> None:
    _, spec = _make_archive(server, GOOD_FILES)
    bad_spec = DatasetSpec(**{**spec.__dict__, "inner_sha256": "0" * 64})
    with pytest.raises(ChecksumMismatchError, match="Inner archive"):
        ensure_dataset(tmp_path / "raw", bad_spec)


def test_extracted_file_checksum_mismatch_fails(tmp_path: Path, server: tuple[Path, str]) -> None:
    _, spec = _make_archive(server, GOOD_FILES)
    wrong = MappingProxyType({**spec.verified_files, "RUL_FD001.txt": "0" * 64})
    bad_spec = DatasetSpec(**{**spec.__dict__, "verified_files": wrong})
    with pytest.raises(ChecksumMismatchError, match="RUL_FD001.txt"):
        ensure_dataset(tmp_path / "raw", bad_spec)


def test_missing_member_fails(tmp_path: Path, server: tuple[Path, str]) -> None:
    _, spec = _make_archive(server, {"train_FD001.txt": GOOD_FILES["train_FD001.txt"]})
    with pytest.raises(ChecksumMismatchError, match="RUL_FD001.txt"):
        ensure_dataset(tmp_path / "raw", spec)


def test_unexpected_and_traversal_members_are_not_extracted(
    tmp_path: Path, server: tuple[Path, str]
) -> None:
    members = {**GOOD_FILES, "../evil.txt": b"x", "sub/train_FD001.txt": b"y", "other.txt": b"z"}
    _, spec = _make_archive(server, members)
    ensure_dataset(tmp_path / "raw", spec)
    extracted = sorted(p.name for p in (tmp_path / "raw" / "CMAPSSData").iterdir())
    assert extracted == sorted(GOOD_FILES)
    assert not (tmp_path / "raw" / "evil.txt").exists()
    assert not (tmp_path / "evil.txt").exists()


def test_http_error_raises_download_error(tmp_path: Path, server: tuple[Path, str]) -> None:
    _, spec = _make_archive(server, GOOD_FILES)
    bad_spec = DatasetSpec(**{**spec.__dict__, "url": f"{server[1]}/missing.zip"})
    with pytest.raises(DownloadError, match="Download failed"):
        ensure_dataset(tmp_path / "raw", bad_spec)
    assert not (tmp_path / "raw" / f"{OUTER_ZIP_NAME}.part").exists()


def test_not_a_zip_raises_download_error(tmp_path: Path, server: tuple[Path, str]) -> None:
    payload = b"not a zip"
    (server[0] / "bad.zip").write_bytes(payload)
    url = f"{server[1]}/bad.zip"
    spec = DatasetSpec(**{**CMAPSS_SPEC.__dict__, "url": url, "outer_sha256": _sha(payload)})
    with pytest.raises(DownloadError, match="not a valid zip"):
        ensure_dataset(tmp_path / "raw", spec)


def test_cmapss_spec_matches_documented_values() -> None:
    """Guards against typos versus docs/04 §1-§3."""
    assert CMAPSS_SPEC.url.startswith("https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan")
    assert len(CMAPSS_SPEC.allowed_members) == 14
    assert set(CMAPSS_SPEC.verified_files) <= CMAPSS_SPEC.allowed_members
    assert CMAPSS_SPEC.verified_files["train_FD001.txt"].startswith("963b5e22")
    assert CMAPSS_SPEC.outer_sha256.startswith("c9c5dec1")


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == _sha(b"abc")


@pytest.mark.network
def test_live_download_from_nasa(tmp_path: Path) -> None:
    result = ensure_dataset(tmp_path / "raw")
    assert result.action == "downloaded"
    for path in result.verified_files:
        assert sha256_file(path) == CMAPSS_SPEC.verified_files[path.name]
