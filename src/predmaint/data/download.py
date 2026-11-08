"""Download, verify and extract the NASA C-MAPSS archive (docs/04 §1-§3).

The S3 archive is nested: it contains one folder holding ``CMAPSSData.zip``. Only the 14
expected file names are extracted (flat), which also guards against path traversal.

Downloads use ``requests`` (certifi CA bundle) rather than ``urllib``: on python.org macOS
builds ``urllib`` has no default CA file and fails TLS verification (CR-001, docs/15).
"""

import hashlib
import io
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import requests

ARCHIVE_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
OUTER_ZIP_NAME = "cmapss_outer.zip"
EXTRACT_DIRNAME = "CMAPSSData"
_CHUNK = 1 << 20


class DownloadError(RuntimeError):
    """The dataset could not be downloaded or extracted."""


class ChecksumMismatchError(DownloadError):
    """A downloaded or extracted file does not match its recorded SHA-256."""


@dataclass(frozen=True)
class DatasetSpec:
    """Where the archive lives and what its contents must hash to."""

    url: str
    outer_sha256: str
    inner_member: str
    inner_sha256: str
    allowed_members: frozenset[str]
    verified_files: Mapping[str, str]


CMAPSS_SPEC = DatasetSpec(
    url=ARCHIVE_URL,
    outer_sha256="c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2",
    inner_member="6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip",
    inner_sha256="74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f",
    allowed_members=frozenset(
        ["readme.txt", "Damage Propagation Modeling.pdf"]
        + [f"{kind}_FD00{i}.txt" for kind in ("train", "test", "RUL") for i in range(1, 5)]
    ),
    verified_files=MappingProxyType(
        {
            "train_FD001.txt": "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8",
            "test_FD001.txt": "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851",
            "RUL_FD001.txt": "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca",
        }
    ),
)


@dataclass(frozen=True)
class EnsureResult:
    """Outcome of :func:`ensure_dataset`.

    ``action`` is ``"skipped"`` (files already valid), ``"extracted"`` (re-extracted from a
    valid cached archive) or ``"downloaded"`` (archive fetched from ``spec.url``).
    """

    action: Literal["skipped", "extracted", "downloaded"]
    extract_dir: Path
    verified_files: tuple[Path, ...]


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _invalid_files(extract_dir: Path, spec: DatasetSpec) -> list[str]:
    """Names of verified files that are missing or do not match their checksum."""
    bad = []
    for name, expected in spec.verified_files.items():
        path = extract_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            bad.append(name)
    return bad


def _download(url: str, target: Path, expected_sha256: str, timeout: float) -> None:
    """Stream ``url`` to ``target`` via a ``.part`` file; keep it only if the hash matches."""
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with partial.open("wb") as fh:
                for chunk in response.iter_content(_CHUNK):
                    fh.write(chunk)
                    digest.update(chunk)
    except (requests.RequestException, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"Download failed from {url}: {exc}") from exc
    actual = digest.hexdigest()
    if actual != expected_sha256:
        partial.unlink(missing_ok=True)
        raise ChecksumMismatchError(
            f"Archive checksum mismatch for {url}: expected {expected_sha256}, got {actual}. "
            "The hosted file may have changed; see docs/04 §1 and risk R1."
        )
    partial.replace(target)


def _extract(outer_zip: Path, extract_dir: Path, spec: DatasetSpec) -> None:
    """Extract the allowed members of the nested inner zip into ``extract_dir``."""
    try:
        with zipfile.ZipFile(outer_zip) as outer:
            inner_bytes = outer.read(spec.inner_member)
    except KeyError as exc:
        raise DownloadError(f"Archive does not contain {spec.inner_member!r}") from exc
    except zipfile.BadZipFile as exc:
        raise DownloadError(f"Archive {outer_zip} is not a valid zip file: {exc}") from exc

    actual = hashlib.sha256(inner_bytes).hexdigest()
    if actual != spec.inner_sha256:
        raise ChecksumMismatchError(
            f"Inner archive checksum mismatch: expected {spec.inner_sha256}, got {actual}"
        )

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        for info in inner.infolist():
            # Exact flat names only: anything else (dirs, "../x", unknown files) is ignored.
            if info.filename in spec.allowed_members:
                (extract_dir / info.filename).write_bytes(inner.read(info))


def ensure_dataset(
    raw_dir: Path, spec: DatasetSpec = CMAPSS_SPEC, timeout: float = 60.0
) -> EnsureResult:
    """Make sure the verified dataset files exist under ``raw_dir/CMAPSSData``.

    Idempotent: if every file in ``spec.verified_files`` already matches its checksum,
    nothing is downloaded or written. A valid cached outer archive is re-used.

    Raises:
        DownloadError: on network/IO failure or a malformed archive.
        ChecksumMismatchError: if the archive or any verified file has an unexpected hash.
    """
    extract_dir = raw_dir / EXTRACT_DIRNAME
    verified = tuple(extract_dir / name for name in spec.verified_files)
    if not _invalid_files(extract_dir, spec):
        return EnsureResult("skipped", extract_dir, verified)

    raw_dir.mkdir(parents=True, exist_ok=True)
    outer_zip = raw_dir / OUTER_ZIP_NAME
    action: Literal["extracted", "downloaded"] = "extracted"
    if not (outer_zip.is_file() and sha256_file(outer_zip) == spec.outer_sha256):
        _download(spec.url, outer_zip, spec.outer_sha256, timeout)
        action = "downloaded"

    _extract(outer_zip, extract_dir, spec)
    bad = _invalid_files(extract_dir, spec)
    if bad:
        raise ChecksumMismatchError(
            f"Extracted files missing or failing checksum verification: {', '.join(bad)}"
        )
    return EnsureResult(action, extract_dir, verified)
