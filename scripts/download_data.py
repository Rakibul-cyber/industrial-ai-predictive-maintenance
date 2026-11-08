"""Download and verify the NASA C-MAPSS dataset into ``$DATA_DIR/raw/CMAPSSData``.

Thin CLI over :func:`predmaint.data.download.ensure_dataset` (docs/04 §1). Idempotent.
"""

import argparse
import sys
from pathlib import Path

from predmaint.config import get_settings
from predmaint.data.download import DownloadError, ensure_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="data root (default: $DATA_DIR or 'data')"
    )
    args = parser.parse_args()
    raw_dir = (args.data_dir or get_settings().data_dir) / "raw"
    try:
        result = ensure_dataset(raw_dir)
    except DownloadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    messages = {
        "skipped": "Dataset already present and checksums match; nothing downloaded.",
        "extracted": "Re-extracted from the cached, verified archive.",
        "downloaded": "Downloaded, verified and extracted the archive.",
    }
    print(messages[result.action])
    for path in result.verified_files:
        print(f"  verified {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
