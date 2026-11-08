"""Docker smoke test wrapper (marker ``docker``): runs scripts/smoke_test.sh.

Opt-in: ``RUN_DOCKER_TESTS=1 pytest -m docker --no-cov``. CI runs the script directly in its
``docker`` job.
"""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.docker
def test_docker_compose_smoke() -> None:
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "smoke_test.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "SMOKE TEST PASSED" in result.stdout
