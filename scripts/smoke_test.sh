#!/usr/bin/env bash
# Docker smoke test (docs/09, M6): build the image, start compose, wait for both healthchecks,
# check /health, POST the late-life example and assert its recorded critical result, check the
# dashboard health endpoint, tear down. Exits non-zero on any failure.
#
# Usage: bash scripts/smoke_test.sh   (needs docker compose, curl, python3; ports 8000/8501 free)
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT="${SMOKE_PROJECT:-predmaint-smoke}"
TIMEOUT_S="${SMOKE_TIMEOUT_S:-240}"
# The dashboard writes FD001 into the ./data bind mount; on Linux it must run as the host user.
export APP_UID="${APP_UID:-$(id -u)}" APP_GID="${APP_GID:-$(id -g)}"
compose() { docker compose -p "$PROJECT" "$@"; }

fail() {
  echo "SMOKE TEST FAILED: $*" >&2
  compose ps || true
  compose logs --tail 40 || true
  exit 1
}
cleanup() { compose down --remove-orphans >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "== build"
compose build
mkdir -p data
echo "== up"
compose up -d

echo "== waiting up to ${TIMEOUT_S}s for both services to be healthy"
deadline=$((SECONDS + TIMEOUT_S))
while true; do
  api=$(docker inspect -f '{{.State.Health.Status}}' "$(compose ps -q api)" 2>/dev/null || echo none)
  dash=$(docker inspect -f '{{.State.Health.Status}}' "$(compose ps -q dashboard)" 2>/dev/null || echo none)
  echo "   api=${api} dashboard=${dash}"
  [[ "$api" == healthy && "$dash" == healthy ]] && break
  [[ "$api" == unhealthy || "$dash" == unhealthy ]] && fail "a service is unhealthy"
  ((SECONDS < deadline)) || fail "timed out waiting for healthy services"
  sleep 3
done

echo "== GET /health"
health=$(curl -fsS http://127.0.0.1:8000/health) || fail "/health request failed"
echo "   $health"
python3 -c 'import json,sys; b=json.loads(sys.argv[1]); assert b["model_loaded"] is True, b' "$health" \
  || fail "/health does not report model_loaded=true"

echo "== POST /predict examples/predict_request_late.json"
result=$(curl -fsS -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' \
  -d @examples/predict_request_late.json) || fail "/predict request failed"
echo "   $result"
python3 - "$result" <<'PY' || fail "/predict result does not match examples/expected_predictions.json"
import json, sys
got = json.loads(sys.argv[1])
expected = json.load(open("examples/expected_predictions.json"))["predict_request_late.json"]
assert got["health_status"] == "critical", got
for key in ("predicted_rul", "health_status", "cycles_received", "last_cycle", "model_version"):
    assert got[key] == expected[key], (key, got[key], expected[key])
PY

echo "== GET dashboard /_stcore/health"
dash_health=$(curl -fsS http://127.0.0.1:8501/_stcore/health) || fail "dashboard health request failed"
[[ "$dash_health" == ok ]] || fail "dashboard health returned '${dash_health}'"
echo "   ${dash_health}"

echo "SMOKE TEST PASSED ($(docker version --format '{{.Server.Arch}}'))"
