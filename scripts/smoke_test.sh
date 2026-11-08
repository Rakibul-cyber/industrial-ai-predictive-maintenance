#!/usr/bin/env bash
# Docker smoke test (docs/09, M6/M8/M9): build the image, start compose, wait for both
# healthchecks, check /health (model + intervals + anomaly loaded), POST the late-life example and
# assert its recorded critical result and rul_lower <= predicted_rul <= rul_upper (ADR-034),
# POST both examples to /anomaly and assert the late one is anomalous and scores above engine 1
# (ADR-035), check the dashboard health endpoint, tear down. Exits non-zero on any failure.
#
# Usage: bash scripts/smoke_test.sh   (needs docker compose, curl, python3)
# Runs on its own host ports (default 18000/18501), so a running demo stack is not disturbed.
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT="${SMOKE_PROJECT:-predmaint-smoke}"
TIMEOUT_S="${SMOKE_TIMEOUT_S:-240}"
export API_HOST_PORT="${SMOKE_API_PORT:-18000}" DASHBOARD_HOST_PORT="${SMOKE_DASHBOARD_PORT:-18501}"
API="http://127.0.0.1:${API_HOST_PORT}"
DASH="http://127.0.0.1:${DASHBOARD_HOST_PORT}"
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
health=$(curl -fsS "$API/health") || fail "/health request failed"
echo "   $health"
python3 -c 'import json,sys; b=json.loads(sys.argv[1]); assert b["model_loaded"] is True, b' "$health" \
  || fail "/health does not report model_loaded=true"
python3 -c 'import json,sys; b=json.loads(sys.argv[1]); assert b["intervals_loaded"] is True, b' \
  "$health" || fail "/health does not report intervals_loaded=true"
python3 -c 'import json,sys; b=json.loads(sys.argv[1]); assert b["anomaly_loaded"] is True, b' \
  "$health" || fail "/health does not report anomaly_loaded=true"

echo "== POST /predict examples/predict_request_late.json"
result=$(curl -fsS -X POST "$API/predict" -H 'Content-Type: application/json' \
  -d @examples/predict_request_late.json) || fail "/predict request failed"
echo "   $result"
python3 - "$result" <<'PY' || fail "/predict result does not match examples/expected_predictions.json"
import json, sys
got = json.loads(sys.argv[1])
expected = json.load(open("examples/expected_predictions.json"))["predict_request_late.json"]
assert got["health_status"] == "critical", got
assert got["rul_lower"] <= got["predicted_rul"] <= got["rul_upper"], got
keys = ("predicted_rul", "health_status", "cycles_received", "last_cycle", "model_version")
for key in (*keys, "rul_lower", "rul_upper", "interval_level"):
    assert got[key] == expected[key], (key, got[key], expected[key])
PY

echo "== POST /anomaly examples/predict_request.json and examples/predict_request_late.json"
early=$(curl -fsS -X POST "$API/anomaly" -H 'Content-Type: application/json' \
  -d @examples/predict_request.json) || fail "/anomaly request (engine 1) failed"
late=$(curl -fsS -X POST "$API/anomaly" -H 'Content-Type: application/json' \
  -d @examples/predict_request_late.json) || fail "/anomaly request (engine 34) failed"
echo "   $early"
echo "   $late"
python3 - "$early" "$late" <<'PY' || fail "/anomaly results do not match expectations"
import json, sys
early, late = json.loads(sys.argv[1]), json.loads(sys.argv[2])
assert late["is_anomalous"] is True, late
assert early["normalized_score"] < late["normalized_score"], (early, late)
PY

echo "== GET dashboard /_stcore/health"
dash_health=$(curl -fsS "$DASH/_stcore/health") || fail "dashboard health request failed"
[[ "$dash_health" == ok ]] || fail "dashboard health returned '${dash_health}'"
echo "   ${dash_health}"

echo "SMOKE TEST PASSED ($(docker version --format '{{.Server.Arch}}'))"
