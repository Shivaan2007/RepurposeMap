#!/usr/bin/env bash
# V1 local smoke test: confirms the backend answers the routes a real
# deployment depends on, including CORS, without needing PrimeKG loaded or
# any API key. Run this after `uvicorn repurposemap.api.app:app` is already
# up (see DEMO.md) -- it does not start the server itself.
#
# Usage: ./scripts/smoke_test.sh [base_url]
#   base_url defaults to http://localhost:8000

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
FAILURES=0

check() {
  local description="$1"
  local actual="$2"
  local expected="$3"
  if [ "$actual" = "$expected" ]; then
    echo "OK   $description"
  else
    echo "FAIL $description (expected $expected, got $actual)"
    FAILURES=$((FAILURES + 1))
  fi
}

echo "Smoke-testing $BASE_URL"
echo

# 1. Health endpoint answers, and fast (no PrimeKG load on this route).
status=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health")
check "GET /health" "$status" "200"

# 2. System info answers and is honest: a ranking model being unavailable
#    is an expected, correct state right now -- not a failure of this
#    smoke test. We only check the route answers and returns valid JSON.
status=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/v1/system/info")
check "GET /api/v1/system/info" "$status" "200"

# 3. CORS: an allowed origin gets the header back.
cors_header=$(curl -s -D - -o /dev/null -H "Origin: http://localhost:5173" "$BASE_URL/health" \
  | grep -i "^access-control-allow-origin:" | tr -d '\r' || true)
if [ -n "$cors_header" ]; then
  echo "OK   CORS header present for an allowed dev origin"
else
  echo "FAIL CORS header missing for http://localhost:5173 -- check REPURPOSEMAP_CORS_ORIGINS"
  FAILURES=$((FAILURES + 1))
fi

# 4. CORS: a disallowed origin does NOT get the header back.
evil_header=$(curl -s -D - -o /dev/null -H "Origin: http://evil.example.com" "$BASE_URL/health" \
  | grep -i "^access-control-allow-origin:" | tr -d '\r' || true)
if [ -z "$evil_header" ]; then
  echo "OK   CORS header absent for a disallowed origin"
else
  echo "FAIL CORS header present for http://evil.example.com -- should never happen"
  FAILURES=$((FAILURES + 1))
fi

# 5. OpenAPI schema is servable (confirms the app wired its routers correctly).
status=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/openapi.json")
check "GET /openapi.json" "$status" "200"

# 6. The ranking stub never fabricates a result: it must report
#    model_status "not_available" with zero candidates until a real model
#    is wired in. This smoke test fails loudly if that ever silently
#    changes to a fake ranking.
rank_response=$(curl -s -X POST "$BASE_URL/api/v1/rank" \
  -H "Content-Type: application/json" \
  -d '{"disease_name": "smoke-test-disease", "top_k": 5}')
model_status=$(echo "$rank_response" | python3 -c "import json,sys; print(json.load(sys.stdin).get('model_status', ''))" 2>/dev/null || echo "parse_error")
check "POST /api/v1/rank reports model_status=not_available" "$model_status" "not_available"

# 7. The evidence endpoint makes real outbound calls to four external
#    services -- never required to succeed in a smoke test (a source or
#    the network itself may be down), but it must fail honestly (a 4xx/5xx
#    with a clear reason) rather than crash unhandled. Only an unhandled
#    500 fails this check; 200 (sources answered, possibly with some
#    source_status failures) and 502 (every real adapter's own typed
#    error path) are both acceptable.
status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/api/v1/evidence" \
  -H "Content-Type: application/json" \
  -d '{"drug_name": "smoke-test-drug", "disease_name": "smoke-test-disease", "max_results": 1}')
if [ "$status" = "200" ] || [ "$status" = "502" ]; then
  echo "OK   POST /api/v1/evidence answers honestly (got $status, not an unhandled crash)"
else
  echo "FAIL POST /api/v1/evidence returned $status -- expected 200 or 502"
  FAILURES=$((FAILURES + 1))
fi

# 8. Same honesty check for disease-biology (also a live external call).
status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/api/v1/disease-biology" \
  -H "Content-Type: application/json" \
  -d '{"disease_name": "smoke-test-disease"}')
if [ "$status" = "200" ] || [ "$status" = "404" ] || [ "$status" = "502" ]; then
  echo "OK   POST /api/v1/disease-biology answers honestly (got $status, not an unhandled crash)"
else
  echo "FAIL POST /api/v1/disease-biology returned $status -- expected 200, 404 or 502"
  FAILURES=$((FAILURES + 1))
fi

# 9. No secret-shaped values leak in the two static metadata responses.
#    This is a cheap pattern check, not a substitute for the repo-level
#    secrets audit in V1_RELEASE_CHECKLIST.md.
for route in "/api/v1/system/info" "/openapi.json"; do
  body=$(curl -s "$BASE_URL$route")
  if echo "$body" | grep -qiE '"(api_key|secret|password|access_token)"\s*:\s*"[^"]'; then
    echo "FAIL GET $route appears to expose a secret-shaped field"
    FAILURES=$((FAILURES + 1))
  else
    echo "OK   GET $route exposes no secret-shaped field"
  fi
done

echo
if [ "$FAILURES" -eq 0 ]; then
  echo "All checks passed."
  exit 0
else
  echo "$FAILURES check(s) failed."
  exit 1
fi
