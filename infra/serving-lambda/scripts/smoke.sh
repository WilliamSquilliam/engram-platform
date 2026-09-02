#!/usr/bin/env bash
# ============================================================================
# smoke.sh — end-to-end health check of the Lambda serving unit, from the
# operator's machine over the PUBLIC HTTPS endpoints (Caddy vhosts). The Bearer
# token is read locally from .state/ml_auth_token (persisted by provision.sh);
# compat_check runs ON the box over SSH (needs the installed `cartridges` env).
#
# Steps (each prints PASS/FAIL; any FAIL exits non-zero):
#   1. serve /health -> wait for engine_ready=true (weights seeded/loaded)
#   2. compat_check   -> python -m cartridges.serve.compat_check on the box (ssh)
#   3. onboard        -> POST https://gpu-onboard.../onboard_cag with ONE test doc
#   4. query          -> POST https://gpu.../query, assert a grounded answer + metrics
#   5. describe       -> POST https://gpu.../describe, assert a non-null description
#
# HOST_SERVE/HOST_ONBOARD override the hostnames; the token never leaves this box
# except as the Bearer header to the unit's own endpoints.
# ============================================================================
set -euo pipefail
SCRIPT_TAG="smoke"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

HOST_SERVE="${HOST_SERVE:-gpu.engramdynamics.org}"
HOST_ONBOARD="${HOST_ONBOARD:-gpu-onboard.engramdynamics.org}"
SERVE_URL="https://$HOST_SERVE"
ONBOARD_URL="https://$HOST_ONBOARD"
HEALTH_WAIT_MIN="${HEALTH_WAIT_MIN:-45}"   # first cold weight seed/load can be long
SSH_USER="ubuntu"
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KEYS_DIR/known_hosts")

[ -f "$TOKEN_FILE" ] || die "no token at $TOKEN_FILE — run provision.sh first"
TOKEN="$(cat "$TOKEN_FILE")"
AUTH="Authorization: Bearer $TOKEN"

pass=0; fail=0
step(){ # step "name" "0|1 ok" "detail"
  if [ "$2" = "1" ]; then echo "[smoke] PASS  $1${3:+ — $3}"; pass=$((pass+1));
  else echo "[smoke] FAIL  $1${3:+ — $3}"; fail=$((fail+1)); fi
}

# ---------------------------------------------------------------------------
# 1. Wait for :8002 /health engine_ready (auth-open route) over HTTPS.
# ---------------------------------------------------------------------------
log "waiting up to ${HEALTH_WAIT_MIN}m for engine_ready at $SERVE_URL/health"
deadline=$(( $(date +%s) + HEALTH_WAIT_MIN*60 ))
ready=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  body="$(curl -sf "$SERVE_URL/health" || true)"
  echo "$body" | grep -q '"engine_ready":true' && { ready=1; break; }
  sleep 15
done
[ "$ready" = "1" ] && step "serve /health engine_ready" 1 "model loaded" \
  || step "serve /health engine_ready" 0 "engine not ready within ${HEALTH_WAIT_MIN}m"

# ---------------------------------------------------------------------------
# 2. compat_check ON the box (static version-surface check; needs the venv). SSH
# is the only path with the installed cartridges package.
# ---------------------------------------------------------------------------
if [ -f "$STATE_FILE" ]; then IP="$(json_get 'd.get("ip","")' < "$STATE_FILE" 2>/dev/null || echo '')"; else IP=""; fi
if [ -n "$IP" ]; then
  C_OUT="$(ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" \
    'source /opt/engram/venv/bin/activate && python -m cartridges.serve.compat_check; echo "SMOKE_COMPAT_RC=$?"' 2>/dev/null || true)"
  echo "$C_OUT" | grep -q 'SMOKE_COMPAT_RC=0' \
    && step "cartridges compat_check (ssh)" 1 \
    || step "cartridges compat_check (ssh)" 0 "see box output above"
else
  step "cartridges compat_check (ssh)" 0 "no IP in $STATE_FILE"
fi

# ---------------------------------------------------------------------------
# 3. Onboard ONE tiny test doc via :8001 over HTTPS. The doc carries a
# distinctive fact the query answer must contain.
# ---------------------------------------------------------------------------
FACT="The Engram smoke-test mascot is a teal axolotl named Pixel."
CORPUS="/opt/engram/smoke_corpus"
# corpus_dir is created on the box by the onboarding worker; a stable path is fine.
ONB="$(curl -sf -X POST "$ONBOARD_URL/onboard_cag" \
  -H "$AUTH" -H 'Content-Type: application/json' \
  -d "$(python3 -c '
import json,sys
fact, corpus = sys.argv[1], sys.argv[2]
print(json.dumps({"corpus_dir":corpus,
                  "docs":[{"doc_id":"smoke-doc","text":"Engram Smoke Test\n\n"+fact}]}))' "$FACT" "$CORPUS")" || true)"
echo "$ONB" | grep -q '"n_cartridges"' \
  && step "onboard one test doc (:8001)" 1 \
  || step "onboard one test doc (:8001)" 0 "onboard did not return n_cartridges"

# ---------------------------------------------------------------------------
# 4. Query it via :8002 over HTTPS; assert a grounded answer + the metrics shape.
# ---------------------------------------------------------------------------
Q="$(curl -sf -X POST "$SERVE_URL/query" \
  -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"doc_ids":["smoke-doc"],"question":"What is the name and species of the Engram smoke-test mascot?","max_tokens":64}' || true)"
echo "$Q" | grep -qi 'axolotl' && echo "$Q" | grep -qi 'pixel' \
  && step "query grounded answer (:8002)" 1 "answer names the mascot fact" \
  || step "query grounded answer (:8002)" 0 "answer missing the doc fact"
echo "$Q" | grep -q '"resident_kv_tokens"' && echo "$Q" | grep -q '"prompt_tokens"' && echo "$Q" | grep -q '"measured":true' \
  && step "measured metrics shape" 1 "resident_kv_tokens + prompt_tokens + measured" \
  || step "measured metrics shape" 0 "metrics fields missing"

# ---------------------------------------------------------------------------
# 5. Describe it via :8002 over HTTPS; assert a non-null one-sentence description
# for the doc (the newer /describe endpoint the control plane uses for metadata).
# ---------------------------------------------------------------------------
D="$(curl -sf -X POST "$SERVE_URL/describe" \
  -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"doc_ids":["smoke-doc"],"max_tokens":60}' || true)"
# {"descriptions":{"smoke-doc":"...text..."}} — the value must be a non-empty string.
DESC_OK="$(printf '%s' "$D" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)["descriptions"]["smoke-doc"]
    print("1" if isinstance(d, str) and d.strip() else "0")
except Exception:
    print("0")' 2>/dev/null || echo 0)"
[ "$DESC_OK" = "1" ] \
  && step "describe returns a description (:8002)" 1 "non-null one-liner for the doc" \
  || step "describe returns a description (:8002)" 0 "descriptions[smoke-doc] was null/absent"

echo
log "==================== SMOKE SUMMARY ===================="
log "PASS=$pass  FAIL=$fail"
[ "$fail" -eq 0 ] && { log "ALL PASS — the Lambda serving unit answers grounded questions + describes carts over HTTPS."; exit 0; }
die "$fail smoke step(s) failed — see the FAIL lines above."
