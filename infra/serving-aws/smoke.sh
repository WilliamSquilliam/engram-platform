#!/usr/bin/env bash
# ============================================================================
# smoke.sh — end-to-end health check of the serving unit, from the operator's
# machine. It drives the box over SSM (the box has localhost access to :8001/:8002
# and the installed `cartridges` env), so no port-forward or in-VPC host is needed.
#
# Steps (each prints PASS/FAIL; any FAIL exits non-zero):
#   1. serve /health -> wait for engine_ready=true (the model finished loading)
#   2. compat_check   -> python -m cartridges.serve.compat_check on the box
#   3. onboard        -> POST :8001/onboard_cag with ONE tiny test doc
#   4. query          -> POST :8002/query with that doc_id, assert a grounded answer
#                        + the measured metrics shape (resident_kv_tokens / prompt_tokens)
#
# The Bearer token is read from /etc/engram/serving.env ON the box (never leaves it),
# so the operator machine needs no secret. Run after provision.sh.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROFILE="${AWS_PROFILE:-Engram-Dynamics}"
REGION="${AWS_REGION:-us-east-1}"
TF="terraform -chdir=$HERE"
AWS="aws --profile $PROFILE --region $REGION"
HEALTH_WAIT_MIN="${HEALTH_WAIT_MIN:-30}"   # how long to wait for the model to load + engine_ready

log(){ echo "[smoke] $*"; }
die(){ echo "[smoke] ERROR: $*" >&2; exit 1; }

INSTANCE_ID="$($TF output -raw instance_id 2>/dev/null)"
[ -n "$INSTANCE_ID" ] || die "no instance_id output — is the unit applied?"
log "target $INSTANCE_ID  (waiting up to ${HEALTH_WAIT_MIN}m for engine_ready)"

# Run a script ON the box via SSM and return its stdout. Fails loud on a non-Success
# SSM status so a broken box is never mistaken for a failed assertion. The script is
# passed as the `commands` parameter (AWS-RunShellScript joins the list with newlines).
run_on_box(){
  local script="$1" comment="$2"
  local cid
  cid="$($AWS ssm send-command --instance-ids "$INSTANCE_ID" \
    --document-name AWS-RunShellScript --comment "$comment" --timeout-seconds 1800 \
    --parameters commands="$script" \
    --query 'Command.CommandId' --output text)" || die "send-command failed ($comment)"
  local st=""
  for _ in $(seq 1 120); do
    sleep 5
    st="$($AWS ssm get-command-invocation --command-id "$cid" --instance-id "$INSTANCE_ID" \
      --query 'Status' --output text 2>/dev/null || echo Pending)"
    case "$st" in Success|Failed|Cancelled|TimedOut) break ;; esac
  done
  local out err
  out="$($AWS ssm get-command-invocation --command-id "$cid" --instance-id "$INSTANCE_ID" \
    --query 'StandardOutputContent' --output text 2>/dev/null || true)"
  err="$($AWS ssm get-command-invocation --command-id "$cid" --instance-id "$INSTANCE_ID" \
    --query 'StandardErrorContent' --output text 2>/dev/null || true)"
  printf '%s\n' "$out"
  [ "$st" = "Success" ] || { echo "$err" >&2; die "on-box step '$comment' SSM status $st"; }
}

pass=0; fail=0
step(){ # step "name" "0|1 ok" "detail"
  if [ "$2" = "1" ]; then echo "[smoke] PASS  $1${3:+ — $3}"; pass=$((pass+1));
  else echo "[smoke] FAIL  $1${3:+ — $3}"; fail=$((fail+1)); fi
}

# ---------------------------------------------------------------------------
# 1. Wait for :8002 /health engine_ready. The health route is auth-open, so a plain
# curl works; we poll until engine_ready flips true or the timeout elapses.
# ---------------------------------------------------------------------------
HEALTH_SCRIPT="$(cat <<EOS
deadline=\$(( \$(date +%s) + $HEALTH_WAIT_MIN*60 ))
while [ \$(date +%s) -lt \$deadline ]; do
  body=\$(curl -sf http://localhost:8002/health || true)
  echo "\$body"
  echo "\$body" | grep -q '"engine_ready":true' && { echo "SMOKE_ENGINE_READY"; exit 0; }
  sleep 15
done
echo SMOKE_ENGINE_NOT_READY
EOS
)"
H_OUT="$(run_on_box "$HEALTH_SCRIPT" "smoke: wait engine_ready")"
echo "$H_OUT" | grep -q SMOKE_ENGINE_READY \
  && step "serve /health engine_ready" 1 "model loaded" \
  || { step "serve /health engine_ready" 0 "engine not ready within ${HEALTH_WAIT_MIN}m"; }

# ---------------------------------------------------------------------------
# 2. compat_check on the box (static version-surface check; no GPU needed).
# ---------------------------------------------------------------------------
COMPAT_SCRIPT='source /opt/engram/venv/bin/activate && python -m cartridges.serve.compat_check; echo "SMOKE_COMPAT_RC=$?"'
C_OUT="$(run_on_box "$COMPAT_SCRIPT" "smoke: compat_check")"
echo "$C_OUT" | grep -q 'SMOKE_COMPAT_RC=0' \
  && step "cartridges compat_check" 1 \
  || step "cartridges compat_check" 0 "see FAIL lines above"

# ---------------------------------------------------------------------------
# 3+4. Onboard ONE tiny test doc via :8001, then query it via :8002 and assert a
# grounded answer + the metrics shape. Token is read from /etc/engram/serving.env
# on the box. The doc carries a distinctive fact the answer must contain.
# ---------------------------------------------------------------------------
E2E_SCRIPT="$(cat <<'EOS'
set -e
. /etc/engram/serving.env
AUTH="Authorization: Bearer ${ML_AUTH_TOKEN}"
CORPUS=/opt/engram/work/smoke_corpus
mkdir -p "$CORPUS"
FACT="The Engram smoke-test mascot is a teal axolotl named Pixel."
DOC='{"doc_id":"smoke-doc","text":"Engram Smoke Test\n\n'"$FACT"'"}'

echo "== onboard =="
ONB=$(curl -sf -X POST http://localhost:8001/onboard_cag \
  -H "$AUTH" -H 'Content-Type: application/json' \
  -d "{\"corpus_dir\":\"$CORPUS\",\"docs\":[$DOC]}") || { echo "SMOKE_ONBOARD_FAIL"; exit 0; }
echo "$ONB"
echo "$ONB" | grep -q '"n_cartridges"' && echo SMOKE_ONBOARD_OK || echo SMOKE_ONBOARD_FAIL

echo "== query =="
Q=$(curl -sf -X POST http://localhost:8002/query \
  -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"doc_ids":["smoke-doc"],"question":"What is the name and species of the Engram smoke-test mascot?","max_tokens":64}') \
  || { echo "SMOKE_QUERY_FAIL"; exit 0; }
echo "$Q"
# Grounded-answer assertion: the distinctive fact tokens must appear in the answer.
echo "$Q" | grep -qi 'axolotl' && echo "$Q" | grep -qi 'pixel' && echo SMOKE_ANSWER_GROUNDED || echo SMOKE_ANSWER_UNGROUNDED
# Metrics-shape assertion: the measured serve metrics must be present.
echo "$Q" | grep -q '"resident_kv_tokens"' && echo "$Q" | grep -q '"prompt_tokens"' && echo "$Q" | grep -q '"measured":true' \
  && echo SMOKE_METRICS_OK || echo SMOKE_METRICS_MISSING
EOS
)"
E_OUT="$(run_on_box "$E2E_SCRIPT" "smoke: onboard+query")"
echo "$E_OUT"
echo "$E_OUT" | grep -q SMOKE_ONBOARD_OK       && step "onboard one test doc (:8001)" 1 || step "onboard one test doc (:8001)" 0
echo "$E_OUT" | grep -q SMOKE_ANSWER_GROUNDED  && step "query grounded answer (:8002)" 1 "answer names the mascot fact" || step "query grounded answer (:8002)" 0 "answer missing the doc fact"
echo "$E_OUT" | grep -q SMOKE_METRICS_OK       && step "measured metrics shape" 1 "resident_kv_tokens + prompt_tokens + measured" || step "measured metrics shape" 0

echo
log "==================== SMOKE SUMMARY ===================="
log "PASS=$pass  FAIL=$fail"
[ "$fail" -eq 0 ] && { log "ALL PASS — the serving unit answers grounded questions from onboarded carts."; exit 0; }
die "$fail smoke step(s) failed — see the FAIL lines above."
