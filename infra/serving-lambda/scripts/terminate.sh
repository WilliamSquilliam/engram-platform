#!/usr/bin/env bash
# ============================================================================
# terminate.sh — terminate the current Lambda GPU box (reads .state/current.json).
#
# Lambda has NO stop state: terminate is the only way to stop paying compute. When
# you terminate:
#   * the local SSD root DIES (HF live cache, cart hot-mirror, registry) — but
#     that's the loss-tolerant tier by design;
#   * the persistent filesystem SURVIVES (region-bound) with the seeded HF weights,
#     so the next launch.sh + provision.sh relaunches in minutes, not the ~120GB
#     download;
#   * the S3 cart store is untouched (durable), so onboarded memory persists;
#   * .state/ml_auth_token is kept so the platform env values don't churn.
#
# Idle economics: terminated = $0 compute. You keep paying only the persistent FS
# (~$0.20/GB/mo) and S3 (pennies/GB-mo). NEVER prints LAMBDA_API_KEY.
# ============================================================================
set -euo pipefail
SCRIPT_TAG="terminate"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

require_api_key
[ -f "$STATE_FILE" ] || die "no $STATE_FILE — nothing to terminate (already gone?)"

INSTANCE_ID="$(json_get 'd["id"]' < "$STATE_FILE")"
IP="$(json_get 'd.get("ip","")' < "$STATE_FILE" 2>/dev/null || echo '')"
FS_NAME="$(json_get 'd.get("fs_name","engram-fs")' < "$STATE_FILE" 2>/dev/null || echo engram-fs)"
[ -n "$INSTANCE_ID" ] || die "no instance id in $STATE_FILE"

log "terminating instance $INSTANCE_ID (${IP:-no-ip})"
BODY="$(python3 -c 'import json,sys;print(json.dumps({"instance_ids":[sys.argv[1]]}))' "$INSTANCE_ID")"
lambda_api POST /instance-operations/terminate "$BODY" >/dev/null
log "terminate requested."

# Drop the instance id/ip from state (keep the token + fs name). Next launch writes fresh.
python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    d = {}
d.pop("id", None); d.pop("ip", None)
json.dump(d, open(sys.argv[1], "w"), indent=2)' "$STATE_FILE" 2>/dev/null || rm -f "$STATE_FILE"

echo
log "==================== TERMINATED ===================="
log "Local SSD (HF live cache / cart mirror / registry) is GONE — the loss-tolerant tier."
log "Persistent FS '$FS_NAME' KEPT the seeded weights; S3 cart store KEPT your memory."
log "Compute is now \$0. Relaunch:  bash $LIB_DIR/launch.sh  &&  bash $LIB_DIR/provision.sh  (minutes)."
log "If DNS was auto-set, the two A records now point at a dead IP — re-run launch.sh to refresh them."
