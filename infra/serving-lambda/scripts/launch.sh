#!/usr/bin/env bash
# ============================================================================
# launch.sh — stand up the Lambda Cloud GPU box for the serving unit.
#
# Runs from the OPERATOR'S machine. It does the CLOUD-SIDE bring-up only (venv +
# services come later, in provision.sh -> bootstrap-lambda.sh). Steps:
#   1. ensure a local ed25519 SSH keypair (.keys/, gitignored) and register the
#      public key with Lambda (/ssh-keys) if not already there
#   2. resolve the instance type by DISCOVERY: list /instance-types and pick the
#      first whose name matches $INSTANCE_TYPE_FILTER (default b200), else the
#      h100_sxm fallback, that currently HAS capacity somewhere
#   3. pick a region with capacity for that type ($REGION overrides)
#   4. ensure the persistent filesystem 'engram-fs' exists in that region
#      (survives terminate; holds the seeded HF weights so relaunches are fast)
#   5. optionally set the account firewall to 22/80/443 only (MANAGE_FIREWALL=1)
#   6. launch with the filesystem attached, poll until active + IP
#   7. upsert the two Cloudflare A records if CLOUDFLARE_* are in .env; else print
#      the manual DNS steps
#   8. save {id, ip, region, instance_type} to .state/current.json
#
# It does NOT install anything on the box and NEVER prints LAMBDA_API_KEY.
# ============================================================================
set -euo pipefail
SCRIPT_TAG="launch"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

require_api_key
mkdirs

INSTANCE_TYPE_FILTER="${INSTANCE_TYPE_FILTER:-b200}"     # primary target substring
FALLBACK_FILTER="${FALLBACK_FILTER:-2x_h100_sxm}"        # 2x H100 SXM fallback — MUST be
# the 2x (160GB): a 1x H100 (80GB) cannot hold Command A+ W4A4 (~120GB weights);
# an under-matched filter launched exactly that once (caught + terminated).
FS_NAME="${FS_NAME:-engram-fs}"                          # persistent filesystem (region-bound)
INSTANCE_NAME="${INSTANCE_NAME:-engram-serving}"

# Two hostnames the box serves behind (Caddy vhosts). launch prints/updates their
# A records; provision + smoke assume these resolve to the box.
HOST_SERVE="${HOST_SERVE:-gpu.engramdynamics.org}"
HOST_ONBOARD="${HOST_ONBOARD:-gpu-onboard.engramdynamics.org}"

# ---------------------------------------------------------------------------
# 1. SSH keypair — generate ed25519 if absent, register the public key with Lambda.
# ---------------------------------------------------------------------------
if [ ! -f "$SSH_KEY" ]; then
  log "generating ed25519 SSH keypair at $SSH_KEY (gitignored)"
  ssh-keygen -t ed25519 -N '' -C "engram-lambda-serving" -f "$SSH_KEY" >/dev/null
  chmod 600 "$SSH_KEY"
fi
PUBKEY="$(cat "$SSH_KEY.pub")"

log "ensuring SSH key '$SSH_KEY_NAME' is registered with Lambda"
KEYS_JSON="$(lambda_api GET /ssh-keys)"
if printf '%s' "$KEYS_JSON" | json_get "any(k['name']=='$SSH_KEY_NAME' for k in d['data'])" | grep -qi true; then
  log "  key already registered"
else
  # POST the public key. If Lambda already has this exact key under another name it
  # errors — fine, the operator can set SSH_KEY_NAME to the existing one.
  BODY="$(python3 -c 'import json,sys; print(json.dumps({"name":sys.argv[1],"public_key":sys.argv[2]}))' "$SSH_KEY_NAME" "$PUBKEY")"
  lambda_api POST /ssh-keys "$BODY" >/dev/null
  log "  registered public key as '$SSH_KEY_NAME'"
fi

# ---------------------------------------------------------------------------
# 2. Resolve the instance type by DISCOVERY (don't hardcode gpu_1x_b200_sxm6).
# /instance-types returns a map keyed by type name; each carries
# regions_with_capacity_available. Prefer a $INSTANCE_TYPE_FILTER match with live
# capacity; fall back to $FALLBACK_FILTER.
# ---------------------------------------------------------------------------
log "discovering instance types (filter='$INSTANCE_TYPE_FILTER', fallback='$FALLBACK_FILTER')"
TYPES_JSON="$(lambda_api GET /instance-types)"

# Returns "<type_name>\t<region_name>" for the first type matching a substring that
# has capacity, or empty. python does the pick so region/type stay consistent.
resolve_type_region(){
  local filter="$1"
  printf '%s' "$TYPES_JSON" | python3 -c '
import sys, json
d = json.load(sys.stdin); flt = sys.argv[1]
data = d.get("data", d)
for name, info in sorted(data.items()):
    if flt not in name:
        continue
    regs = (info.get("regions_with_capacity_available")
            or (info.get("instance_type", {}) or {}).get("regions_with_capacity_available")
            or [])
    names = [r["name"] if isinstance(r, dict) else r for r in regs]
    if names:
        print(f"{name}\t{names[0]}")
        break
' "$filter"
}

PICK="$(resolve_type_region "$INSTANCE_TYPE_FILTER" || true)"
if [ -z "$PICK" ]; then
  log "no '$INSTANCE_TYPE_FILTER' capacity right now — trying fallback '$FALLBACK_FILTER'"
  PICK="$(resolve_type_region "$FALLBACK_FILTER" || true)"
fi
[ -n "$PICK" ] || die "no capacity for '$INSTANCE_TYPE_FILTER' or '$FALLBACK_FILTER' in any region right now — retry later or widen INSTANCE_TYPE_FILTER."

INSTANCE_TYPE="$(printf '%s' "$PICK" | cut -f1)"
DISCOVERED_REGION="$(printf '%s' "$PICK" | cut -f2)"
# $REGION var overrides the discovered region (operator preference) — but only if
# that region actually appears; otherwise we keep the one with capacity.
REGION_NAME="${REGION:-$DISCOVERED_REGION}"
log "resolved instance_type=$INSTANCE_TYPE region=$REGION_NAME"

# ---------------------------------------------------------------------------
# 3. Ensure the persistent filesystem exists IN THAT REGION. Filesystems are
# region-bound and survive terminate — this is where the ~120GB HF weights are
# seeded so relaunches skip the download. Create if absent.
# ---------------------------------------------------------------------------
log "ensuring persistent filesystem '$FS_NAME' in $REGION_NAME"
FS_JSON="$(lambda_api GET /file-systems)"
FS_IN_REGION="$(printf '%s' "$FS_JSON" | python3 -c '
import sys, json
d = json.load(sys.stdin); name, region = sys.argv[1], sys.argv[2]
for fs in d.get("data", []):
    r = fs.get("region")
    rn = r.get("name") if isinstance(r, dict) else r
    if fs.get("name") == name and rn == region:
        print("yes"); break
' "$FS_NAME" "$REGION_NAME" || true)"
if [ "$FS_IN_REGION" = "yes" ]; then
  log "  filesystem exists in region"
else
  log "  creating filesystem '$FS_NAME' in $REGION_NAME"
  FS_BODY="$(python3 -c 'import json,sys;print(json.dumps({"name":sys.argv[1],"region":sys.argv[2]}))' "$FS_NAME" "$REGION_NAME")"
  # Endpoint is POST /filesystems (note: no hyphen) per the Lambda API.
  lambda_api POST /filesystems "$FS_BODY" >/dev/null
  log "  created"
fi

# ---------------------------------------------------------------------------
# 4. (optional) Firewall. Rulesets are ACCOUNT-GLOBAL on Lambda, so changing them
# affects every instance — gate behind MANAGE_FIREWALL=1. We allow only 22/80/443
# (SSH + Caddy's HTTP-01 + HTTPS); the serving ports 8001/8002 bind to 127.0.0.1
# so they are never exposed regardless.
# ---------------------------------------------------------------------------
if [ "${MANAGE_FIREWALL:-0}" = "1" ]; then
  log "MANAGE_FIREWALL=1 — setting account firewall to 22/80/443 inbound only (GLOBAL)"
  FW_BODY="$(python3 -c '
import json
rule = lambda proto,port,desc: {"protocol":proto,"port_range":[port,port],
                                 "source_network":"0.0.0.0/0","description":desc}
print(json.dumps({"data":[rule("tcp",22,"ssh"),
                          rule("tcp",80,"http-01 + redirect"),
                          rule("tcp",443,"https (Caddy)")]}))')"
  lambda_api PUT /firewall-rules "$FW_BODY" >/dev/null || log "  WARN firewall PUT failed (continuing; set rules in the console)"
else
  log "skipping firewall (set MANAGE_FIREWALL=1 to enforce 22/80/443-only — it is account-GLOBAL)"
fi

# ---------------------------------------------------------------------------
# 5. Launch. Attach the filesystem so the box mounts it at /lambda/nfs/<name>.
# ---------------------------------------------------------------------------
log "launching $INSTANCE_TYPE in $REGION_NAME (name=$INSTANCE_NAME, fs=$FS_NAME)"
LAUNCH_BODY="$(python3 -c '
import json, sys
print(json.dumps({
  "region_name": sys.argv[1],
  "instance_type_name": sys.argv[2],
  "ssh_key_names": [sys.argv[3]],
  "file_system_names": [sys.argv[4]],
  "name": sys.argv[5],
}))' "$REGION_NAME" "$INSTANCE_TYPE" "$SSH_KEY_NAME" "$FS_NAME" "$INSTANCE_NAME")"
LAUNCH_RES="$(lambda_api POST /instance-operations/launch "$LAUNCH_BODY")"
INSTANCE_ID="$(printf '%s' "$LAUNCH_RES" | json_get 'd["data"]["instance_ids"][0]')"
[ -n "$INSTANCE_ID" ] || die "launch returned no instance id"
log "launched instance $INSTANCE_ID — polling for active + IP"

# ---------------------------------------------------------------------------
# 6. Poll /instances/{id} until active with a public IP.
# ---------------------------------------------------------------------------
IP=""
for _ in $(seq 1 60); do   # up to ~15 min
  INST="$(lambda_api GET "/instances/$INSTANCE_ID" || true)"
  STATUS="$(printf '%s' "$INST" | json_get 'd["data"].get("status","")' 2>/dev/null || echo '')"
  IP="$(printf '%s' "$INST" | json_get 'd["data"].get("ip") or ""' 2>/dev/null || echo '')"
  log "  status=${STATUS:-?} ip=${IP:-pending}"
  if [ "$STATUS" = "active" ] && [ -n "$IP" ]; then break; fi
  sleep 15
done
[ -n "$IP" ] || die "instance $INSTANCE_ID never reached active+IP (check the Lambda console)"

# ---------------------------------------------------------------------------
# 7. Save state (gitignored) so provision/smoke/terminate find the box.
# ---------------------------------------------------------------------------
python3 -c '
import json, sys
json.dump({"id":sys.argv[1],"ip":sys.argv[2],"region":sys.argv[3],
           "instance_type":sys.argv[4],"fs_name":sys.argv[5]},
          open(sys.argv[6],"w"), indent=2)' \
  "$INSTANCE_ID" "$IP" "$REGION_NAME" "$INSTANCE_TYPE" "$FS_NAME" "$STATE_FILE"
log "saved $STATE_FILE"

# ---------------------------------------------------------------------------
# 8. DNS. Both hostnames must A-record to $IP (DNS-only, no proxy — Caddy needs
# HTTP-01 reachability). Auto-upsert via Cloudflare if the token+zone are in .env;
# else print the exact records to add.
# ---------------------------------------------------------------------------
cf_upsert(){
  local host="$1" ip="$2"
  local base="https://api.cloudflare.com/client/v4/zones/$CLOUDFLARE_ZONE_ID/dns_records"
  local auth="Authorization: Bearer $CLOUDFLARE_API_TOKEN"
  # Find an existing A record for this host.
  local rid
  rid="$(curl -sS -H "$auth" "$base?type=A&name=$host" \
         | python3 -c 'import sys,json;r=json.load(sys.stdin)["result"];print(r[0]["id"] if r else "")')"
  local body
  body="$(python3 -c 'import json,sys;print(json.dumps({"type":"A","name":sys.argv[1],"content":sys.argv[2],"ttl":120,"proxied":False}))' "$host" "$ip")"
  if [ -n "$rid" ]; then
    curl -sS -X PUT -H "$auth" -H 'Content-Type: application/json' -d "$body" "$base/$rid" >/dev/null
  else
    curl -sS -X POST -H "$auth" -H 'Content-Type: application/json' -d "$body" "$base" >/dev/null
  fi
  log "  Cloudflare A $host -> $ip (DNS-only)"
}

if [ -n "${CLOUDFLARE_API_TOKEN:-}" ] && [ -n "${CLOUDFLARE_ZONE_ID:-}" ]; then
  log "upserting Cloudflare A records (DNS-only)"
  cf_upsert "$HOST_SERVE" "$IP"
  cf_upsert "$HOST_ONBOARD" "$IP"
else
  log "no CLOUDFLARE_* in .env — add these two A records MANUALLY (DNS-only, not proxied):"
  echo "    A  $HOST_SERVE     -> $IP"
  echo "    A  $HOST_ONBOARD   -> $IP"
fi

echo
log "==================== LAUNCHED ===================="
log "instance : $INSTANCE_ID  ($INSTANCE_TYPE, $REGION_NAME)"
log "public IP: $IP"
log "next     : point DNS (above) then  bash $LIB_DIR/provision.sh"
log "reminder : the root SSD is EPHEMERAL (dies on terminate); weights persist on FS '$FS_NAME'."
