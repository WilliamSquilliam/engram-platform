#!/usr/bin/env bash
# ============================================================================
# provision.sh — runs from the OPERATOR'S machine after `terraform apply`.
# Builds the install bundle and hands it to the GPU box over SSM. Idempotent:
# safe to re-run (it rebuilds the wheel, re-uploads, and re-runs bootstrap, which
# upgrades the venv/env/units in place — the model on the box's EBS is not touched).
#
# Steps:
#   1. terraform output -> instance id, buckets, model/TP/token (the four contract inputs)
#   2. build the engram-cartridge wheel from the sibling repo (python -m build)
#   3. bundle: wheel + this repo's ml_service/ + scripts/bootstrap.sh -> a .tgz
#   4. upload the bundle to the provisioning S3 bucket
#   5. SSM send-command -> run bootstrap.sh on the box (installs venv + systemd units)
#   6. poll the SSM command to completion and print the box-side log tail
#
# Prereqs: terraform, aws CLI (profile Engram-Dynamics), python3 with `build`
# (pip install build) available for the sibling repo, and `terraform init` already
# run in this module dir.
#
# Favor established tooling: SSM send-command (no SSH), systemd on the box,
# `python -m build` for the wheel — nothing bespoke.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PLATFORM_REPO="$(cd "$HERE/../.." && pwd)"                 # engram-platform (this repo)
SIBLING_REPO="${SIBLING_REPO:-$(cd "$PLATFORM_REPO/../Engram-Smart-CAG" && pwd)}"  # the cartridge package
PROFILE="${AWS_PROFILE:-Engram-Dynamics}"
REGION="${AWS_REGION:-us-east-1}"
TF="terraform -chdir=$HERE"
AWS="aws --profile $PROFILE --region $REGION"

log(){ echo "[provision] $*"; }
die(){ echo "[provision] ERROR: $*" >&2; exit 1; }

command -v terraform >/dev/null || die "terraform not on PATH"
command -v aws >/dev/null || die "aws CLI not on PATH"
command -v python3 >/dev/null || die "python3 not on PATH (needed to build the wheel)"
[ -d "$SIBLING_REPO" ] || die "sibling repo not found at $SIBLING_REPO (set SIBLING_REPO=/path/to/Engram-Smart-CAG)"

# --- 1. terraform outputs (the provisioning inputs) -------------------------
tf(){ $TF output -raw "$1" 2>/dev/null; }
INSTANCE_ID="$(tf instance_id)";        [ -n "$INSTANCE_ID" ] || die "no instance_id output — run terraform apply first"
CART_BUCKET="$(tf cart_bucket)";        [ -n "$CART_BUCKET" ] || die "no cart_bucket output"
PROVISION_BUCKET="$(tf provision_bucket)"; [ -n "$PROVISION_BUCKET" ] || die "no provision_bucket output"
CART_PREFIX="$(tf cart_store_prefix)"
ML_AUTH_TOKEN="$(tf ml_auth_token)";    [ -n "$ML_AUTH_TOKEN" ] || die "no ml_auth_token output"
# model_ref / tensor_parallel / context_tokens come from the tfvars; pull them off the
# rendered registry so provision.sh and the module never disagree about what's served.
REGISTRY_JSON="$(tf model_registry_json)"
MODEL_REF="$(printf '%s' "$REGISTRY_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)[0]["model_ref"])')"
CONTEXT_TOKENS="$(printf '%s' "$REGISTRY_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)[0]["context_tokens"])')"
# tensor_parallel is a dedicated output (a serving detail, not in the registry).
VLLM_TP="$(tf tensor_parallel)"; [ -n "$VLLM_TP" ] || VLLM_TP=4

log "instance=$INSTANCE_ID model=$MODEL_REF tp=$VLLM_TP ctx=$CONTEXT_TOKENS"
log "cart_bucket=$CART_BUCKET provision_bucket=$PROVISION_BUCKET"

# --- 2. build the engram-cartridge wheel from the sibling repo --------------
log "building engram-cartridge wheel from $SIBLING_REPO (python -m build)"
BUILD_TMP="$(mktemp -d)"
( cd "$SIBLING_REPO" && python3 -m build --wheel --outdir "$BUILD_TMP/wheels" ) \
  || die "wheel build failed — is 'build' installed?  pip install build"
WHEEL="$(ls "$BUILD_TMP"/wheels/engram_cartridge-*.whl 2>/dev/null | head -1 || true)"
[ -n "$WHEEL" ] || die "no wheel produced under $BUILD_TMP/wheels"
log "wheel: $(basename "$WHEEL")"

# --- 3. bundle: wheel + ml_service/ + bootstrap.sh --------------------------
STAGE="$BUILD_TMP/stage"
mkdir -p "$STAGE/wheels"
cp "$WHEEL" "$STAGE/wheels/"
# Copy the platform's ml_service (the two GPU processes + requirements). Exclude
# caches so the bundle stays small.
mkdir -p "$STAGE/ml_service"
tar -C "$PLATFORM_REPO" --exclude='__pycache__' --exclude='*.pyc' -cf - ml_service \
  | tar -C "$STAGE" -xf -
cp "$HERE/scripts/bootstrap.sh" "$STAGE/bootstrap.sh"
chmod +x "$STAGE/bootstrap.sh"

TS="$(date +%Y%m%d-%H%M%S)"
BUNDLE="$BUILD_TMP/serving-bundle-$TS.tgz"
tar -C "$STAGE" -czf "$BUNDLE" .
log "bundle: $(basename "$BUNDLE") ($(du -h "$BUNDLE" | cut -f1))"

# --- 4. upload the bundle ---------------------------------------------------
BUNDLE_KEY="bundles/serving-bundle-$TS.tgz"
BUNDLE_S3_URI="s3://$PROVISION_BUCKET/$BUNDLE_KEY"
log "uploading bundle -> $BUNDLE_S3_URI"
$AWS s3 cp "$BUNDLE" "$BUNDLE_S3_URI" >/dev/null || die "bundle upload failed"

# --- 5. SSM send-command: run bootstrap.sh on the box -----------------------
# The bootstrap downloads its OWN bundle (which contains a copy of bootstrap.sh),
# then runs the copy. We pass the runtime inputs as exported env vars in the command
# so the box needs no terraform. The bundle it pulls is the one we just uploaded.
log "waiting for SSM to report the instance online"
for _ in $(seq 1 40); do
  ping="$($AWS ssm describe-instance-information \
    --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
    --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || true)"
  [ "$ping" = "Online" ] && { log "SSM online"; break; }
  log "SSM: ${ping:-pending} …"; sleep 15
done
[ "${ping:-}" = "Online" ] || die "instance $INSTANCE_ID not SSM-online (agent/role/network?)"

# Build the remote command. `set -a` exports every var the here-doc sets so bootstrap.sh sees them.
read -r -d '' REMOTE <<REMOTE || true
set -a
export MODEL_REF='$MODEL_REF'
export VLLM_TP='$VLLM_TP'
export ML_AUTH_TOKEN='$ML_AUTH_TOKEN'
export CART_BUCKET='$CART_BUCKET'
export CART_STORE_PREFIX='$CART_PREFIX'
export CONTEXT_TOKENS='$CONTEXT_TOKENS'
export AWS_REGION='$REGION'
export BUNDLE_S3_URI='$BUNDLE_S3_URI'
set +a
cd /tmp
aws s3 cp "\$BUNDLE_S3_URI" /tmp/bundle.tgz --region '$REGION'
mkdir -p /tmp/bootstrap && tar -xzf /tmp/bundle.tgz -C /tmp/bootstrap bootstrap.sh
bash /tmp/bootstrap/bootstrap.sh
REMOTE

log "SSM send-command: running bootstrap on $INSTANCE_ID"
# jq-free: hand the command to SSM as a JSON parameter file so newlines survive intact.
PARAM_FILE="$BUILD_TMP/ssm-params.json"
python3 - "$PARAM_FILE" <<PY
import json, sys
cmd = """$REMOTE"""
json.dump({"commands": [cmd]}, open(sys.argv[1], "w"))
PY
CID="$($AWS ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name AWS-RunShellScript \
  --comment "engram serving-unit bootstrap $TS" \
  --timeout-seconds 3600 \
  --cli-input-json "$(python3 -c 'import json,sys;print(json.dumps({"Parameters":json.load(open(sys.argv[1]))}))' "$PARAM_FILE")" \
  --query 'Command.CommandId' --output text)" || die "send-command failed"
log "SSM command id: $CID  (installing venv — this takes several minutes)"

# --- 6. poll to completion --------------------------------------------------
status=""
for _ in $(seq 1 120); do   # up to ~60 min (venv build + first pip installs are heavy)
  sleep 30
  status="$($AWS ssm get-command-invocation --command-id "$CID" --instance-id "$INSTANCE_ID" \
    --query 'Status' --output text 2>/dev/null || echo Pending)"
  log "bootstrap: $status"
  case "$status" in
    Success) break ;;
    Failed|Cancelled|TimedOut)
      log "bootstrap $status — stderr tail:"
      $AWS ssm get-command-invocation --command-id "$CID" --instance-id "$INSTANCE_ID" \
        --query 'StandardErrorContent' --output text 2>/dev/null | tail -40 || true
      die "bootstrap did not succeed" ;;
  esac
done

log "stdout tail:"
$AWS ssm get-command-invocation --command-id "$CID" --instance-id "$INSTANCE_ID" \
  --query 'StandardOutputContent' --output text 2>/dev/null | tail -25 || true

rm -rf "$BUILD_TMP"
[ "$status" = "Success" ] || die "bootstrap ended in status $status"
log "PROVISIONED. The serve engine is downloading the model on first start."
log "Next: bash $HERE/smoke.sh   (waits for engine_ready, then onboards + queries one test doc)"
