#!/usr/bin/env bash
# ============================================================================
# emit-env.sh — print the four contract values the control plane consumes, plus a
# ready-to-paste envs/uat tfvars snippet for the platform-aws stack.
#
# The Lambda unit is reached over PUBLIC HTTPS (Caddy vhosts), so the two URLs are
# the hostnames, not a private IP:
#   ML_SERVICE_URL        = https://gpu-onboard.engramdynamics.org   (:8001 onboarding)
#   INFERENCE_SERVICE_URL = https://gpu.engramdynamics.org           (:8002 vLLM serve)
#   ML_AUTH_TOKEN         = the persisted shared bearer (.state/ml_auth_token)
#   MODEL_REGISTRY_JSON   = one enabled "best" tier -> Command A+ W4A4
#
# The token is a secret: printed ONLY with --show-secrets. Without the flag the
# token line is a placeholder pointing at the file (so a plain run never leaks it).
#
# The MODEL_REGISTRY_JSON shape matches backend/app/serving.py `_load_registry`
# (ModelTier: id/label/description/model_ref/precision/context_tokens/enabled).
# ============================================================================
set -euo pipefail
SCRIPT_TAG="emit-env"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

SHOW_SECRETS=0
[ "${1:-}" = "--show-secrets" ] && SHOW_SECRETS=1

HOST_SERVE="${HOST_SERVE:-gpu.engramdynamics.org}"
HOST_ONBOARD="${HOST_ONBOARD:-gpu-onboard.engramdynamics.org}"
CARTRIDGES_MODEL="${CARTRIDGES_MODEL:-CohereLabs/command-a-plus-05-2026-w4a4}"
MODEL_PRECISION="${MODEL_PRECISION:-w4a4}"
CONTEXT_TOKENS="${CONTEXT_TOKENS:-131072}"

ML_SERVICE_URL="https://$HOST_ONBOARD"
INFERENCE_SERVICE_URL="https://$HOST_SERVE"

# --- token (from .state, persisted by provision.sh) -------------------------
if [ -f "$TOKEN_FILE" ]; then
  TOKEN_VALUE="$(cat "$TOKEN_FILE")"
else
  TOKEN_VALUE=""
fi
if [ "$SHOW_SECRETS" = "1" ]; then
  [ -n "$TOKEN_VALUE" ] || die "no token at $TOKEN_FILE — run provision.sh first"
  TOKEN_OUT="$TOKEN_VALUE"
else
  TOKEN_OUT="__read_from: bash $LIB_DIR/emit-env.sh --show-secrets  (or cat $TOKEN_FILE)__"
fi

# --- cart bucket (from aws-support, if applied) -----------------------------
CART_BUCKET="$(terraform -chdir="$UNIT_DIR/aws-support" output -raw cart_bucket 2>/dev/null || true)"
CART_PREFIX="$(terraform -chdir="$UNIT_DIR/aws-support" output -raw cart_store_prefix 2>/dev/null || echo cartridges)"

# --- MODEL_REGISTRY_JSON — one enabled "best" tier. Built in python so it's exact
# and compact; shape matches serving.py ModelTier. ---------------------------
REGISTRY_JSON="$(python3 -c '
import json, sys
model_ref, precision, ctx = sys.argv[1], sys.argv[2], int(sys.argv[3])
print(json.dumps([{
    "id": "best",
    "label": "Best",
    "description": "Highest grounded fact-retrieval accuracy.",
    "model_ref": model_ref,
    "precision": precision,
    "context_tokens": ctx,
    "enabled": True,
}], separators=(",", ":")))' "$CARTRIDGES_MODEL" "$MODEL_PRECISION" "$CONTEXT_TOKENS")"

# ---------------------------------------------------------------------------
# 1. The four contract values as an env block.
# ---------------------------------------------------------------------------
cat <<ENV
# --- Engram serving unit (Lambda Cloud, $CARTRIDGES_MODEL) — the four contract values ---
INFERENCE_BACKEND=vllm
ML_SERVICE_URL=$ML_SERVICE_URL
INFERENCE_SERVICE_URL=$INFERENCE_SERVICE_URL
ML_AUTH_TOKEN=$TOKEN_OUT
MODEL_REGISTRY_JSON=$REGISTRY_JSON
# --- cartridge store the unit reads/writes (control plane shares it for retrieval hydration) ---
CARTRIDGE_STORE_BACKEND=s3
CARTRIDGE_STORE_BUCKET=${CART_BUCKET:-<apply aws-support/ then re-run>}
CARTRIDGE_STORE_PREFIX=$CART_PREFIX
# --- measured \$/query uses the real running price of this box ---
# box on-demand cost ~= \$6.99/hr (1x B200)
ENV

# ---------------------------------------------------------------------------
# 2. Ready-to-paste envs/uat tfvars snippet. Uses the *_override vars +
# read_serving_state=false (the platform-aws module skips the serving remote-state
# read and uses these instead — no serving-lambda terraform state to read from,
# since the compute is on Lambda, not AWS).
# ---------------------------------------------------------------------------
if [ "$SHOW_SECRETS" = "1" ]; then
  TFVARS_TOKEN="$TOKEN_VALUE"
else
  TFVARS_TOKEN="__paste from: bash $LIB_DIR/emit-env.sh --show-secrets__"
fi

cat <<TFVARS

# ============================================================================
# PASTE INTO infra/platform-aws/envs/uat/<name>.tfvars (gitignored)
# Points UAT at this Lambda serving unit via overrides. read_serving_state=false
# because the Lambda unit keeps NO terraform serving-state to read (compute is on
# Lambda); the four values are supplied directly here.
# ============================================================================
read_serving_state             = false
ml_service_url_override         = "$ML_SERVICE_URL"
inference_service_url_override  = "$INFERENCE_SERVICE_URL"
ml_auth_token_override          = "$TFVARS_TOKEN"
model_registry_json_override    = "$(printf '%s' "$REGISTRY_JSON" | sed 's/"/\\"/g')"
# Optional: share the Lambda cart bucket for retrieval hydration.
# cartridge_store_bucket_override = "${CART_BUCKET:-<apply aws-support/>}"
TFVARS
