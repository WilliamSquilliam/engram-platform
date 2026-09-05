#!/usr/bin/env bash
# ============================================================================
# lib.sh — shared helpers for the Lambda serving-unit scripts. Sourced, not run.
#
# Provides: repo-root discovery, .env loading (LAMBDA_API_KEY etc.), a curl
# wrapper for the Lambda Cloud API that NEVER echoes the key, and small json/log
# helpers. All Lambda API calls go through `lambda_api` so auth + base URL live
# in one place. Favor the official API surface (cloud.lambda.ai/api/v1) — no
# scraping, no undocumented endpoints.
# ============================================================================
set -euo pipefail

# --- paths ------------------------------------------------------------------
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"           # infra/serving-lambda/scripts
UNIT_DIR="$(cd "$LIB_DIR/.." && pwd)"                             # infra/serving-lambda
REPO_ROOT="$(cd "$UNIT_DIR/../.." && pwd)"                        # engram-platform
KEYS_DIR="$UNIT_DIR/.keys"
STATE_DIR="$UNIT_DIR/.state"
SSH_KEY="$KEYS_DIR/engram-lambda"                                 # private key path (ed25519)
SSH_KEY_NAME="${SSH_KEY_NAME:-engram-lambda}"                     # name registered with Lambda /ssh-keys
STATE_FILE="$STATE_DIR/current.json"                             # {id, ip, region, instance_type}
TOKEN_FILE="$STATE_DIR/ml_auth_token"                            # persisted ML_AUTH_TOKEN (reused across relaunches)

LAMBDA_API_BASE="${LAMBDA_API_BASE:-https://cloud.lambda.ai/api/v1}"

log(){ echo "[$SCRIPT_TAG] $*"; }
die(){ echo "[$SCRIPT_TAG] ERROR: $*" >&2; exit 1; }
: "${SCRIPT_TAG:=lambda}"

# --- .env loading -----------------------------------------------------------
# Load repo-root .env for LAMBDA_API_KEY (+ optional CLOUDFLARE_* ). We parse it
# ourselves (not `source`) so odd values can't run shell, and we NEVER print it.
load_env(){
  local f="$REPO_ROOT/.env"
  [ -f "$f" ] || return 0
  # Export KEY=VALUE lines, ignoring comments/blanks. Values may contain '='.
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|\#*) continue ;; esac
    local key="${line%%=*}" val="${line#*=}"
    key="$(printf '%s' "$key" | tr -d '[:space:]')"
    [ -n "$key" ] || continue
    # .env is CRLF on Windows; `read` keeps the \r, which then rides into URLs
    # ("curl: (3) Malformed input" on every Cloudflare call — found live twice).
    val="${val%$'\r'}"
    # Strip surrounding single/double quotes if present.
    val="${val%\"}"; val="${val#\"}"; val="${val%\'}"; val="${val#\'}"
    export "$key=$val"
  done < "$f"
}

require_api_key(){
  load_env
  [ -n "${LAMBDA_API_KEY:-}" ] || die "LAMBDA_API_KEY not set (put it in $REPO_ROOT/.env). It is never printed."
}

# --- Lambda API wrapper -----------------------------------------------------
# lambda_api METHOD PATH [json-body]  ->  prints the JSON response body.
# The Bearer key is passed via a curl config on stdin so it never appears in the
# process list or any log. Fails loud on a non-2xx (prints status + body to stderr).
lambda_api(){
  local method="$1" path="$2" body="${3:-}"
  local url="$LAMBDA_API_BASE$path"
  local tmp status
  tmp="$(mktemp)"
  # -K - reads the auth header from stdin so the key is not an argv token.
  if [ -n "$body" ]; then
    status="$(printf 'header = "Authorization: Bearer %s"\n' "$LAMBDA_API_KEY" \
      | curl -sS -K - -o "$tmp" -w '%{http_code}' \
          -X "$method" -H 'Content-Type: application/json' -d "$body" "$url")"
  else
    status="$(printf 'header = "Authorization: Bearer %s"\n' "$LAMBDA_API_KEY" \
      | curl -sS -K - -o "$tmp" -w '%{http_code}' -X "$method" "$url")"
  fi
  local out; out="$(cat "$tmp")"; rm -f "$tmp"
  case "$status" in
    2*) printf '%s' "$out" ;;
    *)  echo "[$SCRIPT_TAG] Lambda API $method $path -> HTTP $status" >&2
        printf '%s\n' "$out" >&2
        return 1 ;;
  esac
}

# --- tiny JSON reader (jq-free; python3 is already a provision prereq) -------
# json_get '<python-expr over `d`>' <<<"$json"   e.g. json_get 'd["data"][0]["id"]'
json_get(){ python3 -c 'import sys,json; d=json.load(sys.stdin); print(eval(sys.argv[1]))' "$1"; }

mkdirs(){ mkdir -p "$KEYS_DIR" "$STATE_DIR"; chmod 700 "$KEYS_DIR" 2>/dev/null || true; }
