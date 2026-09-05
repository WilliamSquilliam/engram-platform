#!/usr/bin/env bash
# ============================================================================
# provision.sh — runs from the OPERATOR'S machine after launch.sh.
# Builds the install bundle, generates serving.env, ships it to the Lambda box
# over SSH, and runs bootstrap-lambda.sh there. Idempotent: re-running rebuilds
# the wheel, re-uploads, and re-runs bootstrap (upgrades the venv/env/units in
# place; the HF weights on the persistent FS / local SSD cache are not touched).
#
# Steps:
#   1. read .state/current.json (ip from launch.sh) + aws-support tf outputs
#      (cart bucket + bucket-scoped AWS key) + persist/reuse ML_AUTH_TOKEN
#   2. build the engram-cartridge wheel from ../Engram-Smart-CAG (python -m build)
#   3. generate serving.env (the four contract values + store/cache config +
#      the least-privilege bucket-only AWS creds)
#   4. bundle: wheel + ml_service/ + bootstrap-lambda.sh + serving.env -> .tgz
#   5. scp the bundle to the box and run bootstrap-lambda.sh over ssh
#
# Favor established tooling: OpenSSH + systemd + `python -m build`; nothing bespoke.
# NEVER prints ML_AUTH_TOKEN, the AWS secret, or LAMBDA_API_KEY.
# ============================================================================
set -euo pipefail
SCRIPT_TAG="provision"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

SIBLING_REPO="${SIBLING_REPO:-$(cd "$REPO_ROOT/../Engram-Smart-CAG" && pwd)}"
AWS_SUPPORT_DIR="$UNIT_DIR/aws-support"
SSH_USER="ubuntu"                         # Lambda instances log in as ubuntu
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KEYS_DIR/known_hosts")

# Lambda-side vars (pivot the model/TP here; mirror launch.sh's B200->H100 story).
CARTRIDGES_MODEL="${CARTRIDGES_MODEL:-CohereLabs/command-a-plus-05-2026-w4a4}"
VLLM_TP="${VLLM_TP:-1}"                    # 1 for a single B200; set 2 for the 2x H100 SXM fallback
# 65536, NOT 131072, on the 2x H100 (160GB): serving 131k needs an >=8GB KV pool which
# forces GPU_MEM_UTIL>=0.90 and leaves <1GB free — cart load/scatter's ~2GB/GPU transient
# fp32 decode buffer then OOMs and requests degrade to blank KV (found live). vLLM REJECTS
# the expandable_segments allocator workaround when a KV connector is configured (VMM page
# remapping would invalidate connector KV views). 64k covers real product prompts (top-k
# carts + question + history is ~<32k); 131k returns with the B200 (more VRAM) or the
# chunked per-layer cart decode (wheel backlog) that shrinks the transient ~64x.
# 40960 (was 45056, was 65536): the gated config's graph capture reserves KV-pool memory and the
# pool must still fit one max-length request; and at 0.85 util a large cart's ~3GB fp32 decode
# buffer no longer fit beside a 29-cart resident set (OOM -> blank-KV degrade, found live
# 2026-09-05). Trimmed with the util drop below. Max real k=3 request is ~8k tokens — still ample.
CONTEXT_TOKENS="${CONTEXT_TOKENS:-40960}"
FS_NAME="${FS_NAME:-engram-fs}"
HOST_SERVE="${HOST_SERVE:-gpu.engramdynamics.org}"
HOST_ONBOARD="${HOST_ONBOARD:-gpu-onboard.engramdynamics.org}"

command -v python3 >/dev/null || die "python3 not on PATH (needed to build the wheel)"
command -v terraform >/dev/null || die "terraform not on PATH (needed for aws-support outputs)"
command -v ssh >/dev/null || die "ssh not on PATH"
command -v scp >/dev/null || die "scp not on PATH"
[ -d "$SIBLING_REPO" ] || die "sibling repo not found at $SIBLING_REPO (set SIBLING_REPO=/path/to/Engram-Smart-CAG)"
[ -f "$STATE_FILE" ] || die "no $STATE_FILE — run launch.sh first"
[ -f "$SSH_KEY" ] || die "no SSH key at $SSH_KEY — run launch.sh first"

IP="$(json_get 'd["ip"]' < "$STATE_FILE")"
[ -n "$IP" ] || die "no ip in $STATE_FILE"
log "target box $IP (fs=$FS_NAME model=$CARTRIDGES_MODEL tp=$VLLM_TP)"

# ---------------------------------------------------------------------------
# 1a. aws-support outputs — cart bucket + bucket-scoped AWS creds. The box has no
# AWS instance-profile, so these least-privilege creds go into serving.env.
# ---------------------------------------------------------------------------
TFS="terraform -chdir=$AWS_SUPPORT_DIR"
tfs(){ $TFS output -raw "$1" 2>/dev/null; }
CART_BUCKET="$(tfs cart_bucket)";            [ -n "$CART_BUCKET" ] || die "no cart_bucket output — apply aws-support/ first"
CART_PREFIX="$(tfs cart_store_prefix)";      [ -n "$CART_PREFIX" ] || CART_PREFIX="cartridges"
CART_REGION="$(tfs cart_bucket_region)";     [ -n "$CART_REGION" ] || CART_REGION="us-east-1"
AWS_KEY_ID="$(tfs serving_access_key_id)";   [ -n "$AWS_KEY_ID" ] || die "no serving_access_key_id — apply aws-support/ first"
AWS_SECRET="$(tfs serving_secret_access_key)"; [ -n "$AWS_SECRET" ] || die "no serving_secret_access_key"
log "cart store: s3://$CART_BUCKET/$CART_PREFIX ($CART_REGION) via bucket-scoped IAM key"

# ---------------------------------------------------------------------------
# 1b. ML_AUTH_TOKEN — generate ONCE, persist, reuse across relaunches so the
# platform env values don't churn every time the box is recycled.
# ---------------------------------------------------------------------------
mkdirs
if [ ! -f "$TOKEN_FILE" ]; then
  log "generating ML_AUTH_TOKEN (persisted once at $TOKEN_FILE, reused on relaunch)"
  # base62-ish (no shell-special chars) so it's safe in env files / Authorization headers.
  python3 -c 'import secrets;print(secrets.token_urlsafe(36).replace("-","").replace("_","")[:48])' > "$TOKEN_FILE"
  chmod 600 "$TOKEN_FILE"
fi
ML_AUTH_TOKEN="$(cat "$TOKEN_FILE")"

# ---------------------------------------------------------------------------
# 2. Build the engram-cartridge wheel from the sibling repo.
# ---------------------------------------------------------------------------
log "building engram-cartridge wheel from $SIBLING_REPO (python -m build)"
BUILD_TMP="$(mktemp -d)"
trap 'rm -rf "$BUILD_TMP"' EXIT
( cd "$SIBLING_REPO" && python3 -m build --wheel --outdir "$BUILD_TMP/wheels" ) \
  || die "wheel build failed — is 'build' installed?  pip install build"
WHEEL="$(ls "$BUILD_TMP"/wheels/engram_cartridge-*.whl 2>/dev/null | head -1 || true)"
[ -n "$WHEEL" ] || die "no wheel produced under $BUILD_TMP/wheels"
log "wheel: $(basename "$WHEEL")"

# ---------------------------------------------------------------------------
# 3. Generate serving.env — the shared env both systemd units read on the box.
# Contract values + S3 store + the bucket-only AWS creds + storage paths. Written
# 0600 into the bundle; bootstrap installs it to /etc/engram/serving.env (0600).
# The AWS creds are LEAST-PRIVILEGE (aws-support scopes them to this bucket only).
# ---------------------------------------------------------------------------
STAGE="$BUILD_TMP/stage"
mkdir -p "$STAGE/wheels" "$STAGE/ml_service"
cp "$WHEEL" "$STAGE/wheels/"
umask 077
cat > "$STAGE/serving.env" <<ENV
# Generated by provision.sh — the Lambda serving unit's runtime env. Both services read this.
# --- model + serving ---
CARTRIDGES_MODEL=$CARTRIDGES_MODEL
VLLM_TP=$VLLM_TP
VLLM_MAX_MODEL_LEN=$CONTEXT_TOKENS
# 0.82, NOT 0.90/0.85: each cart load/scatter needs a transient fp32 decode buffer OUTSIDE the
# vLLM pool — ~2GB/GPU for typical carts, 3GB+ for large-document carts. At 0.90 only ~1GB/GPU
# was free (bench run); at 0.85 a 29-cart tenant's large cart still OOM'd to blank-KV degrade
# (live, 2026-09-05). 0.82 + the smaller CONTEXT_TOKENS keeps ~5GB/GPU free for loads. The REAL
# fix is the wheel's chunked per-layer cart decode (backlog) which shrinks the transient ~64x.
VLLM_GPU_MEM_UTIL=0.82
# NOTE: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is NOT safe here — vLLM refuses to start
# it with a KV connector (VMM can remap KV cache pages, invalidating the connector's pinned
# addresses; found live as a crash-loop 2026-09-05). Headroom comes from the util/ctx trims alone.
VLLM_TORCH_DTYPE=auto
# --- GATED engine config (adopted 2026-09-03, lossless gate passed: accuracy identical
# 13/15 before/after; anchors remeasured — 3.6x qps at conc 1 tapering to 1.24x at 128).
# CUDA graphs (eager off) + ngram speculative decoding: grounded answers QUOTE their
# resident documents, so ngram drafts verify at high rates (greedy = byte-identical
# output). NOTE: graph capture reserves pool memory — at ctx 65536 the pool no longer
# fit one max-length request, hence the CONTEXT_TOKENS=45056 default above; a bigger
# pool (B200 / higher util) can raise it back.
VLLM_ENFORCE_EAGER=0
SERVE_SPEC=ngram
# Structured reasoning: Command A+ thinks between <|START_THINKING|>/<|END_THINKING|>; the
# server streams that span as 'thinking' frames so deliberation never lands in the answer
# (prompt-side suppression demonstrably failed — the model quoted the instruction back).
SERVE_REASONING=channel
# Deterministic loop-breaker for greedy decoding: argmax over penalized logits; the spec-decode
# verifier scores drafts under the same distribution, so ngram spec stays lossless. 1.0 disables.
# Without it a thinking-channel repeat loop burns the whole budget and yields NO answer (seen live).
SERVE_REP_PENALTY=1.05
# --- ML-plane shared-token auth (enforced on every route except /health) ---
ML_AUTH_TOKEN=$ML_AUTH_TOKEN
# --- onboard THROUGH the engine: the engine is the only stack that can run this model class; the
# transformers path stays for models it can load. INFERENCE_SERVICE_URL is box-local (never Caddy). ---
ONBOARD_VIA_ENGINE=1
INFERENCE_SERVICE_URL=http://127.0.0.1:8002
# --- durable cartridge store (S3): onboarding writes, serve reads ---
# LEAST-PRIVILEGE creds: the aws-support IAM user is scoped to THIS bucket only.
CARTRIDGE_STORE_BACKEND=s3
CARTRIDGE_STORE_BUCKET=$CART_BUCKET
CARTRIDGE_STORE_PREFIX=$CART_PREFIX
AWS_REGION=$CART_REGION
AWS_DEFAULT_REGION=$CART_REGION
AWS_ACCESS_KEY_ID=$AWS_KEY_ID
AWS_SECRET_ACCESS_KEY=$AWS_SECRET
# --- three-tier storage (paths bootstrap creates on the local SSD root) ---
# HF weights cache (seeded from / synced to the persistent FS), cart hot-mirror,
# and the cross-subprocess cart registry. See bootstrap-lambda.sh for the tiers.
HF_HOME=/home/ubuntu/engram/hf
CART_CACHE_DIR=/home/ubuntu/engram/cart_cache
CARTRIDGE_REGISTRY_DIR=/home/ubuntu/engram/registry
HF_HUB_DISABLE_PROGRESS_BARS=1
# The connector crosses the vLLM EngineCore subprocess boundary (cart KV serialization).
VLLM_ALLOW_INSECURE_SERIALIZATION=1
# TP>1 REQUIRES spawn workers: forked subprocesses cannot re-init CUDA ("Cannot
# re-initialize CUDA in forked subprocess" crash-loop, hit on the 2026-09-03 fresh
# 2x H100 relaunch — the fix had lived only on the old box's env, never in this
# template). Safe for TP=1 too, so set unconditionally.
VLLM_WORKER_MULTIPROC_METHOD=spawn
# RoPE pairing convention of the served model for the multi-cart rebase: Cohere family
# (Command A/A+) is INTERLEAVED; Llama/Qwen are half-split (the default). A wrong value
# corrupts 2nd+ cart keys instead of shifting them.
CARTRIDGE_ROPE_CONVENTION=interleaved
PYTHONUNBUFFERED=1
# --- box-side inputs bootstrap-lambda.sh reads (not app config) ---
LAMBDA_FS_NAME=$FS_NAME
CADDY_HOST_SERVE=$HOST_SERVE
CADDY_HOST_ONBOARD=$HOST_ONBOARD
ENV
umask 022

# ---------------------------------------------------------------------------
# 4. Bundle: wheel + ml_service/ + bootstrap-lambda.sh + serving.env +
# self-provision.sh. self-provision.sh is the UNATTENDED entry point: the
# platform-admin Start button launches a fresh box with a cloud-init user_data
# that runs it off the persistent FS (step 6 publishes it there) — one
# provisioning path whether an operator SSHes or the backend presses the button.
# ---------------------------------------------------------------------------
tar -C "$REPO_ROOT" --exclude='__pycache__' --exclude='*.pyc' -cf - ml_service \
  | tar -C "$STAGE" -xf -
cp "$LIB_DIR/bootstrap-lambda.sh" "$STAGE/bootstrap-lambda.sh"
chmod +x "$STAGE/bootstrap-lambda.sh"
# The dependency lockfile (pip freeze from a proven-green box) rides along as a pip
# CONSTRAINTS file: bootstrap keeps its install lines but every transitive version is
# pinned, so a fresh region's resolve can't drift (2026-09-03: an unpinned resolve on
# an older image produced a stack that couldn't run — flashinfer/torch/driver matrix).
cp "$LIB_DIR/../requirements.lock" "$STAGE/requirements.lock" 2>/dev/null \
  || log "WARN: no requirements.lock beside scripts/ — bootstrap will resolve unpinned"

cat > "$STAGE/self-provision.sh" <<'SELFPROV'
#!/usr/bin/env bash
# Runs as root ON a fresh Lambda box at first boot (cloud-init user_data from the
# platform-admin Start button). The persistent FS carries this script + the
# provision bundle; unpack to the same path provision.sh uses and run the same
# bootstrap — no operator SSH involved. Idempotent.
set -euo pipefail
FS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # /lambda/nfs/<fs>/engram
BUNDLE="$FS_DIR/bundle.tgz"
[ -f "$BUNDLE" ] || { echo "[self-provision] no bundle at $BUNDLE — run provision.sh once from the operator machine"; exit 1; }
rm -rf /tmp/engram-bootstrap && mkdir -p /tmp/engram-bootstrap   # never extract over a stale staging dir
tar -C /tmp/engram-bootstrap -xzf "$BUNDLE"
chmod +x /tmp/engram-bootstrap/bootstrap-lambda.sh
bash /tmp/engram-bootstrap/bootstrap-lambda.sh
SELFPROV
chmod +x "$STAGE/self-provision.sh"
BUNDLE="$BUILD_TMP/lambda-bundle.tgz"
tar -C "$STAGE" -czf "$BUNDLE" .
log "bundle: $(du -h "$BUNDLE" | cut -f1)"

# ---------------------------------------------------------------------------
# 5. Ship it + run bootstrap over SSH. accept-new pins the host key on first
# connect (recorded in .keys/known_hosts, gitignored).
# ---------------------------------------------------------------------------
log "waiting for SSH on $IP"
for _ in $(seq 1 40); do
  ssh "${SSH_OPTS[@]}" -o ConnectTimeout=8 "$SSH_USER@$IP" true 2>/dev/null && break
  sleep 8
done
ssh "${SSH_OPTS[@]}" -o ConnectTimeout=8 "$SSH_USER@$IP" true 2>/dev/null || die "SSH to $IP not ready"

log "uploading bundle to the box"
# CLEAN the staging dir first: extractions accumulate across provisions, and a
# stale older wheel beside the new one gets picked by bootstrap's lexical glob
# (0.4.2 sorted before 0.5.0 and silently reinstalled the old wheel).
ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" 'rm -rf /tmp/engram-bootstrap && mkdir -p /tmp/engram-bootstrap'
scp "${SSH_OPTS[@]}" "$BUNDLE" "$SSH_USER@$IP:/tmp/engram-bootstrap/bundle.tgz" >/dev/null

log "running bootstrap-lambda.sh on the box (venv + vLLM + Caddy — several minutes)"
# -tt so systemctl/apt output streams back; bootstrap is idempotent + set -euo pipefail.
ssh "${SSH_OPTS[@]}" -tt "$SSH_USER@$IP" '
  set -e
  cd /tmp/engram-bootstrap
  tar -xzf bundle.tgz
  chmod +x bootstrap-lambda.sh
  sudo bash bootstrap-lambda.sh
'

# ---------------------------------------------------------------------------
# 6. Publish the self-provision assets to the persistent FS so the backend's
# Start button can relaunch UNATTENDED (cloud-init runs self-provision.sh off
# the FS; no operator SSH). The bundle holds serving.env (ML_AUTH_TOKEN + the
# bucket-scoped AWS key) — same trust domain as the box itself; the FS is
# account-private. Re-running provision refreshes the published copy.
# ---------------------------------------------------------------------------
log "publishing self-provision assets to the persistent FS (/lambda/nfs/$FS_NAME/engram)"
ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" "
  set -e
  sudo mkdir -p /lambda/nfs/$FS_NAME/engram
  sudo cp /tmp/engram-bootstrap/bundle.tgz /tmp/engram-bootstrap/self-provision.sh /lambda/nfs/$FS_NAME/engram/
  sudo chmod 700 /lambda/nfs/$FS_NAME/engram
  sudo chmod +x /lambda/nfs/$FS_NAME/engram/self-provision.sh
"

echo
log "==================== PROVISIONED ===================="
log "The serve engine seeds/loads Command A+ W4A4 on first start."
log "Weights seed from the persistent FS if present; otherwise download (~120GB) then sync UP to the FS."
log "Next: bash $LIB_DIR/smoke.sh    (waits for engine_ready over HTTPS, then onboards + queries + describes)"
