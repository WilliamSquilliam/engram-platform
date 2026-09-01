#!/usr/bin/env bash
# ============================================================================
# bootstrap.sh — runs ON the GPU box (invoked by provision.sh via SSM send-command).
# Idempotent: re-running upgrades the venv + configs + units in place without
# re-downloading the model (that lives on the EBS-backed HF cache).
#
# Mirrors the sibling repo's validated single-venv install (setup_cloud_vllm.sh):
# install vLLM FIRST (it pins torch), then the engram-cartridge wheel + the platform
# ml_service requirements into the SAME venv. vLLM >= 0.26 ships transformers 5.14,
# which satisfies BOTH onboarding (DynamicCache .layers) and serving — one env does both.
#
# It then writes /etc/engram/serving.env (the four contract values + store/cache
# config) and installs two systemd units:
#   engram-onboard.service  -> uvicorn app:app            :8001  (onboard / train / HF infer)
#   engram-serve.service    -> uvicorn vllm_inference:app :8002  (vLLM resident-KV serve)
#
# Inputs arrive as environment variables (provision.sh exports them into the SSM
# command). Model weights pull on the first serve-engine start (HF cache on the NVMe).
# ============================================================================
set -euo pipefail
: "${HOME:=/root}"; export HOME   # cloud-init/SSM runs with no HOME; venvs + HF cache need it

# --- required inputs (provision.sh sets these) ------------------------------
: "${MODEL_REF:?MODEL_REF is required}"           # HF weights id (e.g. RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic)
: "${VLLM_TP:?VLLM_TP is required}"               # tensor-parallel size (GPUs to shard across)
: "${ML_AUTH_TOKEN:?ML_AUTH_TOKEN is required}"   # shared bearer both services enforce
: "${CART_BUCKET:?CART_BUCKET is required}"       # S3 durable cart store bucket
: "${AWS_REGION:?AWS_REGION is required}"
: "${BUNDLE_S3_URI:?BUNDLE_S3_URI is required}"   # s3://.../serving-bundle-<ts>.tgz

# --- optional inputs (sane defaults) ----------------------------------------
CART_STORE_PREFIX="${CART_STORE_PREFIX:-cartridges}"
CONTEXT_TOKENS="${CONTEXT_TOKENS:-131072}"        # vLLM max_model_len + advertised tier context
VLLM_VERSION="${VLLM_VERSION:-0.26.0}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.90}"    # single-tenant serve box: give vLLM most of the VRAM
VLLM_TORCH_DTYPE="${VLLM_TORCH_DTYPE:-auto}"      # 'auto' for pre-quantized (FP8) checkpoints
APP_DIR="/opt/engram"
VENV="/opt/engram/venv"
WORK_NVME="/opt/engram/work"                      # HF cache + local cart mirror live on the instance store / big EBS

log(){ echo "[bootstrap] $*"; }

# ---------------------------------------------------------------------------
# 0. Base packages. The DLAMI ships python3 + the NVIDIA driver; we only need a
# venv toolchain + awscli (to pull the bundle) + git (pip build backends sometimes want it).
# ---------------------------------------------------------------------------
log "installing base packages (python venv, awscli)"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y -qq || true
sudo apt-get install -y -qq python3-venv python3-pip unzip >/dev/null 2>&1 || true
command -v aws >/dev/null 2>&1 || { sudo snap install aws-cli --classic >/dev/null 2>&1 || sudo apt-get install -y -qq awscli >/dev/null 2>&1 || true; }

# ---------------------------------------------------------------------------
# 1. Fetch + unpack the install bundle (wheel + ml_service/ + this script's peers).
# ---------------------------------------------------------------------------
log "fetching bundle $BUNDLE_S3_URI"
sudo mkdir -p "$APP_DIR" "$WORK_NVME"
sudo chown -R "$(id -u)":"$(id -g)" "$APP_DIR"
tmp="$(mktemp -d)"
aws s3 cp "$BUNDLE_S3_URI" "$tmp/bundle.tgz" --region "$AWS_REGION"
tar -xzf "$tmp/bundle.tgz" -C "$APP_DIR"
rm -rf "$tmp"
# Bundle layout: $APP_DIR/ml_service/  and  $APP_DIR/wheels/engram_cartridge-*.whl
WHEEL="$(ls "$APP_DIR"/wheels/engram_cartridge-*.whl 2>/dev/null | head -1 || true)"
[ -n "$WHEEL" ] || { echo "[bootstrap] ERROR: no engram-cartridge wheel in bundle" >&2; exit 1; }
log "wheel: $WHEEL"

# ---------------------------------------------------------------------------
# 2. Single venv: vLLM first (pins torch, ships transformers 5.14), then the
# cartridge wheel with the [s3,build] extras, then the platform ml_service
# requirements MINUS the engram-cartridge line (the wheel already supplies it).
# ---------------------------------------------------------------------------
if [ ! -d "$VENV" ]; then
  log "creating venv $VENV"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip wheel >/dev/null

log "installing vllm==$VLLM_VERSION (pins torch; ships transformers 5.14)"
pip install "vllm==$VLLM_VERSION"

log "installing the engram-cartridge wheel with [s3,build] extras"
# Install the wheel under an extras spec so boto3 (S3 store) + transformers>=5.12/accelerate
# (onboarding) come along. pip resolves the extras against the local wheel by name.
pip install "engram-cartridge[s3,build] @ file://$WHEEL"

log "installing ml_service requirements (minus engram-cartridge — the wheel supplies it)"
# Strip the engram-cartridge requirement line so pip doesn't try to fetch it from an
# index (it's already installed from the wheel above). Everything else (fastapi,
# uvicorn, pydantic, httpx, sentence-transformers, compressed-tensors) still installs.
REQ="$APP_DIR/ml_service/requirements.txt"
FILTERED="$(mktemp)"
grep -v -iE '^\s*engram-cartridge' "$REQ" > "$FILTERED" || true
pip install -r "$FILTERED"
rm -f "$FILTERED"

# Verify the stack imports before we wire services (fail loud on the box, not at first query).
python -c "import vllm, transformers, cartridges, fastapi, uvicorn; print('[bootstrap] stack ok — vllm', vllm.__version__, 'transformers', transformers.__version__)"
deactivate

# ---------------------------------------------------------------------------
# 3. /etc/engram/serving.env — the shared env both systemd units read.
# Cache dirs go on the big root/NVMe volume (each cart ~0.4GB; the model is ~70GB).
# CARTRIDGE_REGISTRY_DIR is a shared dir the vLLM EngineCore subprocess inherits
# (per-request cart routing rendezvous). CARTRIDGES_MODEL / VLLM_TP drive both
# onboarding (weights it builds carts against) and serving (weights it loads).
# ---------------------------------------------------------------------------
log "writing /etc/engram/serving.env"
sudo mkdir -p /etc/engram
sudo tee /etc/engram/serving.env >/dev/null <<ENV
# Generated by bootstrap.sh — the serving unit's runtime env. Both services read this.
# --- model + serving ---
CARTRIDGES_MODEL=$MODEL_REF
VLLM_TP=$VLLM_TP
VLLM_MAX_MODEL_LEN=$CONTEXT_TOKENS
VLLM_GPU_MEM_UTIL=$VLLM_GPU_MEM_UTIL
VLLM_TORCH_DTYPE=$VLLM_TORCH_DTYPE
# --- ML-plane shared-token auth (enforced on every route except /health) ---
ML_AUTH_TOKEN=$ML_AUTH_TOKEN
# --- durable cartridge store (S3): onboarding writes, serve reads ---
CARTRIDGE_STORE_BACKEND=s3
CARTRIDGE_STORE_BUCKET=$CART_BUCKET
CARTRIDGE_STORE_PREFIX=$CART_STORE_PREFIX
AWS_REGION=$AWS_REGION
AWS_DEFAULT_REGION=$AWS_REGION
# --- caches + registry on the big local volume (NOT the container layer / tmpfs) ---
CART_CACHE_DIR=$WORK_NVME/cart_cache
CARTRIDGE_REGISTRY_DIR=$WORK_NVME/cartridge_registry
HF_HOME=$WORK_NVME/hf
HF_HUB_DISABLE_PROGRESS_BARS=1
# The connector crosses the vLLM EngineCore subprocess boundary (cart KV serialization).
VLLM_ALLOW_INSECURE_SERIALIZATION=1
PYTHONUNBUFFERED=1
ENV
sudo mkdir -p "$WORK_NVME/cart_cache" "$WORK_NVME/cartridge_registry" "$WORK_NVME/hf"

# ---------------------------------------------------------------------------
# 4. systemd units. Both run uvicorn from the venv with the ml_service dir on the
# app path, load /etc/engram/serving.env, and restart on failure. The serve unit
# builds the vLLM engine at startup and pulls the model on first start (minutes).
# ---------------------------------------------------------------------------
log "installing systemd units"
sudo tee /etc/systemd/system/engram-onboard.service >/dev/null <<UNIT
[Unit]
Description=Engram onboarding worker (ml_service app:app, :8001)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=/etc/engram/serving.env
WorkingDirectory=$APP_DIR/ml_service
ExecStart=$VENV/bin/uvicorn app:app --host 0.0.0.0 --port 8001
Restart=on-failure
RestartSec=10
# Onboarding builds carts on the GPU; give it room and a clean kill.
TimeoutStartSec=600
KillMode=mixed

[Install]
WantedBy=multi-user.target
UNIT

sudo tee /etc/systemd/system/engram-serve.service >/dev/null <<UNIT
[Unit]
Description=Engram vLLM Inference Service (ml_service vllm_inference:app, :8002)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=/etc/engram/serving.env
WorkingDirectory=$APP_DIR/ml_service
ExecStart=$VENV/bin/uvicorn vllm_inference:app --host 0.0.0.0 --port 8002
Restart=on-failure
RestartSec=15
# The vLLM engine cold build + first model pull takes minutes; don't let systemd
# kill it as a failed start. The service warms the engine in a background thread
# (SERVE_WARMUP=1), so :8002/health answers quickly and reports engine_ready.
TimeoutStartSec=1800
KillMode=mixed

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable engram-onboard.service engram-serve.service
# Restart (not just start) so a re-provision picks up the new venv/env/units.
sudo systemctl restart engram-onboard.service engram-serve.service

log "done. Units enabled + (re)started. First serve start downloads the model (~70GB)."
log "  onboard: systemctl status engram-onboard  | curl localhost:8001/health"
log "  serve:   systemctl status engram-serve    | curl localhost:8002/health   (engine_ready flips true after the model loads)"
