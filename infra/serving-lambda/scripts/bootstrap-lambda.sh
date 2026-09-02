#!/usr/bin/env bash
# ============================================================================
# bootstrap-lambda.sh — runs ON the Lambda GPU box (invoked by provision.sh over
# SSH as root). Idempotent: re-running upgrades the venv + configs + units in
# place without re-downloading the model (weights live on the persistent FS /
# local SSD cache).
#
# Adapts serving-aws/scripts/bootstrap.sh to Lambda Cloud:
#   * Lambda ships Ubuntu 22.04 + Lambda Stack (NVIDIA driver + CUDA preinstalled),
#     so we install a Python venv on top — same as the DLAMI path.
#   * The box runs as user `ubuntu`, not root/SSM; paths live under /home/ubuntu.
#   * THREE STORAGE TIERS (see below), with HF-weight seeding to/from the
#     persistent filesystem so relaunches skip the ~120GB download.
#   * Caddy fronts the two services with auto-HTTPS; :8001/:8002 bind to 127.0.0.1
#     so ONLY Caddy is exposed (defense in depth on top of the Lambda firewall).
#
# Inputs arrive in /etc/engram/serving.env, written from the bundle (provision.sh
# generated it). This script reads a few box-side keys from it (LAMBDA_FS_NAME,
# CADDY_HOST_*, CARTRIDGES_MODEL, VLLM_TP).
#
# THREE STORAGE TIERS (matching the README's story):
#   1. Local SSD root (/home/ubuntu/engram) — EPHEMERAL (dies on terminate). Holds
#      the live HF weights cache, cart hot-mirror, and cart registry. Big + fast.
#   2. Persistent filesystem (/lambda/nfs/<fs>) — SURVIVES terminate (region-bound).
#      Holds the SEEDED HF weights: rsync FS->local before engine start (fast
#      relaunch), and local->FS after the first successful warm (one-time seed).
#   3. S3 cart bucket (aws-support) — DURABLE cart store. Onboarding writes, serve
#      reads; loss-tolerant local mirror re-warms from here.
# ============================================================================
set -euo pipefail

APP_DIR="/opt/engram"                     # code (ml_service) + venv live here
VENV="$APP_DIR/venv"
HOME_ENGRAM="/home/ubuntu/engram"         # local-SSD storage tier (ephemeral)
BUNDLE_DIR="/tmp/engram-bootstrap"        # provision.sh unpacked the bundle here
ENV_FILE="/etc/engram/serving.env"
RUN_USER="ubuntu"

log(){ echo "[bootstrap-lambda] $*"; }
die(){ echo "[bootstrap-lambda] ERROR: $*" >&2; exit 1; }

[ -f "$BUNDLE_DIR/serving.env" ] || die "no serving.env in the bundle ($BUNDLE_DIR)"
[ -d "$BUNDLE_DIR/ml_service" ] || die "no ml_service/ in the bundle"
WHEEL="$(ls "$BUNDLE_DIR"/wheels/engram_cartridge-*.whl 2>/dev/null | head -1 || true)"
[ -n "$WHEEL" ] || die "no engram-cartridge wheel in the bundle"

# ---------------------------------------------------------------------------
# 0. Install the shared env + code from the bundle.
# ---------------------------------------------------------------------------
log "installing /etc/engram/serving.env + code from bundle"
mkdir -p /etc/engram
install -m 0600 "$BUNDLE_DIR/serving.env" "$ENV_FILE"
# Pull the box-side inputs we need here (model/TP for the comment, fs name, hosts).
# shellcheck disable=SC1090
set -a; . "$ENV_FILE"; set +a
FS_NAME="${LAMBDA_FS_NAME:-engram-fs}"
FS_ROOT="/lambda/nfs/$FS_NAME"            # Lambda mounts persistent filesystems here
HOST_SERVE="${CADDY_HOST_SERVE:-gpu.engramdynamics.org}"
HOST_ONBOARD="${CADDY_HOST_ONBOARD:-gpu-onboard.engramdynamics.org}"

# TP must equal the GPU count on THIS box. The B200(TP=1) vs 2x H100(TP=2) choice
# happens at LAUNCH time by capacity, and the unattended relaunch path (platform
# admin Start button) reuses one bundle for either shape — so the bundled VLLM_TP
# is only a fallback. Autodetect and pin it in the installed env file.
NGPU="$(nvidia-smi --list-gpus 2>/dev/null | grep -c GPU || true)"
if [ "${NGPU:-0}" -ge 1 ]; then
  sed -i "s/^VLLM_TP=.*/VLLM_TP=$NGPU/" "$ENV_FILE"
  log "VLLM_TP pinned to detected GPU count: $NGPU"
else
  log "WARN: could not detect GPU count (nvidia-smi) — keeping bundled VLLM_TP"
fi

mkdir -p "$APP_DIR"
rm -rf "$APP_DIR/ml_service"
cp -r "$BUNDLE_DIR/ml_service" "$APP_DIR/ml_service"
# CLEAR stale wheels first: this dir accumulates across provisions and the glob
# below picks lexically-first — 0.4.2 sorted before 0.5.0 and silently reinstalled
# the OLD wheel on a re-provision. Only the bundle's wheel may live here.
rm -rf "$APP_DIR/wheels"
mkdir -p "$APP_DIR/wheels"
cp "$WHEEL" "$APP_DIR/wheels/"
WHEEL="$(ls "$APP_DIR"/wheels/engram_cartridge-*.whl | head -1)"

# ---------------------------------------------------------------------------
# 1. Storage tiers — local SSD dirs (tier 1). The persistent FS (tier 2) is
# mounted by Lambda at $FS_ROOT already (we attached it at launch).
# ---------------------------------------------------------------------------
log "creating local-SSD storage tier under $HOME_ENGRAM (ephemeral)"
mkdir -p "$HOME_ENGRAM/hf" "$HOME_ENGRAM/cart_cache" "$HOME_ENGRAM/registry"
chown -R "$RUN_USER":"$RUN_USER" "$HOME_ENGRAM"
# /data is the CONTRACT path: the control plane sends corpus_dir=/data/corpora/<id>
# (its own volume layout) and the ML service's corpus-dir allowlist defaults to
# /data. The AWS unit had a RAID0 mounted there; on Lambda it's a plain dir on the
# big root SSD. Without it every onboard 400s ("outside the allowed data roots").
mkdir -p /data/corpora
chown -R "$RUN_USER":"$RUN_USER" /data
if [ -d "$FS_ROOT" ]; then
  mkdir -p "$FS_ROOT/hf"
  log "persistent FS present at $FS_ROOT (weights seed target)"
else
  log "WARN: persistent FS not mounted at $FS_ROOT — relaunches will re-download weights"
fi

# ---------------------------------------------------------------------------
# 2. Base packages + Python venv. Lambda Stack already has the driver/CUDA; we
# only need the venv toolchain, rsync (weight seeding), and Caddy's prereqs.
# ---------------------------------------------------------------------------
log "installing base packages (python venv, rsync, caddy prereqs)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y -qq || true
apt-get install -y -qq python3-venv python3-pip rsync curl debian-keyring debian-archive-keyring apt-transport-https >/dev/null 2>&1 || true

if [ ! -d "$VENV" ]; then
  log "creating venv $VENV"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip wheel >/dev/null

# INSTALL ORDER MATTERS: wheel + app requirements FIRST, vllm LAST. vllm pins its
# exact torch (and matching torchvision/pydantic floors); anything installed after
# it can silently downgrade those and leave an ABI-mismatched stack that only
# explodes at import time inside the service (torchvision::nms does not exist —
# this happened: engram-cartridge's torch<2.12 pin dragged torch under vllm 0.28's
# torch==2.13). Installing vllm last makes its resolver the final word.
log "installing the engram-cartridge wheel with [s3,build] extras"
pip install "engram-cartridge[s3,build] @ file://$WHEEL"

log "installing ml_service requirements (minus engram-cartridge — the wheel supplies it)"
REQ="$APP_DIR/ml_service/requirements.txt"
FILTERED="$(mktemp)"
grep -v -iE '^\s*engram-cartridge' "$REQ" > "$FILTERED" || true
pip install -r "$FILTERED"
rm -f "$FILTERED"

# --- vLLM LAST (see order note above). Blackwell (B200, SM100) needs recent CUDA
# wheels, so we install the LATEST vllm (>=0.26) rather than pinning an older one.
# ASSUMPTION: the latest vllm on PyPI ships CUDA wheels compatible with Lambda
# Stack's driver; if a fresh Blackwell build regresses, pin a known-good version.
# -U so an already-satisfied vllm still gets its dep pins re-enforced on re-runs.
log "installing latest vllm (>=0.26, LAST so its torch/pydantic pins win)"
pip install -U "vllm>=0.26"

# Fail loud on the box (not at first query) if the stack doesn't import. Must hit
# the LAZY import paths the services actually hit: plain `import vllm` passed while
# vllm.transformers_utils.config (-> transformers -> torchvision) was broken.
python - <<'CHECK'
import torch, torchvision, torchaudio, fastapi, pydantic, uvicorn
import vllm, transformers
from vllm.transformers_utils import config as _lazy_path  # crashes on torch/torchvision ABI mismatch
import cartridges
print("[bootstrap-lambda] stack ok — vllm", vllm.__version__, "torch", torch.__version__,
      "torchvision", torchvision.__version__, "transformers", transformers.__version__,
      "pydantic", pydantic.VERSION)
CHECK
deactivate
chown -R "$RUN_USER":"$RUN_USER" "$APP_DIR"

# ---------------------------------------------------------------------------
# 3. Weight seeding units (tier 2 <-> tier 1). A PRE unit rsyncs FS->local before
# the engine starts (fast relaunch); a POST unit rsyncs local->FS after the first
# successful warm (one-time ~120GB seed). Both are idempotent no-ops when the
# source is empty or already in sync.
# ---------------------------------------------------------------------------
log "installing HF-weight seed scripts + units"
mkdir -p "$APP_DIR/bin"

cat > "$APP_DIR/bin/seed-weights-in.sh" <<SEEDIN
#!/usr/bin/env bash
# Tier2->Tier1: seed local HF cache from the persistent FS before the engine starts.
# No-op (fast) when the FS has no weights yet (first ever launch).
set -euo pipefail
SRC="$FS_ROOT/hf"; DST="$HOME_ENGRAM/hf"
mkdir -p "\$DST"
if [ -d "\$SRC" ] && [ -n "\$(ls -A "\$SRC" 2>/dev/null || true)" ]; then
  echo "[seed-in] rsync FS weights -> local (\$SRC -> \$DST)"
  rsync -a --info=stats1 "\$SRC/" "\$DST/"
else
  echo "[seed-in] no weights on the FS yet — engine will download to local, then seed-out"
fi
SEEDIN

cat > "$APP_DIR/bin/seed-weights-out.sh" <<SEEDOUT
#!/usr/bin/env bash
# Tier1->Tier2: after the engine is warm (engine_ready), sync local HF weights UP
# to the persistent FS so the NEXT launch seeds them back in (skips the download).
# Waits for engine_ready, then rsyncs once. Idempotent; skips if the FS is absent.
set -euo pipefail
SRC="$HOME_ENGRAM/hf"; DST="$FS_ROOT/hf"
[ -d "$FS_ROOT" ] || { echo "[seed-out] no persistent FS at $FS_ROOT — skipping"; exit 0; }
mkdir -p "\$DST"
echo "[seed-out] waiting for engine_ready before syncing weights up"
for _ in \$(seq 1 240); do   # up to ~60 min for the first cold model load
  if curl -sf http://127.0.0.1:8002/health 2>/dev/null | grep -q '"engine_ready":true'; then
    echo "[seed-out] engine ready — rsync local weights -> FS (\$SRC -> \$DST)"
    rsync -a --info=stats1 "\$SRC/" "\$DST/"
    echo "[seed-out] weights seeded to the persistent FS; relaunches will be fast"
    exit 0
  fi
  sleep 15
done
echo "[seed-out] engine never became ready within the window — leaving FS unseeded"
exit 0
SEEDOUT
chmod +x "$APP_DIR/bin/seed-weights-in.sh" "$APP_DIR/bin/seed-weights-out.sh"

cat > /etc/systemd/system/engram-seed-in.service <<UNIT
[Unit]
Description=Engram HF-weight seed IN (persistent FS -> local SSD, before serve)
After=network-online.target
Wants=network-online.target
Before=engram-serve.service

[Service]
Type=oneshot
RemainAfterExit=yes
User=$RUN_USER
ExecStart=$APP_DIR/bin/seed-weights-in.sh

[Install]
WantedBy=multi-user.target
UNIT

cat > /etc/systemd/system/engram-seed-out.service <<UNIT
[Unit]
Description=Engram HF-weight seed OUT (local SSD -> persistent FS, after first warm)
After=engram-serve.service
Wants=engram-serve.service

[Service]
Type=oneshot
User=$RUN_USER
ExecStart=$APP_DIR/bin/seed-weights-out.sh

[Install]
WantedBy=multi-user.target
UNIT

# ---------------------------------------------------------------------------
# 4. systemd app units. Both run uvicorn from the venv, load serving.env, and
# bind to 127.0.0.1 ONLY (Caddy is the sole public exposure). The serve unit
# orders After= the seed-in unit so weights are local before the engine starts.
# ---------------------------------------------------------------------------
log "installing systemd app units (bound to 127.0.0.1)"
cat > /etc/systemd/system/engram-onboard.service <<UNIT
[Unit]
Description=Engram onboarding worker (ml_service app:app, 127.0.0.1:8001)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
EnvironmentFile=$ENV_FILE
# venv/bin FIRST on PATH: the stack shells out to pip-installed console tools
# (ninja for vLLM's JIT kernel warmup, etc.) that live only in the venv — the
# systemd default PATH broke engine startup with "No such file or directory:
# 'ninja'" inside determine_available_memory.
Environment=PATH=$VENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
WorkingDirectory=$APP_DIR/ml_service
ExecStart=$VENV/bin/uvicorn app:app --host 127.0.0.1 --port 8001
Restart=on-failure
RestartSec=10
TimeoutStartSec=600
KillMode=mixed

[Install]
WantedBy=multi-user.target
UNIT

cat > /etc/systemd/system/engram-serve.service <<UNIT
[Unit]
Description=Engram vLLM Inference Service (ml_service vllm_inference:app, 127.0.0.1:8002)
After=network-online.target engram-seed-in.service
Wants=network-online.target engram-seed-in.service

[Service]
Type=simple
User=$RUN_USER
EnvironmentFile=$ENV_FILE
# venv/bin FIRST on PATH — same constraint as the onboard unit (ninja et al.).
Environment=PATH=$VENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
WorkingDirectory=$APP_DIR/ml_service
ExecStart=$VENV/bin/uvicorn vllm_inference:app --host 127.0.0.1 --port 8002
Restart=on-failure
RestartSec=15
# Cold engine build + first weight seed/download takes many minutes; don't let
# systemd kill it as a failed start. The service warms in a background thread so
# :8002/health answers quickly and reports engine_ready.
TimeoutStartSec=3600
KillMode=mixed

[Install]
WantedBy=multi-user.target
UNIT

# ---------------------------------------------------------------------------
# 5. Caddy — official apt repo. Two vhosts, auto-HTTPS via Let's Encrypt HTTP-01.
# Caddy is the ONLY thing on 80/443; it reverse-proxies to the localhost services.
# Ports 80/443 must be reachable (Lambda firewall 22/80/443 — see launch.sh /
# README). :8001/:8002 never leave the box.
# ---------------------------------------------------------------------------
if ! command -v caddy >/dev/null 2>&1; then
  log "installing Caddy (official apt repo)"
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -y -qq || true
  apt-get install -y -qq caddy >/dev/null 2>&1 || die "caddy install failed"
else
  log "Caddy already installed"
fi

log "writing Caddyfile (two vhosts, auto-HTTPS)"
cat > /etc/caddy/Caddyfile <<CADDY
# Engram Lambda serving unit — Caddy is the sole public face (auto-HTTPS via
# Let's Encrypt HTTP-01). It reverse-proxies to the localhost-bound services;
# :8001/:8002 are never exposed. Requires inbound 80/443 (Lambda firewall).
$HOST_SERVE {
	reverse_proxy 127.0.0.1:8002
}

$HOST_ONBOARD {
	reverse_proxy 127.0.0.1:8001
}
CADDY
caddy fmt --overwrite /etc/caddy/Caddyfile >/dev/null 2>&1 || true

# ---------------------------------------------------------------------------
# 6. Enable + (re)start everything. Restart (not just start) so a re-provision
# picks up the new venv/env/units/Caddyfile.
# ---------------------------------------------------------------------------
log "enabling + (re)starting units"
systemctl daemon-reload
systemctl enable engram-seed-in.service engram-onboard.service engram-serve.service engram-seed-out.service >/dev/null 2>&1 || true
systemctl restart engram-seed-in.service
systemctl restart engram-onboard.service engram-serve.service
# --no-block is REQUIRED: seed-out is a oneshot whose ExecStart waits (up to 60
# min) for engine_ready before rsyncing weights up — a plain restart blocks this
# whole script (and the provision.sh SSH session driving it) until the engine warms.
systemctl restart --no-block engram-seed-out.service || true
# RESTART caddy, never reload: on first provision caddy is already running with the
# apt-default :80 config, and `systemctl reload` silently kept serving it (no TLS,
# nothing on 443). Restart always loads /etc/caddy/Caddyfile. Then verify 443.
systemctl restart caddy
sleep 3
ss -tln | grep -q ':443 ' || die "caddy is not listening on 443 after restart — check journalctl -u caddy"

log "done. Services bound to 127.0.0.1; Caddy fronts them with HTTPS."
log "  onboard: https://$HOST_ONBOARD/health   (local: curl 127.0.0.1:8001/health)"
log "  serve:   https://$HOST_SERVE/health      (engine_ready flips true after weights seed/load)"
log "  weights: seed-in pulled from the FS if present; seed-out syncs UP after the first warm."
