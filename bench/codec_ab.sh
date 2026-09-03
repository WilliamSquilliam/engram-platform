#!/usr/bin/env bash
# codec_ab.sh — does gentler cart compression fix k=3 composed-cart accuracy?
#
# Baseline already measured (cc_aggr, 3-bit K/V + rank-16): k=3 accuracy 8-9/15,
# k=1 15/15. This runs the SAME bench (same 13 docs, same 15 questions, same
# production retriever, same engine) at two gentler store dtypes:
#   cc_safe — 4-bit K/V + low-rank (the "less aggressive" preset, 3.48x)
#   bf16    — lossless control (isolates compression noise from composition
#             interference: if bf16 k=3 still fails, compression is exonerated)
# Each cycle: set CART_STORE_DTYPE + FORCE_REONBOARD on the engine env, restart,
# force-rebuild the bench carts through the engine, run accuracy at k=3 and k=1.
# Ends by RESTORING production state: default dtype (cc_aggr), carts rebuilt,
# force flag removed, engine healthy.
#
# Runs ON the GPU box as ubuntu (passwordless sudo), driven by nohup:
#   nohup bash codec_ab.sh > codec_ab.log 2>&1 &
set -euo pipefail
PY=/home/ubuntu/benchvenv/bin/python
ENVF=/etc/engram/serving.env
cd /home/ubuntu/bench

# The bench authenticates with the engine's own bearer token (never printed).
export ML_AUTH_TOKEN="$(sudo sed -n 's/^ML_AUTH_TOKEN=//p' "$ENVF" | head -1)"

# /health answers 200 as soon as the HTTP layer is up — engine_ready is a FIELD in
# the body and flips true only once the vLLM engine is actually serving. Gate on it
# (the bare curl -sf gate let the first onboard race the warmup and die on a 503).
wait_healthy() {
  for i in $(seq 1 90); do
    sleep 10
    if curl -sf -m 5 http://127.0.0.1:8002/health | grep -q '"engine_ready": *true'; then
      echo "[ab] engine ready after ~$((i * 10))s"
      return 0
    fi
  done
  echo "[ab] FATAL: engine not ready after 900s" >&2
  exit 1
}

# Retry wrapper for the onboard phase: the :8001 proxy can still 503 briefly around
# engine warm even after engine_ready flips (store/registry init).
onboard_retry() {
  for i in 1 2 3; do
    if $PY headtohead.py onboard; then return 0; fi
    echo "[ab] onboard attempt $i failed; retrying in 60s"
    sleep 60
  done
  echo "[ab] FATAL: onboard failed 3 times" >&2
  exit 1
}

# set_env <dtype|""> — swap the marked env block and bounce the engine. The marker
# comments sit on their OWN lines: systemd's EnvironmentFile does not strip trailing
# comments, so an inline "# codec-ab" would corrupt the value.
set_env() {
  sudo sed -i '/^# codec-ab begin$/,/^# codec-ab end$/d' "$ENVF"
  {
    echo "# codec-ab begin"
    [ -n "$1" ] && echo "CART_STORE_DTYPE=$1"
    echo "FORCE_REONBOARD=1"
    echo "# codec-ab end"
  } | sudo tee -a "$ENVF" >/dev/null
  sudo systemctl restart engram-serve
  wait_healthy
}

cycle() { # $1 = label
  echo "=== [$1] onboard (forced rebuild) ==="
  onboard_retry
  grep -o '"n_built": *[0-9]*' results/onboard.json || true
  cp results/onboard.json "results/onboard_$1.json"
  echo "=== [$1] accuracy k=3 ==="
  $PY headtohead.py accuracy --topk 3
  cp results/accuracy.json "results/accuracy_k3_$1.json"
  echo "=== [$1] accuracy k=1 ==="
  $PY headtohead.py accuracy --topk 1
  cp results/accuracy.json "results/accuracy_k1_$1.json"
}

echo "[ab] ===== cycle 1: cc_safe ====="
set_env cc_safe
cycle cc_safe

echo "[ab] ===== cycle 2: bf16 ====="
set_env bf16
cycle bf16

echo "[ab] ===== restore: production default (cc_aggr) ====="
set_env "" # dtype unset -> s3 default cc_aggr; force stays on for this rebuild
onboard_retry
cp results/onboard.json results/onboard_restored_cc_aggr.json
sudo sed -i '/^# codec-ab begin$/,/^# codec-ab end$/d' "$ENVF" # drop FORCE_REONBOARD
sudo systemctl restart engram-serve
wait_healthy
echo "[ab] ALL DONE"
