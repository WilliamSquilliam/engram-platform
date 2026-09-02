#!/usr/bin/env bash
# ============================================================================
# logs.sh — one door into every log the platform produces, for a human or an
# agent doing prod support. No dashboards required; everything is reachable
# from this machine with the aws CLI (profile Engram-Dynamics) + the SSH key
# the serving scripts already manage.
#
# WHERE THE LOGS LIVE (the map this script wraps):
#   * Control plane (Fargate)  -> CloudWatch Logs, group /ecs/<env>-engram/<svc>
#       svc = backend | frontend, env = uat | prod. Retention set by terraform
#       (platform-env log_retention_days). Structured app logs + uvicorn access.
#   * GPU box (Lambda Cloud)   -> journald on the box, via SSH with
#       infra/serving-lambda/.keys/engram-lambda and the IP in .state/current.json.
#       Units: engram-serve (vLLM engine), engram-onboard (onboarding worker),
#       caddy (TLS edge), engram-seed-in/out (weight seeding), plus
#       /var/log/engram-self-provision.log on button-started boxes.
#
# Usage:
#   bash infra/logs.sh uat backend  [--since 15m] [--filter ERROR]   # tail CloudWatch
#   bash infra/logs.sh prod frontend --since 1h
#   bash infra/logs.sh gpu serve    [--since "30 min ago"] [-f]      # journald over SSH
#   bash infra/logs.sh gpu onboard | gpu caddy | gpu seed | gpu provision
#   bash infra/logs.sh gpu health                                    # both /health bodies
# ============================================================================
set -euo pipefail

# Git Bash on Windows rewrites args that start with "/" into C:/... paths, which
# corrupts CloudWatch log-group names like /ecs/uat-engram/backend. Disable the
# conversion; both vars are ignored on real Linux.
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*"

HERE="$(cd "$(dirname "$0")" && pwd)"
PROFILE="${AWS_PROFILE:-Engram-Dynamics}"
REGION="${AWS_REGION:-us-east-1}"
SSH_KEY="$HERE/serving-lambda/.keys/engram-lambda"
STATE_FILE="$HERE/serving-lambda/.state/current.json"

die(){ echo "[logs] ERROR: $*" >&2; exit 1; }

TARGET="${1:-}"; shift || true
KIND="${1:-}"; shift || true

case "$TARGET" in
  uat|prod)
    case "$KIND" in backend|frontend) ;; *) die "usage: logs.sh $TARGET <backend|frontend> [--since 15m] [--filter PATTERN]";; esac
    SINCE="15m"; FILTER=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --since)  SINCE="$2"; shift 2 ;;
        --filter) FILTER="$2"; shift 2 ;;
        *) die "unknown arg $1" ;;
      esac
    done
    GROUP="/ecs/${TARGET}-engram/${KIND}"
    echo "[logs] CloudWatch $GROUP (since $SINCE${FILTER:+, filter '$FILTER'})"
    if [ -n "$FILTER" ]; then
      aws --profile "$PROFILE" --region "$REGION" logs tail "$GROUP" --since "$SINCE" --filter-pattern "$FILTER" --format short
    else
      aws --profile "$PROFILE" --region "$REGION" logs tail "$GROUP" --since "$SINCE" --format short
    fi
    ;;

  gpu)
    [ -f "$SSH_KEY" ] || die "no SSH key at $SSH_KEY (run serving-lambda/scripts/launch.sh once)"
    IP="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("ip",""))' "$STATE_FILE" 2>/dev/null || true)"
    [ -n "$IP" ] || die "no box IP in $STATE_FILE — is the GPU running? (Platform Admin tab or scripts/launch.sh)"
    SSH=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "ubuntu@$IP")
    case "$KIND" in
      serve)     "${SSH[@]}" "sudo journalctl -u engram-serve --no-pager -o cat $*" ;;
      onboard)   "${SSH[@]}" "sudo journalctl -u engram-onboard --no-pager -o cat $*" ;;
      caddy)     "${SSH[@]}" "sudo journalctl -u caddy --no-pager -o cat $*" ;;
      seed)      "${SSH[@]}" "sudo journalctl -u engram-seed-in -u engram-seed-out --no-pager -o cat $*" ;;
      provision) "${SSH[@]}" "sudo cat /var/log/engram-self-provision.log 2>/dev/null || echo '(no self-provision log — box was provisioned by an operator, see provision.sh output)'" ;;
      health)    "${SSH[@]}" 'echo "--- serve (:8002) ---"; curl -sS -m 5 http://127.0.0.1:8002/health; echo; echo "--- onboard (:8001) ---"; curl -sS -m 5 http://127.0.0.1:8001/health; echo; echo "--- gpu ---"; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader' ;;
      *) die "usage: logs.sh gpu <serve|onboard|caddy|seed|provision|health> [journalctl args e.g. --since '30 min ago' -f]" ;;
    esac
    ;;

  *) die "usage: logs.sh <uat|prod> <backend|frontend> | logs.sh gpu <serve|onboard|caddy|seed|provision|health>" ;;
esac
