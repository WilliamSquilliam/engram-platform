#!/usr/bin/env bash
# ============================================================================
# deploy.sh <env> [sha] — roll an env's two services to a set of image tags.
#
# For each service (backend, frontend) it:
#   1. reads the CURRENT task definition (registered by terraform),
#   2. swaps ONLY the container image to the requested tag,
#   3. registers a new task-def revision,
#   4. update-service --task-definition <new-rev> --force-new-deployment,
#   5. waits for the service to reach steady state.
#
# Image tags (matching build_push.sh):
#   backend  = <repo>:<sha>
#   frontend = <repo>:<sha>-<env>
#
# terraform still OWNS the task defs; a deploy just points the service at a new
# revision. The next `terraform apply` re-asserts the terraform-managed revision
# (same image if you pass the same sha via -var, or it rolls forward) — so pin the
# tag in the env's tfvars once a sha is promoted, or let deploy.sh drive day-to-day.
#
# Prereqs: aws CLI (profile Engram-Dynamics), terraform, git, python3 (JSON edit).
#
# Usage:
#   bash infra/platform-aws/deploy.sh uat            # current git sha
#   bash infra/platform-aws/deploy.sh prod abc1234   # a specific (UAT-proven) sha
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
PROFILE="${AWS_PROFILE:-Engram-Dynamics}"
REGION="${AWS_REGION:-us-east-1}"
AWS="aws --profile $PROFILE --region $REGION"

log(){ echo "[deploy] $*"; }
die(){ echo "[deploy] ERROR: $*" >&2; exit 1; }

command -v aws >/dev/null || die "aws CLI not on PATH"
command -v terraform >/dev/null || die "terraform not on PATH"
command -v python3 >/dev/null || die "python3 not on PATH"

ENV="${1:-}"
case "$ENV" in
  uat|prod) ;;
  *) die "usage: deploy.sh <uat|prod> [sha]" ;;
esac
SHA="${2:-$(git -C "$REPO_ROOT" rev-parse --short HEAD)}"
[ -n "$SHA" ] || die "could not resolve git sha"
log "env=$ENV sha=$SHA"

# --- pull handles from the env's terraform outputs --------------------------
TF="terraform -chdir=$HERE/envs/$ENV"
tf(){ $TF output -raw "$1" 2>/dev/null; }
CLUSTER="$(tf cluster_name)";            [ -n "$CLUSTER" ] || die "no cluster_name output — apply envs/$ENV first"
BE_SERVICE="$(tf backend_service_name)"; [ -n "$BE_SERVICE" ] || die "no backend_service_name"
FE_SERVICE="$(tf frontend_service_name)";[ -n "$FE_SERVICE" ] || die "no frontend_service_name"
BE_FAMILY="$(tf backend_task_family)";   [ -n "$BE_FAMILY" ] || die "no backend_task_family"
FE_FAMILY="$(tf frontend_task_family)";  [ -n "$FE_FAMILY" ] || die "no frontend_task_family"

# ECR repo urls (from common) to form full image refs.
COMMON="terraform -chdir=$HERE/common"
BACKEND_REPO="$($COMMON output -raw ecr_backend_repo_url 2>/dev/null)"  || die "no ecr_backend_repo_url"
FRONTEND_REPO="$($COMMON output -raw ecr_frontend_repo_url 2>/dev/null)" || die "no ecr_frontend_repo_url"

BE_IMAGE="$BACKEND_REPO:$SHA"
FE_IMAGE="$FRONTEND_REPO:${SHA}-${ENV}"

# roll one service: read current task def, swap image, register, update, wait.
roll(){
  local family="$1" service="$2" image="$3"
  log "$service: reading current task def ($family)"
  local current
  current="$($AWS ecs describe-task-definition --task-definition "$family" \
    --query 'taskDefinition' --output json)" || die "describe-task-definition $family failed"

  # Produce a register-task-definition input from the current def: drop the
  # server-managed fields and set the (single) container's image to the new tag.
  local newdef
  newdef="$(printf '%s' "$current" | python3 - "$image" <<'PY'
import json, sys
image = sys.argv[1]
td = json.load(sys.stdin)
for f in ("taskDefinitionArn","revision","status","requiresAttributes",
          "compatibilities","registeredAt","registeredBy","deregisteredAt"):
    td.pop(f, None)
# single-container task defs here; point the essential container at the new image.
for c in td["containerDefinitions"]:
    c["image"] = image
json.dump(td, sys.stdout)
PY
)"

  local arn
  arn="$($AWS ecs register-task-definition --cli-input-json "$newdef" \
    --query 'taskDefinition.taskDefinitionArn' --output text)" || die "register-task-definition failed"
  log "$service: registered $arn (image=$image)"

  $AWS ecs update-service --cluster "$CLUSTER" --service "$service" \
    --task-definition "$arn" --force-new-deployment >/dev/null || die "update-service $service failed"
  log "$service: update-service issued; waiting for steady state …"
  $AWS ecs wait services-stable --cluster "$CLUSTER" --services "$service" || die "$service did not stabilize"
  log "$service: STABLE on $arn"
}

roll "$BE_FAMILY" "$BE_SERVICE" "$BE_IMAGE"
roll "$FE_FAMILY" "$FE_SERVICE" "$FE_IMAGE"

APP_URL="$(tf app_url)"
log "DONE. $ENV is running sha $SHA."
log "Verify: $APP_URL  (log in; check the health + a chat round-trip)."
