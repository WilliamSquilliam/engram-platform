#!/usr/bin/env bash
# ============================================================================
# build_push.sh — build + push the platform images to ECR.
#
#   backend : built ONCE (env-agnostic image), tagged with the short git sha.
#   frontend: built PER ENV, because NEXT_PUBLIC_API_URL is inlined at build time.
#             Two builds: one with the uat api host, one with the prod api host,
#             tagged <sha>-uat and <sha>-prod in the same repo.
#
# The image tags this produces feed deploy.sh <env> <sha>. Build order and the
# common-stack ECR repo URLs are read from terraform (envs/uat outputs cluster/
# repo info via common), so nothing is hardcoded.
#
# Prereqs: docker, aws CLI (profile Engram-Dynamics), terraform, git. The common
# stack must be applied (the ECR repos exist) — `terraform -chdir=common apply`.
#
# Usage:
#   bash infra/platform-aws/build_push.sh            # both images, both envs
#   SHA=abc1234 bash infra/platform-aws/build_push.sh   # pin a specific sha
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
PROFILE="${AWS_PROFILE:-Engram-Dynamics}"
REGION="${AWS_REGION:-us-east-1}"
AWS="aws --profile $PROFILE --region $REGION"

# Product hosts (kept in sync with the env roots' api_host). NEXT_PUBLIC_API_URL is
# the api host the SPA calls; it is baked into the frontend bundle per env.
UAT_API_URL="https://uat-api.engramdynamics.org"
PROD_API_URL="https://api.engramdynamics.org"

log(){ echo "[build_push] $*"; }
die(){ echo "[build_push] ERROR: $*" >&2; exit 1; }

command -v docker >/dev/null || die "docker not on PATH"
command -v aws >/dev/null || die "aws CLI not on PATH"
command -v terraform >/dev/null || die "terraform not on PATH"
command -v git >/dev/null || die "git not on PATH"

SHA="${SHA:-$(git -C "$REPO_ROOT" rev-parse --short HEAD)}"
[ -n "$SHA" ] || die "could not resolve git sha"
log "image sha: $SHA"

# --- ECR repo URLs from the applied common stack ----------------------------
COMMON="terraform -chdir=$HERE/common"
BACKEND_REPO="$($COMMON output -raw ecr_backend_repo_url 2>/dev/null)"  || die "no ecr_backend_repo_url — apply common first"
FRONTEND_REPO="$($COMMON output -raw ecr_frontend_repo_url 2>/dev/null)" || die "no ecr_frontend_repo_url — apply common first"
REGISTRY="${BACKEND_REPO%/*}"   # <acct>.dkr.ecr.<region>.amazonaws.com
log "backend repo:  $BACKEND_REPO"
log "frontend repo: $FRONTEND_REPO"

# --- ECR login --------------------------------------------------------------
log "logging in to ECR ($REGISTRY)"
$AWS ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY" >/dev/null \
  || die "ecr login failed"

# --- backend: built once (context = repo root, Dockerfile = backend/Dockerfile)
BACKEND_TAG="$BACKEND_REPO:$SHA"
log "building backend -> $BACKEND_TAG"
docker build -f "$REPO_ROOT/backend/Dockerfile" -t "$BACKEND_TAG" "$REPO_ROOT" || die "backend build failed"
log "pushing backend -> $BACKEND_TAG"
docker push "$BACKEND_TAG" || die "backend push failed"

# --- frontend: one build per env (NEXT_PUBLIC_API_URL differs) ---------------
# context = frontend/ (its Dockerfile COPYs from the build context root).
build_frontend(){
  local env="$1" api_url="$2"
  local tag="$FRONTEND_REPO:${SHA}-${env}"
  log "building frontend[$env] (NEXT_PUBLIC_API_URL=$api_url) -> $tag"
  docker build \
    -f "$REPO_ROOT/frontend/Dockerfile" \
    --build-arg "NEXT_PUBLIC_API_URL=$api_url" \
    -t "$tag" \
    "$REPO_ROOT/frontend" || die "frontend[$env] build failed"
  log "pushing frontend[$env] -> $tag"
  docker push "$tag" || die "frontend[$env] push failed"
}

build_frontend uat  "$UAT_API_URL"
build_frontend prod "$PROD_API_URL"

log "DONE. Images pushed:"
log "  backend      $BACKEND_REPO:$SHA"
log "  frontend uat $FRONTEND_REPO:${SHA}-uat"
log "  frontend prod $FRONTEND_REPO:${SHA}-prod"
log "Next: bash $HERE/deploy.sh uat $SHA   (verify at uat-app), then deploy.sh prod $SHA"
