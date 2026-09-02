# ===========================================================================
# UAT root — reads the SHARED common stack (cluster / ALB / ECR / cert) via remote
# state and instantiates the platform-env module with env=uat + the uat-* hosts.
# UAT shares the ONE GPU serving unit with prod (default: serving-aws remote state),
# so keep bulk onboards off during customer hours — see README guardrails.
# ===========================================================================
data "terraform_remote_state" "common" {
  backend = "s3"
  config = {
    bucket = var.tfstate_bucket
    key    = "platform-aws/common.tfstate"
    # data-source creds resolve like backend creds: pin the profile
    profile = "Engram-Dynamics"
    region  = var.region
  }
}

locals {
  common = data.terraform_remote_state.common.outputs
}

module "env" {
  source = "../../modules/platform-env"

  env    = "uat"
  region = var.region

  # shared handles from common
  vpc_id                = local.common.vpc_id
  subnet_ids            = local.common.subnet_ids
  cluster_arn           = local.common.cluster_arn
  alb_security_group_id = local.common.alb_security_group_id
  https_listener_arn    = local.common.https_listener_arn

  # images (repo urls from common + the tags this root runs)
  backend_image  = "${local.common.ecr_backend_repo_url}:${var.backend_image_tag}"
  frontend_image = "${local.common.ecr_frontend_repo_url}:${var.frontend_image_tag}"

  # hosts + a listener-rule priority band distinct from prod's
  app_host                    = "uat-app.engramdynamics.org"
  api_host                    = "uat-api.engramdynamics.org"
  listener_rule_base_priority = 100

  # app config
  platform_admin_email  = var.platform_admin_email
  bootstrap_admin_email = var.bootstrap_admin_email
  google_client_id      = var.google_client_id
  google_client_secret  = var.google_client_secret

  # serving-unit wiring: gate the remote-state read (false until serving-aws's
  # first apply) + per-value overrides (empty = use the state's outputs)
  read_serving_state = var.read_serving_state
  # serving-unit overrides
  ml_service_url_override        = var.ml_service_url_override
  inference_service_url_override = var.inference_service_url_override
  ml_auth_token_override         = var.ml_auth_token_override
  model_registry_json_override   = var.model_registry_json_override
  tfstate_bucket                 = var.tfstate_bucket
}
