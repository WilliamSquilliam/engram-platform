# ===========================================================================
# PROD root — reads the SHARED common stack (cluster / ALB / ECR / cert) via remote
# state and instantiates the platform-env module with env=prod + the prod hosts.
# Deploy the SAME image sha here that was verified in UAT (see README flow).
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

  env    = "prod"
  region = var.region

  # shared handles from common
  vpc_id                = local.common.vpc_id
  subnet_ids            = local.common.subnet_ids
  cluster_arn           = local.common.cluster_arn
  alb_security_group_id = local.common.alb_security_group_id
  https_listener_arn    = local.common.https_listener_arn

  # images
  backend_image  = "${local.common.ecr_backend_repo_url}:${var.backend_image_tag}"
  frontend_image = "${local.common.ecr_frontend_repo_url}:${var.frontend_image_tag}"

  # hosts + a listener-rule priority band distinct from uat's
  app_host                    = "app.engramdynamics.org"
  api_host                    = "api.engramdynamics.org"
  listener_rule_base_priority = 200

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
