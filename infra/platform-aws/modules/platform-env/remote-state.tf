# ===========================================================================
# The serving-unit contract flows in here. The GPU serving stack (infra/serving-aws)
# emits its four values as terraform outputs into its own state; we read that state
# and feed the values into the backend task env. A pivot to a different serving unit
# is a state-key change (var.serving_state_key) or a set of *_override vars — never a
# change here. Each value is: override-if-set, else the serving state's output.
# ===========================================================================
# count-gated: before serving-aws has EVER been applied its state object doesn't
# exist and this data source would hard-error. read_serving_state=false skips the
# read entirely (supply the *_override vars instead); flip it back to true after
# the serving unit's first apply.
data "terraform_remote_state" "serving" {
  count   = var.read_serving_state ? 1 : 0
  backend = "s3"
  config = {
    bucket = var.tfstate_bucket
    key    = var.serving_state_key
    region = var.region
  }
}

locals {
  serving = try(data.terraform_remote_state.serving[0].outputs, {})

  ml_service_url        = var.ml_service_url_override != "" ? var.ml_service_url_override : try(local.serving.ml_service_url, "")
  inference_service_url = var.inference_service_url_override != "" ? var.inference_service_url_override : try(local.serving.inference_service_url, "")
  ml_auth_token         = var.ml_auth_token_override != "" ? var.ml_auth_token_override : try(local.serving.ml_auth_token, "")
  model_registry_json   = var.model_registry_json_override != "" ? var.model_registry_json_override : try(local.serving.model_registry_json, "[]")
  cartridge_bucket      = var.cartridge_store_bucket_override != "" ? var.cartridge_store_bucket_override : try(local.serving.cart_bucket, "")
  cartridge_prefix      = try(local.serving.cart_store_prefix, "cartridges")

  name_prefix = "${var.env}-engram"
}
