# ===========================================================================
# THE SWAPPABLE-UNIT CONTRACT. The control plane consumes exactly four values
# (backend/app/config.py + backend/app/serving.py); this unit emits them here.
# Any serving stack that produces these four is a drop-in replacement — a pivot
# is a variable change or a sibling module, never a product-code change.
#
#   ml_service_url        :8001  onboarding / train worker  -> config.ML_SERVICE_URL
#   inference_service_url :8002  vLLM resident-KV serve      -> config.INFERENCE_SERVICE_URL
#   ml_auth_token         shared bearer (sensitive)         -> config.ML_AUTH_TOKEN
#   model_registry_json   tier -> model_ref mapping         -> serving.MODEL_REGISTRY_JSON
# ===========================================================================

locals {
  # Private IP is stable across stop/start, so these URLs keep working after an
  # idle stop + later start (the weights persist on EBS). A co-located control
  # plane reaches the box at its private IP over the VPC.
  serve_host = aws_instance.gpu.private_ip

  # MODEL_REGISTRY_JSON — the exact ModelTier shape serving.py parses:
  # id/label/description/model_ref/precision/context_tokens/enabled. One "balanced"
  # tier, enabled, pointing at the served model_ref. Other tiers stay placeholders
  # (control-plane default) until more boxes exist.
  model_registry = [
    {
      id             = "balanced"
      label          = "Balanced"
      description    = "Best grounded accuracy per cost. The default."
      model_ref      = var.model_ref
      precision      = var.model_precision
      context_tokens = var.context_tokens
      enabled        = true
    }
  ]
}

output "ml_service_url" {
  description = "Contract value 1/4 -> config.ML_SERVICE_URL. Onboarding / train worker on :8001."
  value       = "http://${local.serve_host}:8001"
}

output "inference_service_url" {
  description = "Contract value 2/4 -> config.INFERENCE_SERVICE_URL. vLLM Inference Service on :8002."
  value       = "http://${local.serve_host}:8002"
}

output "ml_auth_token" {
  description = "Contract value 3/4 -> config.ML_AUTH_TOKEN. Shared bearer both serving processes enforce. Read with: terraform output -raw ml_auth_token"
  value       = random_password.ml_auth_token.result
  sensitive   = true
}

output "model_registry_json" {
  description = "Contract value 4/4 -> serving.MODEL_REGISTRY_JSON. The product model tiers (id/label/model_ref/precision/context_tokens/enabled)."
  value       = jsonencode(local.model_registry)
}

# --- operational outputs ----------------------------------------------------
output "instance_id" {
  description = "The GPU box instance id (provision.sh / smoke.sh / stop-start target)."
  value       = aws_instance.gpu.id
}

output "private_ip" {
  description = "Stable private IP the serving URLs resolve to (survives stop/start)."
  value       = aws_instance.gpu.private_ip
}

output "cart_bucket" {
  description = "S3 durable cartridge store bucket (CARTRIDGE_STORE_BUCKET)."
  value       = local.cart_bucket
}

output "provision_bucket" {
  description = "S3 bucket provision.sh uploads the install bundle to."
  value       = local.provision_b
}

output "cart_store_prefix" {
  description = "Key prefix under the cart bucket for cart blobs (CARTRIDGE_STORE_PREFIX)."
  value       = var.cart_store_prefix
}

output "tensor_parallel" {
  description = "vLLM tensor-parallel size (provision.sh reads it to set VLLM_TP on the box)."
  value       = var.tensor_parallel
}

# ---------------------------------------------------------------------------
# Ready-to-paste env block for the control plane. Everything except the bearer
# token (sensitive — read it separately) is inlined so the operator can drop this
# straight into the backend's env. Print with:  terraform output -raw serving_unit_env
# The token line uses a placeholder so a plain `terraform output` never leaks it;
# fill it from:  terraform output -raw ml_auth_token
# ---------------------------------------------------------------------------
output "serving_unit_env" {
  description = "Paste into the control plane env. Fill ML_AUTH_TOKEN from `terraform output -raw ml_auth_token`."
  value       = <<-EOT
    # --- Engram serving unit (AWS GPU, ${var.instance_type}) — the four contract values ---
    INFERENCE_BACKEND=vllm
    ML_SERVICE_URL=http://${local.serve_host}:8001
    INFERENCE_SERVICE_URL=http://${local.serve_host}:8002
    ML_AUTH_TOKEN=__read_from: terraform output -raw ml_auth_token__
    MODEL_REGISTRY_JSON=${jsonencode(local.model_registry)}
    # --- cartridge store the unit reads/writes (control plane shares it for retrieval hydration) ---
    CARTRIDGE_STORE_BACKEND=s3
    CARTRIDGE_STORE_BUCKET=${local.cart_bucket}
    CARTRIDGE_STORE_PREFIX=${var.cart_store_prefix}
    # --- measured $/query uses the real running price of this box ---
    # box on-demand cost ~= $${var.hourly_cost_usd}/hr
  EOT
}
