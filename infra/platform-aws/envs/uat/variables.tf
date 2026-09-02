variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "Local AWS CLI profile (Engram Dynamics account)."
  type        = string
  default     = "Engram-Dynamics"
}

variable "tfstate_bucket" {
  description = "Shared S3 backend bucket (holds common + serving states this root reads)."
  type        = string
  default     = "cartridge-tfstate-808379776072-us-east-1"
}

# --- image tags deploy.sh rolls (defaults let a first apply stand the stack up on
#     :latest-shaped tags; deploy.sh then points services at real git-sha tags) ----
variable "backend_image_tag" {
  description = "Backend image tag to run. Default 'latest' for first bring-up; deploy.sh sets the git sha."
  type        = string
  default     = "latest"
}

variable "frontend_image_tag" {
  description = "Frontend image tag to run (built per-env as <sha>-uat). Default 'latest-uat'."
  type        = string
  default     = "latest-uat"
}

# --- optional app config passthroughs ---------------------------------------
variable "platform_admin_email" {
  description = "PLATFORM_ADMIN_EMAIL for UAT."
  type        = string
  default     = ""
}

variable "bootstrap_admin_email" {
  description = "One-shot operator seed email for UAT."
  type        = string
  default     = ""
}

variable "google_client_id" {
  description = "Google OAuth client id (optional)."
  type        = string
  default     = ""
}

variable "google_client_secret" {
  description = "Google OAuth client secret (optional)."
  type        = string
  default     = ""
  sensitive   = true
}

# --- optional serving-unit override: point UAT at an ephemeral serving box ----
variable "ml_service_url_override" {
  type    = string
  default = ""
}
variable "inference_service_url_override" {
  type    = string
  default = ""
}
variable "ml_auth_token_override" {
  type      = string
  default   = ""
  sensitive = true
}
variable "model_registry_json_override" {
  type    = string
  default = ""
}

variable "cartridge_store_bucket_override" {
  description = "Pass-through: CARTRIDGE_STORE_BUCKET override (the Lambda-unit cart bucket)."
  type        = string
  default     = ""
}

variable "read_serving_state" {
  description = "Pass-through: read serving-aws remote state (false until that stack's first apply)."
  type        = bool
  default     = true
}

# --- GPU control plane (platform-admin start/stop of the Lambda serving box) ---
variable "lambda_api_key" {
  description = "Pass-through: Lambda Cloud API key (empty = GPU panel off)."
  type        = string
  default     = ""
  sensitive   = true
}

variable "cloudflare_api_token" {
  description = "Pass-through: Cloudflare DNS-edit token for gpu/gpu-onboard A-record reconcile."
  type        = string
  default     = ""
  sensitive   = true
}

variable "cloudflare_zone_id" {
  description = "Pass-through: Cloudflare zone id for engramdynamics.org."
  type        = string
  default     = ""
}

# --- document connectors ------------------------------------------------------
variable "connector_enc_key" {
  type      = string
  default   = ""
  sensitive = true
}
variable "sharepoint_client_id" {
  type    = string
  default = ""
}
variable "sharepoint_client_secret" {
  type      = string
  default   = ""
  sensitive = true
}
