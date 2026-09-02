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

variable "backend_image_tag" {
  description = "Backend image tag to run. Default 'latest' for first bring-up; deploy.sh sets the git sha (the SAME sha proven in UAT)."
  type        = string
  default     = "latest"
}

variable "frontend_image_tag" {
  description = "Frontend image tag to run (built per-env as <sha>-prod). Default 'latest-prod'."
  type        = string
  default     = "latest-prod"
}

# PLATFORM_ADMIN_EMAIL — the founder / cross-tenant superuser promoted at startup.
variable "platform_admin_email" {
  description = "PLATFORM_ADMIN_EMAIL for prod."
  type        = string
  default     = "will.stephenson@engramdynamics.org"
}

variable "bootstrap_admin_email" {
  description = "One-shot operator seed email for prod (open registration is OFF)."
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

# --- optional serving-unit override (defaults read serving-aws remote state) ---
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

variable "read_serving_state" {
  description = "Pass-through: read serving-aws remote state (false until that stack's first apply)."
  type        = bool
  default     = true
}
