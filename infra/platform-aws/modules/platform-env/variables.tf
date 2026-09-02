# ===========================================================================
# The reusable per-environment module. Instantiated once per env (uat, prod) by
# the thin roots in envs/<env>. Stands up everything an environment owns:
#   * two Fargate services (<env>-engram-backend :8000, <env>-engram-frontend :3000)
#   * per-env target groups + host-header listener rules on the shared ALB
#   * per-env RDS Postgres 16, S3 doc bucket, Secrets Manager entries
#   * task IAM (secrets read, S3 RW own bucket, logs) and the full backend env,
#     including the four serving-unit contract values from remote state.
# The SHARED pieces (cluster, ALB, ECR, cert) come in as inputs from the common
# stack (the env roots read them via terraform_remote_state).
# ===========================================================================

variable "env" {
  description = "Environment name: uat | prod. Namespaces every resource (<env>-engram-*)."
  type        = string
  validation {
    condition     = contains(["uat", "prod"], var.env)
    error_message = "env must be uat or prod."
  }
}

variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

# --- shared handles from the common stack -----------------------------------
variable "vpc_id" {
  description = "VPC id (from common; the shared default VPC)."
  type        = string
}

variable "subnet_ids" {
  description = "Subnets for the Fargate tasks + RDS subnet group (from common)."
  type        = list(string)
}

variable "cluster_arn" {
  description = "ECS cluster ARN (from common)."
  type        = string
}

variable "alb_security_group_id" {
  description = "The public ALB's security group id (from common) — task ingress source."
  type        = string
}

variable "https_listener_arn" {
  description = "The shared ALB :443 listener ARN (from common) — host-header rules attach here."
  type        = string
}

variable "backend_image" {
  description = "Full backend image ref (repo:tag). deploy.sh rolls this to a new git-sha tag."
  type        = string
}

variable "frontend_image" {
  description = "Full frontend image ref (repo:<sha>-<env>). deploy.sh rolls this."
  type        = string
}

# --- DNS: the two hosts this env answers on ---------------------------------
variable "app_host" {
  description = "Public frontend host for this env (e.g. app. / uat-app.)."
  type        = string
}

variable "api_host" {
  description = "Public backend-API host for this env (e.g. api. / uat-api.)."
  type        = string
}

# --- listener-rule priority band (must be unique per env on the shared listener)
variable "listener_rule_base_priority" {
  description = "Base priority for this env's two host-header rules (api = base, app = base+1). uat and prod use different bands so they never collide."
  type        = number
}

# --- RDS ---------------------------------------------------------------------
variable "db_instance_class" {
  description = "RDS instance class. db.t4g.micro is plenty for the control-plane metadata DB."
  type        = string
  default     = "db.t4g.micro"
}

variable "db_allocated_gb" {
  description = "RDS allocated storage (gp3, GB)."
  type        = number
  default     = 20
}

# --- Fargate task sizing (Fargate CPU/memory are paired; see valid combos) --
variable "backend_cpu" {
  description = "Backend task vCPU units (256 = 0.25 vCPU). Default 512 = 0.5 vCPU."
  type        = number
  default     = 512
}

variable "backend_memory" {
  description = "Backend task memory (MiB). Default 1024 = 1GB."
  type        = number
  default     = 1024
}

variable "frontend_cpu" {
  description = "Frontend task vCPU units. Default 256 = 0.25 vCPU."
  type        = number
  default     = 256
}

variable "frontend_memory" {
  description = "Frontend task memory (MiB). Default 512 = 0.5GB."
  type        = number
  default     = 512
}

variable "log_retention_days" {
  description = "CloudWatch retention for this env's task logs."
  type        = number
  default     = 14
}

# --- application config ------------------------------------------------------
variable "email_from" {
  description = "EMAIL_FROM for transactional email (SES). Must be a verified SES identity in prod."
  type        = string
  default     = "will.stephenson@engramdynamics.org"
}

variable "platform_admin_email" {
  description = "PLATFORM_ADMIN_EMAIL — promotes this already-existing user to platform_admin at startup. Empty = only the bootstrap admin."
  type        = string
  default     = ""
}

variable "bootstrap_admin_email" {
  description = "One-shot operator seed email (open registration is OFF in prod). Empty = no seed (seed a user by hand or flip ALLOW_REGISTRATION)."
  type        = string
  default     = ""
}

variable "google_client_id" {
  description = "Google OAuth client id (optional — enables 'Continue with Google'). Empty = hidden."
  type        = string
  default     = ""
}

variable "google_client_secret" {
  description = "Google OAuth client secret (optional). Injected as a task secret when set."
  type        = string
  default     = ""
  sensitive   = true
}

# --- serving-unit contract values (default: read from serving-aws remote state)
# The four values the control plane consumes. By default they flow in from the
# serving stack's state (serving-aws/terraform.tfstate); each can be overridden by
# a non-empty var (e.g. to point UAT at a throwaway ephemeral serving unit).
variable "ml_service_url_override" {
  description = "Override ML_SERVICE_URL (:8001). Empty = read serving-aws remote state."
  type        = string
  default     = ""
}

variable "inference_service_url_override" {
  description = "Override INFERENCE_SERVICE_URL (:8002). Empty = read serving-aws remote state."
  type        = string
  default     = ""
}

variable "ml_auth_token_override" {
  description = "Override ML_AUTH_TOKEN. Empty = read serving-aws remote state (sensitive)."
  type        = string
  default     = ""
  sensitive   = true
}

variable "model_registry_json_override" {
  description = "Override MODEL_REGISTRY_JSON. Empty = read serving-aws remote state."
  type        = string
  default     = ""
}

variable "serving_state_key" {
  description = "State key of the serving unit whose outputs supply the four contract values."
  type        = string
  default     = "serving-aws/terraform.tfstate"
}

variable "tfstate_bucket" {
  description = "S3 bucket holding all stack states (shared backend)."
  type        = string
  default     = "cartridge-tfstate-808379776072-us-east-1"
}

variable "cartridge_store_bucket_override" {
  description = "CARTRIDGE_STORE_BUCKET (the durable cart store the serving unit reads/writes). Empty = read serving-aws remote state cart_bucket."
  type        = string
  default     = ""
}

variable "read_serving_state" {
  description = "Read the serving unit's remote state for the ML-plane values. Set false when serving-aws has never been applied (its state object doesn't exist yet) and supply the *_override vars; flip back to true after its first apply."
  type        = bool
  default     = true
}

# --- GPU control plane (platform-admin start/stop of the Lambda serving box) ---
# All optional: empty leaves the GPU panel hidden (backend GPU_CONTROL_ENABLED is
# driven by LAMBDA_API_KEY presence). Injected as task secrets, never plain env.
variable "lambda_api_key" {
  description = "Lambda Cloud API key — lets platform-admin start/stop/status the GPU box. Empty = feature off."
  type        = string
  default     = ""
  sensitive   = true
}

variable "cloudflare_api_token" {
  description = "Cloudflare API token (zone DNS edit) — backend re-points the gpu/gpu-onboard A records after an unattended relaunch. Empty = no DNS reconcile."
  type        = string
  default     = ""
  sensitive   = true
}

variable "cloudflare_zone_id" {
  description = "Cloudflare zone id for engramdynamics.org (pairs with cloudflare_api_token)."
  type        = string
  default     = ""
}
