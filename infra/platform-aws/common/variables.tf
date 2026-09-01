# ---------------------------------------------------------------------------
# Inputs for the SHARED platform layer. Applied once; both env stacks read its
# outputs via terraform_remote_state. Defaults stand up everything with no tfvars.
# ---------------------------------------------------------------------------

variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "Local AWS CLI profile. The platform runs in AWS account 808379776072 (Engram Dynamics)."
  type        = string
  default     = "Engram-Dynamics"
}

variable "name" {
  description = "Name prefix for shared platform resources (cluster, ALB, repos, cert)."
  type        = string
  default     = "engram-platform"
}

# ---------------------------------------------------------------------------
# Placement. The serving unit (infra/serving-aws) runs in the account's DEFAULT
# VPC; the platform envs MUST join the SAME VPC so the Fargate tasks reach the GPU
# box on its private IP over the VPC. Empty vpc_id = the region's default VPC
# (vpc-00f7fc0cc4895aa81, 172.31.0.0/16 — six public subnets across AZs). Override
# only if the serving unit is ever moved off the default VPC — then set both stacks
# to the same vpc_id. See README "The default-VPC coupling".
# ---------------------------------------------------------------------------
variable "vpc_id" {
  description = "VPC for the platform. Empty = the region's default VPC (same VPC the serving unit uses)."
  type        = string
  default     = ""
}

variable "subnet_ids" {
  description = "Subnets for the public ALB + Fargate tasks (>=2 AZs). Empty = all subnets in the chosen VPC."
  type        = list(string)
  default     = []
}

# ---------------------------------------------------------------------------
# DNS. The four public hosts fronted by the ONE shared ALB. The ACM certificate
# covers all four (DNS-validated); the operator adds the validation CNAMEs and the
# four host CNAMEs (-> ALB dns_name) in Cloudflare by hand — see the outputs and
# the README handoff. Change these only if the product domain changes.
# ---------------------------------------------------------------------------
variable "prod_app_host" {
  description = "Public host for the prod frontend."
  type        = string
  default     = "app.engramdynamics.org"
}

variable "prod_api_host" {
  description = "Public host for the prod backend API."
  type        = string
  default     = "api.engramdynamics.org"
}

variable "uat_app_host" {
  description = "Public host for the UAT frontend."
  type        = string
  default     = "uat-app.engramdynamics.org"
}

variable "uat_api_host" {
  description = "Public host for the UAT backend API."
  type        = string
  default     = "uat-api.engramdynamics.org"
}

# ---------------------------------------------------------------------------
# ALB idle timeout. LLM generations are long-lived requests; the 60s default
# returns 504s on long answers. 300s matches the proven serving stack's ALB.
# ---------------------------------------------------------------------------
variable "alb_idle_timeout" {
  description = "ALB idle timeout (seconds). High enough that long LLM generations don't 504."
  type        = number
  default     = 300
}

variable "log_retention_days" {
  description = "CloudWatch log retention for both envs' task logs."
  type        = number
  default     = 14
}
