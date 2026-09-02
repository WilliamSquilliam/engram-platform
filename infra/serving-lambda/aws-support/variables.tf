# ---------------------------------------------------------------------------
# Inputs for the Lambda unit's AWS-side support stack. The GPU/model pivot lives
# in the Lambda-side scripts + vars; this stack only owns the S3 cart store and
# the bucket-scoped IAM user the Lambda box uses to reach it.
# ---------------------------------------------------------------------------

variable "region" {
  description = "AWS region for the cart bucket. us-east-1 keeps it next to the sibling serving-aws bucket."
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "Local AWS CLI profile. Runs in AWS account 808379776072 (Engram Dynamics)."
  type        = string
  default     = "Engram-Dynamics"
}

variable "name" {
  description = "Name prefix for this support stack's resources."
  type        = string
  default     = "engram-lambda-serving"
}

# ---------------------------------------------------------------------------
# Cartridge store bucket. The onboarding worker (:8001) on the Lambda box WRITES
# CAG carts here and the vLLM serve engine (:8002) READS them by id — the durable
# S3 tier (SSD/persistent-FS on the box are the ephemeral/warm tiers). A dedicated
# bucket per serving unit keeps the Lambda unit self-contained; set to an existing
# bucket name to reuse the control plane's storage bucket instead.
# ---------------------------------------------------------------------------
variable "cart_bucket_name" {
  description = "S3 bucket for the durable cartridge store. Empty = create engram-carts-lambda-<acct>-<region> (versioned)."
  type        = string
  default     = ""
}

variable "cart_store_prefix" {
  description = "Key prefix under the cart bucket for cart blobs (informational; the box sets CARTRIDGE_STORE_PREFIX)."
  type        = string
  default     = "cartridges"
}
