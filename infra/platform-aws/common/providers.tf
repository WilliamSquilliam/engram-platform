# AWS provider + account guard. Mirrors infra/serving-aws/providers.tf: the platform
# deploys ONLY to the Engram Dynamics account (808379776072) via the Engram-Dynamics
# local profile, and a `check` block refuses to plan/apply if the resolved credentials
# point anywhere else (e.g. AWS_PROFILE / env creds overriding the profile).
provider "aws" {
  region  = var.region
  profile = var.aws_profile
  default_tags {
    tags = {
      Project   = "engram-platform"
      Component = "platform-aws-common"
      ManagedBy = "terraform"
    }
  }
}

data "aws_caller_identity" "current" {}

variable "expected_account_id" {
  description = "AWS account this stack must deploy to. Empty disables the check."
  type        = string
  default     = "808379776072" # Engram Dynamics
}

check "correct_aws_account" {
  assert {
    condition     = var.expected_account_id == "" || data.aws_caller_identity.current.account_id == var.expected_account_id
    error_message = "Wrong AWS account: credentials resolve to ${data.aws_caller_identity.current.account_id}, expected ${var.expected_account_id} (the Engram Dynamics account). Check AWS_PROFILE / var.aws_profile."
  }
}
