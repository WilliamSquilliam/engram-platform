# Terraform + provider pins for the SHARED platform infrastructure (applied once,
# consumed by both env stacks). Same encrypted, versioned S3 backend as the sibling
# serving-aws stack (cartridge-tfstate-808379776072-us-east-1 + cartridge-tflock),
# under its OWN key so common / uat / prod apply independently.
#
# `terraform init -backend=false` runs validate/fmt with no AWS creds; a real apply
# uses `terraform init` against the backend below.
terraform {
  required_version = ">= 1.7.0"

  backend "s3" {
    bucket         = "cartridge-tfstate-808379776072-us-east-1"
    key            = "platform-aws/common.tfstate"
    region         = "us-east-1"
    dynamodb_table = "cartridge-tflock"
    encrypt        = true
  }

  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.60" }
  }
}
