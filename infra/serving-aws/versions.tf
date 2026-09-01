# Terraform + provider pins for the AWS GPU serving unit.
#
# State lives in the SAME encrypted, versioned S3 backend the sibling infra uses
# (cartridge-tfstate-808379776072-us-east-1 + the cartridge-tflock DynamoDB lock),
# but under its OWN key ("serving-aws/terraform.tfstate") so this swappable unit can be
# applied / destroyed independently of the control-plane stack. The state holds the
# generated ML_AUTH_TOKEN (random_password), so it must not live on one laptop.
#
# `terraform init -backend=false` runs the validate/fmt checks with no AWS creds; a real
# apply uses `terraform init` against the backend below.
terraform {
  required_version = ">= 1.7.0"

  backend "s3" {
    bucket         = "cartridge-tfstate-808379776072-us-east-1"
    key            = "serving-aws/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "cartridge-tflock"
    encrypt        = true
  }

  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.60" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }
}
