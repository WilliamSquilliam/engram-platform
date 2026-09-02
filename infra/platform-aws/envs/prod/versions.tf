# PROD root. Own state key so it applies independently of common and uat. Same
# shared S3 backend + lock as every other stack.
terraform {
  required_version = ">= 1.7.0"

  backend "s3" {
    # Backend creds resolve separately from the provider: pin the profile here
    # too or terraform falls back to the default chain (IMDS) and fails locally.
    profile        = "Engram-Dynamics"
    bucket         = "cartridge-tfstate-808379776072-us-east-1"
    key            = "platform-aws/prod.tfstate"
    region         = "us-east-1"
    dynamodb_table = "cartridge-tflock"
    encrypt        = true
  }

  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.60" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }
}
