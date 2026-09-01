# UAT root. Own state key so it applies independently of common and prod. Same
# shared S3 backend + lock as every other stack.
terraform {
  required_version = ">= 1.7.0"

  backend "s3" {
    bucket         = "cartridge-tfstate-808379776072-us-east-1"
    key            = "platform-aws/uat.tfstate"
    region         = "us-east-1"
    dynamodb_table = "cartridge-tflock"
    encrypt        = true
  }

  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.60" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }
}
