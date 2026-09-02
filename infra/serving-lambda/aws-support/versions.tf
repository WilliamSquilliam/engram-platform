# Terraform + provider pins for the Lambda serving unit's AWS-side support stack.
#
# The COMPUTE for this unit runs on Lambda Cloud (lambda.ai), not AWS. This tiny
# stack owns only the AWS pieces the Lambda box needs: the durable S3 cartridge
# store and a bucket-scoped IAM user whose access key the box uses to reach it
# (a Lambda instance has no AWS instance-profile, so it authenticates with a
# least-privilege key injected into serving.env by provision.sh).
#
# State lives in the SAME encrypted, versioned S3 backend the sibling infra uses
# (cartridge-tfstate-808379776072-us-east-1 + the cartridge-tflock DynamoDB lock),
# under its OWN key ("serving-lambda/aws-support.tfstate") so it applies/destroys
# independently. The state holds the IAM secret access key (sensitive), so it must
# not live on one laptop.
#
# `terraform init -backend=false` runs the validate/fmt checks with no AWS creds;
# a real apply uses `terraform init` against the backend below.
terraform {
  required_version = ">= 1.7.0"

  backend "s3" {
    # Backend creds resolve separately from the provider: pin the profile here
    # too or terraform falls back to the default chain (IMDS) and fails locally.
    profile        = "Engram-Dynamics"
    bucket         = "cartridge-tfstate-808379776072-us-east-1"
    key            = "serving-lambda/aws-support.tfstate"
    region         = "us-east-1"
    dynamodb_table = "cartridge-tflock"
    encrypt        = true
  }

  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.60" }
  }
}
