# ===========================================================================
# Outputs consumed by the Lambda-side scripts (provision.sh reads these to inject
# bucket-scoped AWS creds into the box's serving.env) and by emit-env.sh (the
# cart bucket flows to the control plane as CARTRIDGE_STORE_BUCKET so retrieval
# hydration shares the same store).
#
# The access key id + secret are SENSITIVE. Read them explicitly:
#   terraform output -raw serving_access_key_id
#   terraform output -raw serving_secret_access_key
# A plain `terraform output` shows them as <sensitive> and never leaks them.
# ===========================================================================

output "cart_bucket" {
  description = "S3 durable cartridge store bucket (CARTRIDGE_STORE_BUCKET)."
  value       = local.cart_bucket
}

output "cart_store_prefix" {
  description = "Key prefix under the cart bucket for cart blobs (CARTRIDGE_STORE_PREFIX)."
  value       = var.cart_store_prefix
}

output "cart_bucket_region" {
  description = "Region the cart bucket lives in (the box sets AWS_REGION to this for the S3 client)."
  value       = var.region
}

output "serving_access_key_id" {
  description = "IAM access key id for the bucket-scoped serving user. provision.sh injects it into serving.env."
  value       = aws_iam_access_key.serving.id
  sensitive   = true
}

output "serving_secret_access_key" {
  description = "IAM secret access key for the bucket-scoped serving user (sensitive). Read: terraform output -raw serving_secret_access_key"
  value       = aws_iam_access_key.serving.secret
  sensitive   = true
}
