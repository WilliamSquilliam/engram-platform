# ===========================================================================
# Per-env document store. The backend runs with PLATFORM_STORAGE_BACKEND=s3 and
# PLATFORM_S3_BUCKET pointed here (see ecs.tf app_env). Versioned + encrypted +
# private. This is the tenant DOC store — distinct from the serving unit's cart
# store bucket (which the GPU box owns and the control plane only references).
# ===========================================================================
data "aws_caller_identity" "current" {}

locals {
  storage_bucket = "${local.name_prefix}-storage-${data.aws_caller_identity.current.account_id}-${var.region}"
}

resource "aws_s3_bucket" "storage" {
  bucket = local.storage_bucket
  # Not force-destroyable: tenant documents are durable data, not scratch.
  force_destroy = false
}

resource "aws_s3_bucket_versioning" "storage" {
  bucket = aws_s3_bucket.storage.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "storage" {
  bucket = aws_s3_bucket.storage.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "storage" {
  bucket                  = aws_s3_bucket.storage.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
