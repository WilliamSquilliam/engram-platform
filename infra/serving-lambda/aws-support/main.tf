# ===========================================================================
# AWS-side support for the Lambda Cloud serving unit.
#
# The compute (1x B200 / 2x H100) runs on Lambda Cloud, which has no AWS
# instance-profile mechanism. So the durable cartridge store still lives in S3
# (same as serving-aws), but the box reaches it with a bucket-scoped IAM ACCESS
# KEY that provision.sh injects into /etc/engram/serving.env. This stack owns:
#   * the versioned, encrypted cart bucket, and
#   * an IAM user + inline policy scoped to THAT bucket only + an access key.
# outputs.tf emits the bucket name + the key id/secret (sensitive).
# ===========================================================================

locals {
  acct        = data.aws_caller_identity.current.account_id
  cart_bucket = var.cart_bucket_name != "" ? var.cart_bucket_name : "engram-carts-lambda-${local.acct}-${var.region}"
  create_cart = var.cart_bucket_name == ""
  # ARNs the IAM policy is scoped to (bucket + its objects, nothing else).
  cart_bucket_arn = "arn:aws:s3:::${local.cart_bucket}"
}

# ---------------------------------------------------------------------------
# S3 — durable cartridge store (versioned, encrypted, private). Created here
# unless cart_bucket_name points at an existing bucket.
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "carts" {
  count         = local.create_cart ? 1 : 0
  bucket        = local.cart_bucket
  force_destroy = false # cart blobs are the durable memory tier — never force-destroyable
}

resource "aws_s3_bucket_versioning" "carts" {
  count  = local.create_cart ? 1 : 0
  bucket = aws_s3_bucket.carts[0].id
  versioning_configuration {
    status = "Enabled"
  }
}

# Versioning is a recycle bin, not an eternal archive: expire deleted cart
# versions after 30 days and drop abandoned multipart parts after 7.
resource "aws_s3_bucket_lifecycle_configuration" "carts" {
  count  = local.create_cart ? 1 : 0
  bucket = aws_s3_bucket.carts[0].id
  rule {
    id     = "expire-noncurrent-versions"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

resource "aws_s3_bucket_public_access_block" "carts" {
  count                   = local.create_cart ? 1 : 0
  bucket                  = aws_s3_bucket.carts[0].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "carts" {
  count  = local.create_cart ? 1 : 0
  bucket = aws_s3_bucket.carts[0].id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ---------------------------------------------------------------------------
# IAM — a dedicated serving user for the Lambda box. It gets ONE inline policy
# scoped to the cart bucket only (Get/Put/Delete/List) and a long-lived access
# key. The box has no other AWS reach: if the key leaks, the blast radius is this
# one bucket. Rotate by tainting aws_iam_access_key.serving and re-provisioning.
# ---------------------------------------------------------------------------
resource "aws_iam_user" "serving" {
  name = "engram-lambda-serving"
  path = "/serving/"
  tags = { Purpose = "Lambda serving box cart-store access (bucket-scoped)" }
}

resource "aws_iam_user_policy" "cart_store" {
  name = "cart-store-rw"
  user = aws_iam_user.serving.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "CartStoreObjectsRW"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        Resource = ["${local.cart_bucket_arn}/*"]
      },
      {
        Sid      = "CartStoreListBucket"
        Effect   = "Allow"
        Action   = ["s3:ListBucket", "s3:GetBucketLocation"]
        Resource = [local.cart_bucket_arn]
      },
    ]
  })
}

resource "aws_iam_access_key" "serving" {
  user = aws_iam_user.serving.name
}
