# ===========================================================================
# Per-env task IAM. Two roles:
#   * execution role — pull images (ECR), read THIS env's secrets, write logs
#   * task role      — app runtime perms: RW this env's own doc bucket + read the
#                      shared cart store bucket (retrieval hydration reads carts the
#                      GPU box wrote). Scoped to those buckets only.
# ===========================================================================
data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# --- execution role ---
resource "aws_iam_role" "ecs_exec" {
  name               = "${local.name_prefix}-ecs-exec"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

resource "aws_iam_role_policy_attachment" "ecs_exec_managed" {
  role       = aws_iam_role.ecs_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_exec_secrets" {
  name = "read-app-secrets"
  role = aws_iam_role.ecs_exec.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = [for s in aws_secretsmanager_secret.app : s.arn]
    }]
  })
}

# --- task role ---
resource "aws_iam_role" "ecs_task" {
  name               = "${local.name_prefix}-ecs-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

# RW this env's own document bucket.
resource "aws_iam_role_policy" "ecs_task_storage" {
  name = "storage-bucket-rw"
  role = aws_iam_role.ecs_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
      Resource = [aws_s3_bucket.storage.arn, "${aws_s3_bucket.storage.arn}/*"]
    }]
  })
}

# Transactional email (invites / password resets / approvals) goes out via SES
# directly from the app (EMAIL_BACKEND=ses); without this the sends fail silently
# and the flows fall back to on-screen links.
resource "aws_iam_role_policy" "ecs_task_ses" {
  name = "ses-send"
  role = aws_iam_role.ecs_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["ses:SendEmail", "ses:SendRawEmail"]
      Resource = "*"
    }]
  })
}

# Read the shared cart store (only created when the serving state exposes a bucket).
# The control plane hydrates its retrieval index from carts the GPU box wrote there.
resource "aws_iam_role_policy" "ecs_task_cartstore" {
  count = local.cartridge_bucket != "" ? 1 : 0
  name  = "cart-store-read"
  role  = aws_iam_role.ecs_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:ListBucket"]
      Resource = ["arn:aws:s3:::${local.cartridge_bucket}", "arn:aws:s3:::${local.cartridge_bucket}/*"]
    }]
  })
}
