# ===========================================================================
# The AWS GPU serving unit — ONE GPU box serving the cartridge stack:
#   :8001  onboarding worker  (ml_service/app.py — build carts, HF inference)
#   :8002  vLLM Inference Service (ml_service/vllm_inference.py — resident-KV serve)
# Both processes are systemd units installed by provision.sh -> bootstrap.sh.
# This file wires the box; the four contract outputs live in outputs.tf.
# ===========================================================================

# --- placement: default VPC + its first subnet unless overridden ------------
data "aws_vpc" "default" {
  count   = var.vpc_id == "" ? 1 : 0
  default = true
}

locals {
  vpc_id = var.vpc_id != "" ? var.vpc_id : data.aws_vpc.default[0].id
}

data "aws_subnets" "in_vpc" {
  count = var.subnet_id == "" ? 1 : 0
  filter {
    name   = "vpc-id"
    values = [local.vpc_id]
  }
}

data "aws_vpc" "selected" {
  id = local.vpc_id
}

locals {
  subnet_id     = var.subnet_id != "" ? var.subnet_id : data.aws_subnets.in_vpc[0].ids[0]
  acct          = data.aws_caller_identity.current.account_id
  cart_bucket   = var.cart_bucket_name != "" ? var.cart_bucket_name : "engram-carts-${local.acct}-${var.region}"
  create_cart_b = var.cart_bucket_name == ""
  provision_b   = var.provision_bucket_name != "" ? var.provision_bucket_name : "engram-serving-provision-${local.acct}-${var.region}"
  create_prov_b = var.provision_bucket_name == ""
}

# ---------------------------------------------------------------------------
# GPU AMI. The pinned DLAMI (var.gpu_ami_id) ships the NVIDIA driver + CUDA on
# Ubuntu 22.04; provision.sh only builds the Python vLLM venv on top. Verified
# 2026-09-01 that the pin still exists (aws ec2 describe-images). The data-source
# lookup below is the fallback for a fresh region / if the pin is ever retired:
# set gpu_ami_id="" to use the most-recent DLAMI instead.
# ---------------------------------------------------------------------------
variable "gpu_ami_id" {
  description = "Pinned DLAMI for the GPU box (empty = most-recent DLAMI lookup). Bumping it REPLACES the instance and wipes its EBS (model cache); do it deliberately."
  type        = string
  default     = "ami-062857f1094ea90ce" # Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20260703
}

data "aws_ami" "gpu" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# ---------------------------------------------------------------------------
# Shared ML-plane bearer token. Generated once and kept in state (sensitive);
# both serving processes enforce it and the control plane attaches it. This is
# ML_AUTH_TOKEN — one of the four contract values.
# ---------------------------------------------------------------------------
resource "random_password" "ml_auth_token" {
  length  = 48
  special = false # base62 keeps it safe in shell env files / Authorization headers
}

# ---------------------------------------------------------------------------
# S3 — durable cartridge store (versioned). Created here unless cart_bucket_name
# points at an existing bucket (e.g. the control plane's storage bucket).
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "carts" {
  count         = local.create_cart_b ? 1 : 0
  bucket        = local.cart_bucket
  force_destroy = false # cart blobs are the durable memory tier — never force-destroyable
}

resource "aws_s3_bucket_versioning" "carts" {
  count  = local.create_cart_b ? 1 : 0
  bucket = aws_s3_bucket.carts[0].id
  versioning_configuration {
    status = "Enabled"
  }
}

# Versioning is a recycle bin, not an eternal archive: a deleted cart set (tens of
# GB) would otherwise keep billing as retained noncurrent versions. Expire them
# after 30 days and drop abandoned multipart parts after 7.
resource "aws_s3_bucket_lifecycle_configuration" "carts" {
  count  = local.create_cart_b ? 1 : 0
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
  count                   = local.create_cart_b ? 1 : 0
  bucket                  = aws_s3_bucket.carts[0].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "carts" {
  count  = local.create_cart_b ? 1 : 0
  bucket = aws_s3_bucket.carts[0].id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ---------------------------------------------------------------------------
# S3 — provisioning bundle scratch. provision.sh uploads the wheel + ml_service
# tarball + bootstrap.sh here; the box pulls them via SSM. Force-destroyable
# (it's transient install material, not data).
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "provision" {
  count         = local.create_prov_b ? 1 : 0
  bucket        = local.provision_b
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "provision" {
  count                   = local.create_prov_b ? 1 : 0
  bucket                  = aws_s3_bucket.provision[0].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------------------------------------------------------------------------
# IAM — instance role: SSM core (management path, no SSH), read the two S3
# buckets, RW the cart bucket. Least privilege scoped to these buckets only.
# ---------------------------------------------------------------------------
data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "gpu" {
  name               = "${var.name}-gpu"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

# SSM is the ONLY access path (send-command for bootstrap, start-session for tunnels).
resource "aws_iam_role_policy_attachment" "gpu_ssm" {
  role       = aws_iam_role.gpu.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

locals {
  cart_bucket_arn = "arn:aws:s3:::${local.cart_bucket}"
  prov_bucket_arn = "arn:aws:s3:::${local.provision_b}"
}

# Cart bucket RW (the onboarding worker writes carts; the serve engine reads them)
# + provisioning bucket read (pull the install bundle).
resource "aws_iam_role_policy" "gpu_s3" {
  name = "cart-store-and-provision"
  role = aws_iam_role.gpu.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "CartStoreRW"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
        Resource = [local.cart_bucket_arn, "${local.cart_bucket_arn}/*"]
      },
      {
        Sid      = "ProvisionBundleRead"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:ListBucket"]
        Resource = [local.prov_bucket_arn, "${local.prov_bucket_arn}/*"]
      },
    ]
  })
}

resource "aws_iam_instance_profile" "gpu" {
  name = "${var.name}-gpu"
  role = aws_iam_role.gpu.name
}

# ---------------------------------------------------------------------------
# Security group. Ports 8001/8002 are NOT public. Ingress sources:
#   * the VPC CIDR (co-located control plane reaches them internally),
#   * an optional operator CIDR (direct in-VPC curl over the SSM tunnel),
#   * an optional peer SG (e.g. the control-plane app SG).
# No port 22 — SSM Session Manager is the shell. Egress is open so the box can
# pull the model from HuggingFace and reach S3.
# ---------------------------------------------------------------------------
resource "aws_security_group" "gpu" {
  name        = "${var.name}-gpu"
  # ASCII only: EC2 rejects non-ASCII GroupDescription (an em-dash here failed the first apply).
  description = "Engram GPU serving unit - internal-only :8001/:8002, SSM-managed"
  vpc_id      = local.vpc_id

  egress {
    description = "all egress (HF model download, S3, SSM endpoints)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.name}-gpu" }
}

# VPC-internal reach to both serving ports (always on — this is how a co-located
# control plane calls the unit).
resource "aws_security_group_rule" "serve_from_vpc" {
  for_each          = toset(["8001", "8002"])
  type              = "ingress"
  security_group_id = aws_security_group.gpu.id
  from_port         = tonumber(each.key)
  to_port           = tonumber(each.key)
  protocol          = "tcp"
  cidr_blocks       = [data.aws_vpc.selected.cidr_block]
  description       = "port ${each.key} from within the VPC (control plane / operator tunnel)"
}

# Optional operator CIDR (e.g. an SSM tunnel origin subnet). Only created when set.
resource "aws_security_group_rule" "serve_from_operator" {
  for_each          = var.allowed_ingress_cidr != "" ? toset(["8001", "8002"]) : toset([])
  type              = "ingress"
  security_group_id = aws_security_group.gpu.id
  from_port         = tonumber(each.key)
  to_port           = tonumber(each.key)
  protocol          = "tcp"
  cidr_blocks       = [var.allowed_ingress_cidr]
  description       = "port ${each.key} from operator CIDR ${var.allowed_ingress_cidr}"
}

# Optional peer SG (e.g. the control-plane app SG when co-located in one VPC).
resource "aws_security_group_rule" "serve_from_peer_sg" {
  for_each                 = var.allowed_ingress_security_group_id != "" ? toset(["8001", "8002"]) : toset([])
  type                     = "ingress"
  security_group_id        = aws_security_group.gpu.id
  from_port                = tonumber(each.key)
  to_port                  = tonumber(each.key)
  protocol                 = "tcp"
  source_security_group_id = var.allowed_ingress_security_group_id
  description              = "port ${each.key} from peer SG ${var.allowed_ingress_security_group_id}"
}

# ---------------------------------------------------------------------------
# The GPU instance. SSM-managed (no key_name, no public IP association changed).
# root volume sized for the ~70GB model + vLLM env. IMDSv2 required; hop limit 2
# so any container/subprocess can still fetch the instance-role creds for S3.
# ---------------------------------------------------------------------------
resource "aws_instance" "gpu" {
  ami                    = var.gpu_ami_id != "" ? var.gpu_ami_id : data.aws_ami.gpu.id
  instance_type          = var.instance_type
  subnet_id              = local.subnet_id
  vpc_security_group_ids = [aws_security_group.gpu.id]
  iam_instance_profile   = aws_iam_instance_profile.gpu.name

  # Spot when var.use_spot (cheaper, interruptible). On-demand by default so the box
  # survives stop/start and the model weights persist on the EBS volume.
  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        spot_instance_type = "one-time"
      }
    }
  }

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
    encrypted   = true
  }

  metadata_options {
    http_tokens                 = "required" # IMDSv2 only
    http_endpoint               = "enabled"
    http_put_response_hop_limit = 2 # a container/subprocess can still reach IMDS for S3 creds
  }

  tags = { Name = "${var.name}-gpu" }

  # Cap the create wait so a scarce-GPU AZ fails FAST (the provider retries
  # InsufficientInstanceCapacity until this timeout — default 10m = ~25 slow retries).
  # 4m is ample for a real launch (a granted instance reaches running in <90s) while
  # letting a capacity hunt rotate AZs quickly instead of burning 10m per dead AZ.
  timeouts {
    create = "4m"
  }

  lifecycle {
    # A new DLAMI release must NOT implicitly replace the box — its EBS holds the
    # downloaded model + HF cache. Bump var.gpu_ami_id deliberately to roll it.
    ignore_changes = [ami]
  }
}
