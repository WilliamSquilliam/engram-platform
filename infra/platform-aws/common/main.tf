# ===========================================================================
# SHARED platform infrastructure — applied ONCE, consumed by both env stacks
# (envs/uat + envs/prod) via terraform_remote_state. Holds the pieces that must
# be singletons across UAT and prod:
#   * one ECS Fargate cluster (both envs' services run in it)
#   * two ECR repos (backend, frontend) with untagged-expiry lifecycle policies
#   * one PUBLIC ALB (HTTP->HTTPS redirect) + one ACM cert covering all 4 hosts
#   * the ALB security group (public :80/:443 in, VPC-wide out to the tasks)
# The per-env pieces (services, target groups, listener rules, RDS, S3, secrets,
# task IAM) live in modules/platform-env, instantiated by each env root.
# ===========================================================================

# --- placement: the SAME default VPC the serving unit uses ------------------
data "aws_vpc" "default" {
  count   = var.vpc_id == "" ? 1 : 0
  default = true
}

locals {
  vpc_id = var.vpc_id != "" ? var.vpc_id : data.aws_vpc.default[0].id
}

data "aws_subnets" "in_vpc" {
  count = length(var.subnet_ids) == 0 ? 1 : 0
  filter {
    name   = "vpc-id"
    values = [local.vpc_id]
  }
}

locals {
  subnet_ids = length(var.subnet_ids) > 0 ? var.subnet_ids : data.aws_subnets.in_vpc[0].ids
  cert_hosts = [var.prod_app_host, var.prod_api_host, var.uat_app_host, var.uat_api_host]
}

# ---------------------------------------------------------------------------
# ECS cluster — one for both envs. Services are namespaced <env>-engram-* so UAT
# and prod tasks never collide inside it.
# ---------------------------------------------------------------------------
resource "aws_ecs_cluster" "main" {
  name = "${var.name}-cluster"
}

# ---------------------------------------------------------------------------
# ECR — one repo per service, shared by both envs (backend image is env-agnostic;
# frontend is built per-env and tagged <sha>-<env> in the same repo). Untagged
# revisions expire after 7 days so orphaned layers from re-tags don't accumulate.
# ---------------------------------------------------------------------------
resource "aws_ecr_repository" "this" {
  for_each             = toset(["backend", "frontend"])
  name                 = "${var.name}/${each.value}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true # so `terraform destroy` removes images too
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "expire_untagged" {
  for_each   = aws_ecr_repository.this
  repository = each.value.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Expire untagged image revisions after 7 days"
      selection = {
        tagStatus   = "untagged"
        countType   = "sinceImagePushed"
        countUnit   = "days"
        countNumber = 7
      }
      action = { type = "expire" }
    }]
  })
}

# ---------------------------------------------------------------------------
# ACM certificate — one cert, four SANs (all public hosts). DNS-validated: the
# operator adds the validation CNAMEs in Cloudflare by hand (same manual handoff
# as the SES DKIM records). The validation records are surfaced in outputs.tf.
# NOTE: this stack does NOT create an aws_acm_certificate_validation resource, so
# `terraform apply` completes without blocking on validation — the cert sits in
# PENDING_VALIDATION until the operator adds the CNAMEs, then AWS issues it. The
# HTTPS listener references the cert ARN regardless (it goes live once ISSUED).
# ---------------------------------------------------------------------------
resource "aws_acm_certificate" "platform" {
  domain_name               = local.cert_hosts[0]
  subject_alternative_names = slice(local.cert_hosts, 1, length(local.cert_hosts))
  validation_method         = "DNS"

  lifecycle {
    create_before_destroy = true
  }
}

# ---------------------------------------------------------------------------
# Public ALB. Internet-facing (the platform is a public SaaS, unlike the serving
# stack's internal ALB). :80 redirects to :443; :443 terminates TLS with the cert
# and, until an env's listener rules match, returns a 404 fixed-response so a bare
# hit to the ALB DNS never leaks another env's app.
# ---------------------------------------------------------------------------
resource "aws_security_group" "alb" {
  name        = "${var.name}-alb"
  description = "Public ALB for the Engram platform (uat + prod)"
  vpc_id      = local.vpc_id

  ingress {
    description = "HTTP (redirected to HTTPS)"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    description = "to Fargate tasks within the VPC"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = { Name = "${var.name}-alb" }
}

resource "aws_lb" "main" {
  name               = "${var.name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = local.subnet_ids
  idle_timeout       = var.alb_idle_timeout
}

# :80 -> :443 permanent redirect. No plaintext app traffic.
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"
  default_action {
    type = "redirect"
    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

# :443 terminates TLS. Default action is a 404 — each env stack ADDS host-header
# listener rules (via aws_lb_listener_rule referencing this listener's ARN) that
# route {uat-}api. -> its backend and {uat-}app. -> its frontend.
resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.main.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = aws_acm_certificate.platform.arn

  default_action {
    type = "fixed-response"
    fixed_response {
      content_type = "text/plain"
      message_body = "No matching host. Use one of the app/api hostnames."
      status_code  = "404"
    }
  }
}
