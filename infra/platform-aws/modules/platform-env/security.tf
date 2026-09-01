# ===========================================================================
# Per-env security groups. The boundary is INGRESS: the ALB is the only thing that
# reaches the tasks, and only the tasks reach RDS. Egress is open so tasks pull
# images (ECR), read/write S3, hit RDS, and call the GPU serving box on the VPC.
# ===========================================================================

# --- Fargate tasks (backend + frontend) ---
resource "aws_security_group" "app" {
  name        = "${local.name_prefix}-app"
  description = "${var.env} Engram Fargate tasks"
  vpc_id      = var.vpc_id
  egress {
    description = "all egress (ECR, S3, RDS, SES, GPU serving box on the VPC)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = { Name = "${local.name_prefix}-app" }
}

# Backend :8000 from the shared ALB only.
resource "aws_security_group_rule" "app_from_alb_backend" {
  type                     = "ingress"
  security_group_id        = aws_security_group.app.id
  from_port                = 8000
  to_port                  = 8000
  protocol                 = "tcp"
  source_security_group_id = var.alb_security_group_id
  description              = "backend API from the public ALB"
}

# Frontend :3000 from the shared ALB only.
resource "aws_security_group_rule" "app_from_alb_frontend" {
  type                     = "ingress"
  security_group_id        = aws_security_group.app.id
  from_port                = 3000
  to_port                  = 3000
  protocol                 = "tcp"
  source_security_group_id = var.alb_security_group_id
  description              = "frontend from the public ALB"
}

# --- RDS Postgres — reachable only from this env's app SG ---
resource "aws_security_group" "rds" {
  name        = "${local.name_prefix}-rds"
  description = "${var.env} Engram RDS Postgres"
  vpc_id      = var.vpc_id
  tags        = { Name = "${local.name_prefix}-rds" }
}

resource "aws_security_group_rule" "rds_from_app" {
  type                     = "ingress"
  security_group_id        = aws_security_group.rds.id
  from_port                = 5432
  to_port                  = 5432
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.app.id
  description              = "Postgres from ${var.env} app tasks only"
}
