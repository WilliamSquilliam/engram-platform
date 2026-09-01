# ===========================================================================
# Per-env RDS Postgres 16 — the control-plane metadata DB (tenants, users, corpora,
# jobs). Single-AZ db.t4g.micro by default (gp3, encrypted), private, reachable only
# from this env's app SG. The DATABASE_URL is composed in secrets.tf and injected as
# a task secret. A random suffix on the final snapshot id avoids collisions across
# destroy/recreate cycles.
# ===========================================================================
resource "random_id" "snap" {
  byte_length = 4
}

resource "aws_db_subnet_group" "main" {
  name       = "${local.name_prefix}-db"
  subnet_ids = var.subnet_ids
}

resource "aws_db_instance" "main" {
  identifier     = "${local.name_prefix}-pg"
  engine         = "postgres"
  engine_version = "16"
  instance_class = var.db_instance_class

  allocated_storage = var.db_allocated_gb
  storage_type      = "gp3"
  storage_encrypted = true

  db_name  = "engram"
  username = "engram"
  password = random_password.db.result

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = false

  multi_az            = false
  deletion_protection = false # kept off so `destroy` stays one-command; the final snapshot below makes it survivable

  # 7-day PITR + a final snapshot on destroy: a `terraform destroy` can't
  # irreversibly delete the tenant/corpus metadata.
  skip_final_snapshot       = false
  final_snapshot_identifier = "${local.name_prefix}-pg-final-${random_id.snap.hex}"
  backup_retention_period   = 7
  apply_immediately         = true
}
