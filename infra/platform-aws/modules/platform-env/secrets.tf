# ===========================================================================
# Per-env secrets — generated here and stored in Secrets Manager, injected into the
# backend task as `secrets` (never plaintext env, never committed). These satisfy the
# prod config.validate() fail-fasts: JWT_SECRET / SESSION_SECRET / INTERNAL_API_TOKEN
# / ML_AUTH_TOKEN are all strong (>=32). recovery_window=0 so destroy/recreate doesn't
# leave secrets "scheduled for deletion".
# ===========================================================================

# Alphanumeric only: safe inside a DATABASE_URL, an HTTP header, and RDS's password
# rules without URL-encoding surprises.
resource "random_password" "db" {
  length  = 32
  special = false
}
resource "random_password" "jwt" {
  length  = 48
  special = false
}
resource "random_password" "session" {
  length  = 48
  special = false
}
resource "random_password" "internal" {
  length  = 48
  special = false
}
# The seeded operator login (registration is off in prod/UAT — without this nobody
# can sign in). Retrieve with:
#   aws secretsmanager get-secret-value --secret-id <env>-engram/BOOTSTRAP_ADMIN_PASSWORD \
#     --query SecretString --output text --profile Engram-Dynamics
resource "random_password" "bootstrap_admin" {
  length  = 20
  special = false
}

locals {
  database_url = "postgresql+psycopg://engram:${random_password.db.result}@${aws_db_instance.main.address}:5432/engram"

  # The four secrets that MUST be injected (all read by config.validate() in prod).
  # ML_AUTH_TOKEN comes from the serving unit (both planes share one bearer); the
  # rest are generated per env. Google secret is added conditionally below.
  base_secret_values = {
    DATABASE_URL             = local.database_url
    JWT_SECRET               = random_password.jwt.result
    SESSION_SECRET           = random_password.session.result
    INTERNAL_API_TOKEN       = random_password.internal.result
    ML_AUTH_TOKEN            = local.ml_auth_token
    BOOTSTRAP_ADMIN_PASSWORD = random_password.bootstrap_admin.result
  }

  google_secret_values = var.google_client_secret != "" ? {
    GOOGLE_CLIENT_SECRET = var.google_client_secret
  } : {}

  # GPU control plane (platform-admin start/stop of the Lambda box). Only present
  # when set — absence keeps the feature off (backend gates on LAMBDA_API_KEY).
  # ZONE_ID rides the secrets map for one delivery path, though it's not sensitive.
  gpu_secret_values = merge(
    var.lambda_api_key != "" ? { LAMBDA_API_KEY = var.lambda_api_key } : {},
    var.cloudflare_api_token != "" ? { CLOUDFLARE_API_TOKEN = var.cloudflare_api_token } : {},
    var.cloudflare_zone_id != "" ? { CLOUDFLARE_ZONE_ID = var.cloudflare_zone_id } : {},
  )

  secret_values = merge(local.base_secret_values, local.google_secret_values, local.gpu_secret_values)
}

resource "aws_secretsmanager_secret" "app" {
  for_each                = local.secret_values
  name                    = "${local.name_prefix}/${each.key}"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "app" {
  for_each      = local.secret_values
  secret_id     = aws_secretsmanager_secret.app[each.key].id
  secret_string = each.value
}
