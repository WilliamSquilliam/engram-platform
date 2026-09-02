# ===========================================================================
# Per-env Fargate services. Backend (:8000) + frontend (:3000), each with a
# CloudWatch log group and a container health check. The backend env block wires
# the full production config that config.validate() demands, plus the four
# serving-unit contract values from remote state.
# ===========================================================================

resource "aws_cloudwatch_log_group" "svc" {
  for_each          = toset(["backend", "frontend"])
  name              = "/ecs/${local.name_prefix}/${each.value}"
  retention_in_days = var.log_retention_days
}

locals {
  app_url = "https://${var.app_host}"
  api_url = "https://${var.api_host}"

  # Full backend env. ENV=production makes config.validate() enforce the strong
  # secrets / Postgres / explicit-CORS / non-'none' email fail-fasts; the secrets
  # themselves arrive via `secrets` (Secrets Manager), not here.
  backend_env = [
    { name = "ENV", value = "production" },
    { name = "LOG_LEVEL", value = "INFO" },
    # Host-header allowlist (TrustedHostMiddleware engages only when set and not '*').
    # localhost MUST stay: the container healthcheck hits http://localhost:8000/health —
    # without it every task fails health and cycles. (2026-09 security sweep H1.)
    { name = "ALLOWED_HOSTS", value = "${var.api_host},localhost" },

    # data stores
    { name = "PLATFORM_STORAGE_BACKEND", value = "s3" },
    { name = "PLATFORM_S3_BUCKET", value = aws_s3_bucket.storage.bucket },
    { name = "PLATFORM_DATA_DIR", value = "/data" },
    { name = "PLATFORM_STORAGE_DIR", value = "/data/storage" },

    # public URLs — CORS is explicit (no localhost, no '*') so the prod validator passes.
    { name = "CORS_ORIGINS", value = local.app_url },
    { name = "FRONTEND_URL", value = local.app_url },
    # The worker->control-plane progress callback + Google redirect target resolve to
    # the public API host (the ALB fronts it).
    { name = "BACKEND_INTERNAL_URL", value = local.api_url },
    { name = "GOOGLE_REDIRECT_URI", value = "${local.api_url}/auth/google/callback" },

    # auth / registration posture
    { name = "AUTH_BACKEND", value = "local" },
    { name = "ALLOW_REGISTRATION", value = "false" },
    { name = "PLATFORM_ADMIN_EMAIL", value = var.platform_admin_email },
    { name = "BOOTSTRAP_ADMIN_EMAIL", value = var.bootstrap_admin_email },

    # transactional email via SES (EMAIL_BACKEND must not be 'none' in prod).
    { name = "EMAIL_BACKEND", value = "ses" },
    { name = "EMAIL_FROM", value = var.email_from },
    { name = "AWS_REGION", value = var.region },

    # Google sign-in (optional; id is public, secret is a task secret when set).
    { name = "GOOGLE_CLIENT_ID", value = var.google_client_id },

    # --- serving unit: the four contract values (from remote state or overrides) ---
    { name = "INFERENCE_BACKEND", value = "vllm" },
    # Top-3 retrieval restored: multi-cart serving is positionally correct as of wheel
    # 0.6.1 (exact-token placement + per-cart RoPE rebase in the model's rotary
    # convention — see the 2026-09-02 decisions-log entries). Requires the serving box
    # to run engram-cartridge >= 0.6.1 with CARTRIDGE_ROPE_CONVENTION set for the model.
    { name = "INFERENCE_TOPK", value = "3" },
    { name = "ML_SERVICE_URL", value = local.ml_service_url },
    { name = "INFERENCE_SERVICE_URL", value = local.inference_service_url },
    { name = "MODEL_REGISTRY_JSON", value = local.model_registry_json },
    # the durable cart store the GPU box reads/writes; the control plane references it
    # for retrieval hydration.
    { name = "CARTRIDGE_STORE_BACKEND", value = "s3" },
    { name = "CARTRIDGE_STORE_BUCKET", value = local.cartridge_bucket },
    { name = "CARTRIDGE_STORE_PREFIX", value = local.cartridge_prefix },
  ]

  # Secrets injected from Secrets Manager (the four config.validate() secrets + the
  # composed DB URL, and the Google secret when present).
  backend_secrets = [
    for k in keys(local.secret_values) :
    { name = k, valueFrom = aws_secretsmanager_secret.app[k].arn }
  ]
}

# --- backend task + service ---
resource "aws_ecs_task_definition" "backend" {
  family                   = "${local.name_prefix}-backend"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.backend_cpu
  memory                   = var.backend_memory
  execution_role_arn       = aws_iam_role.ecs_exec.arn
  task_role_arn            = aws_iam_role.ecs_task.arn
  container_definitions = jsonencode([{
    name         = "backend"
    image        = var.backend_image
    essential    = true
    portMappings = [{ containerPort = 8000 }]
    environment  = local.backend_env
    secrets      = local.backend_secrets
    healthCheck = {
      command     = ["CMD-SHELL", "python -c \"import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status==200 else 1)\""]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 90 # alembic upgrade + fastembed cache warm on first boot
    }
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.svc["backend"].name
        "awslogs-region"        = var.region
        "awslogs-stream-prefix" = "backend"
      }
    }
  }])
}

resource "aws_ecs_service" "backend" {
  name            = "${local.name_prefix}-backend"
  cluster         = var.cluster_arn
  task_definition = aws_ecs_task_definition.backend.arn
  desired_count   = 1
  launch_type     = "FARGATE"
  # Public subnets + a public IP so the task can reach ECR / SES / S3 without a NAT
  # gateway (there is none in the default VPC). Ingress is still locked to the ALB SG.
  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [aws_security_group.app.id]
    assign_public_ip = true
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.backend.arn
    container_name   = "backend"
    container_port   = 8000
  }
  # Fargate deployments roll a task at a time; keep steady state during a deploy.
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200
  depends_on                         = [aws_lb_listener_rule.backend]
}

# --- frontend task + service ---
resource "aws_ecs_task_definition" "frontend" {
  family                   = "${local.name_prefix}-frontend"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.frontend_cpu
  memory                   = var.frontend_memory
  execution_role_arn       = aws_iam_role.ecs_exec.arn
  task_role_arn            = aws_iam_role.ecs_task.arn
  container_definitions = jsonencode([{
    name         = "frontend"
    image        = var.frontend_image
    essential    = true
    portMappings = [{ containerPort = 3000 }]
    healthCheck = {
      command     = ["CMD-SHELL", "node -e \"require('http').get('http://localhost:3000/login',r=>process.exit(r.statusCode===200?0:1)).on('error',()=>process.exit(1))\""]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 30
    }
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.svc["frontend"].name
        "awslogs-region"        = var.region
        "awslogs-stream-prefix" = "frontend"
      }
    }
  }])
}

resource "aws_ecs_service" "frontend" {
  name            = "${local.name_prefix}-frontend"
  cluster         = var.cluster_arn
  task_definition = aws_ecs_task_definition.frontend.arn
  desired_count   = 1
  launch_type     = "FARGATE"
  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [aws_security_group.app.id]
    assign_public_ip = true
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.frontend.arn
    container_name   = "frontend"
    container_port   = 3000
  }
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200
  depends_on                         = [aws_lb_listener_rule.frontend]
}
