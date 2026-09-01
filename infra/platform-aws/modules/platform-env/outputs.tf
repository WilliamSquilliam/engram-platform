# Per-env handles — surfaced by the env root for deploy.sh and operator verification.
output "backend_service_name" {
  description = "ECS service name for the backend (deploy.sh update target)."
  value       = aws_ecs_service.backend.name
}

output "frontend_service_name" {
  description = "ECS service name for the frontend (deploy.sh update target)."
  value       = aws_ecs_service.frontend.name
}

output "backend_task_family" {
  description = "Backend task-definition family (deploy.sh registers new revisions here)."
  value       = aws_ecs_task_definition.backend.family
}

output "frontend_task_family" {
  description = "Frontend task-definition family."
  value       = aws_ecs_task_definition.frontend.family
}

output "storage_bucket" {
  description = "This env's S3 document bucket (PLATFORM_S3_BUCKET)."
  value       = aws_s3_bucket.storage.bucket
}

output "db_endpoint" {
  description = "RDS endpoint (host) for this env."
  value       = aws_db_instance.main.address
}

output "app_url" {
  description = "Public frontend URL for this env."
  value       = local.app_url
}

output "api_url" {
  description = "Public backend-API URL for this env."
  value       = local.api_url
}

output "serving_ml_service_url" {
  description = "Resolved ML_SERVICE_URL fed to the backend (from serving remote state or override)."
  value       = local.ml_service_url
}

output "serving_inference_service_url" {
  description = "Resolved INFERENCE_SERVICE_URL fed to the backend."
  value       = local.inference_service_url
}

output "cartridge_store_bucket" {
  description = "Resolved CARTRIDGE_STORE_BUCKET the control plane references."
  value       = local.cartridge_bucket
}
