output "backend_service_name" {
  description = "ECS backend service name (deploy.sh target)."
  value       = module.env.backend_service_name
}
output "frontend_service_name" {
  description = "ECS frontend service name (deploy.sh target)."
  value       = module.env.frontend_service_name
}
output "backend_task_family" {
  value = module.env.backend_task_family
}
output "frontend_task_family" {
  value = module.env.frontend_task_family
}
output "cluster_name" {
  description = "ECS cluster name (from common; deploy.sh needs it)."
  value       = data.terraform_remote_state.common.outputs.cluster_name
}
output "app_url" {
  value = module.env.app_url
}
output "api_url" {
  value = module.env.api_url
}
output "storage_bucket" {
  value = module.env.storage_bucket
}
output "db_endpoint" {
  value = module.env.db_endpoint
}
