# ===========================================================================
# Shared handles the env stacks consume via terraform_remote_state, PLUS the two
# manual Cloudflare handoffs the operator does by hand (cert-validation CNAMEs and
# the four host CNAMEs -> ALB dns_name) — same pattern as the SES DKIM handoff.
# ===========================================================================

# --- consumed by envs/* via terraform_remote_state -------------------------
output "cluster_arn" {
  description = "ECS cluster the env services run in."
  value       = aws_ecs_cluster.main.arn
}

output "cluster_name" {
  description = "ECS cluster name (deploy.sh update-service target)."
  value       = aws_ecs_cluster.main.name
}

output "vpc_id" {
  description = "VPC the platform + serving unit share."
  value       = local.vpc_id
}

output "subnet_ids" {
  description = "Subnets for the ALB + Fargate tasks."
  value       = local.subnet_ids
}

output "alb_arn" {
  description = "Public ALB ARN."
  value       = aws_lb.main.arn
}

output "alb_dns_name" {
  description = "Public ALB DNS name — the CNAME target for all four host records."
  value       = aws_lb.main.dns_name
}

output "alb_zone_id" {
  description = "ALB hosted-zone id (for an ALIAS record if DNS ever moves to Route 53)."
  value       = aws_lb.main.zone_id
}

output "alb_security_group_id" {
  description = "ALB security group id — each env allows task ingress from this SG."
  value       = aws_security_group.alb.id
}

output "https_listener_arn" {
  description = "The :443 listener ARN — env stacks attach host-header rules to it."
  value       = aws_lb_listener.https.arn
}

output "acm_certificate_arn" {
  description = "ACM cert ARN covering all four hosts."
  value       = aws_acm_certificate.platform.arn
}

output "ecr_backend_repo_url" {
  description = "Backend ECR repository URL (build_push.sh / deploy.sh)."
  value       = aws_ecr_repository.this["backend"].repository_url
}

output "ecr_frontend_repo_url" {
  description = "Frontend ECR repository URL (build_push.sh / deploy.sh)."
  value       = aws_ecr_repository.this["frontend"].repository_url
}

# --- MANUAL Cloudflare handoff 1/2: certificate validation -----------------
# Add each of these as a CNAME in Cloudflare (DNS-only, grey cloud). Once all are
# present AWS moves the cert to ISSUED (minutes) and the HTTPS listener goes live.
# Print with: terraform output -json acm_validation_cname_records
output "acm_validation_cname_records" {
  description = "Cloudflare CNAMEs to add for ACM cert validation (name -> value), DNS-only. One per host; AWS de-dups identical records."
  value = {
    for dvo in aws_acm_certificate.platform.domain_validation_options :
    dvo.domain_name => {
      cname_name  = dvo.resource_record_name
      cname_value = dvo.resource_record_value
      type        = dvo.resource_record_type
    }
  }
}

# --- MANUAL Cloudflare handoff 2/2: the four host records ------------------
# Point each product host at the ALB. In Cloudflare these can be proxied (orange
# cloud) OR DNS-only; DNS-only is simplest and avoids Cloudflare's own TLS in front
# of the ALB's. Print with: terraform output -json host_cname_records
output "host_cname_records" {
  description = "Cloudflare CNAMEs: each public host -> the ALB DNS name."
  value = {
    for h in local.cert_hosts : h => aws_lb.main.dns_name
  }
}
