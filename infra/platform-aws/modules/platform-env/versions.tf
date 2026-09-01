# Provider requirements for the module. No backend block here — the ROOT (envs/<env>)
# owns the backend. random generates the per-env secrets (JWT/SESSION/INTERNAL/DB).
terraform {
  required_version = ">= 1.7.0"
  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.60" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }
}
