# ---------------------------------------------------------------------------
# Inputs for the swappable GPU serving unit. A model/hardware PIVOT is a change
# here (or a sibling module emitting the same four outputs) — never a change to
# product code. See README.md "How to pivot".
# ---------------------------------------------------------------------------

variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "Local AWS CLI profile. The unit runs in AWS account 808379776072 (Engram Dynamics)."
  type        = string
  default     = "Engram-Dynamics"
}

variable "name" {
  description = "Name prefix for this serving unit's resources."
  type        = string
  default     = "engram-serving"
}

# ---------------------------------------------------------------------------
# Model + GPU. The default stands up Llama-3.3-70B-Instruct-FP8-dynamic on a
# g6e.12xlarge (4x L40S 48GB) with tensor-parallel 4. The FP8 checkpoint is ~70GB;
# TP=4 shards it across the four cards with room for the KV cache. Pivot to another
# model by changing model_ref + tensor_parallel + instance_type together (a bigger
# model needs a bigger box; see tiers in the sibling repo for the pattern).
# ---------------------------------------------------------------------------
variable "instance_type" {
  description = "GPU EC2 instance type. Default g6e.12xlarge = 4x L40S 48GB, 48 vCPU (on-demand ~$10.49/hr)."
  type        = string
  default     = "g6e.12xlarge"
}

variable "model_ref" {
  description = "HuggingFace weights id served by vLLM and stamped into carts (cartridges.model_binding). Default = Llama-3.3-70B FP8-dynamic."
  type        = string
  default     = "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"
}

variable "tensor_parallel" {
  description = "vLLM tensor-parallel size (GPUs the serve engine shards the model across). 4 for the 70B-FP8 across 4x L40S."
  type        = number
  default     = 4
}

variable "model_precision" {
  description = "Informational precision label surfaced in the model registry tier (fp8 | bf16 | w4a4)."
  type        = string
  default     = "fp8"
}

variable "context_tokens" {
  description = "Max context window advertised for the tier + used as vLLM max_model_len. 131072 for Llama-3.3-70B."
  type        = number
  default     = 131072
}

variable "root_volume_gb" {
  description = "Root gp3 EBS size (GB). Must hold the ~70GB model weights + the vLLM env + HF cache. >=300 for the 70B-FP8 tier."
  type        = number
  default     = 300

  validation {
    condition     = var.root_volume_gb >= 300
    error_message = "root_volume_gb must be >= 300 to hold the ~70GB model + vLLM env + HF cache."
  }
}

# ---------------------------------------------------------------------------
# Placement. By default the unit runs in the account's DEFAULT VPC (a public
# subnet, no public IP — reached via SSM), so it stands alone with no dependency
# on the control-plane VPC. To co-locate it with the control plane, set
# vpc_id + subnet_id to that VPC's private subnet and add the app SG as an
# ingress source (allowed_ingress_security_group_id).
# ---------------------------------------------------------------------------
variable "vpc_id" {
  description = "VPC to place the instance in. Empty = the region's default VPC."
  type        = string
  default     = ""
}

variable "subnet_id" {
  description = "Subnet for the GPU instance. Empty = the first subnet in the chosen VPC. A private subnet needs NAT for the model download; a default-VPC public subnet works without a public IP via SSM's VPC-less egress."
  type        = string
  default     = ""
}

# ---------------------------------------------------------------------------
# Access. Ports 8001/8002 are NEVER public. Ingress is restricted to (a) the VPC
# CIDR (so a co-located control plane reaches them internally), (b) an optional
# operator CIDR for direct in-VPC curl over the SSM tunnel, and (c) an optional
# peer security group (e.g. the control-plane app SG). SSM is the management path —
# no SSH keypair, no port 22 ingress.
# ---------------------------------------------------------------------------
variable "allowed_ingress_cidr" {
  description = "Extra CIDR allowed to reach :8001/:8002 (operator subnet / SSM tunnel origin). Empty = VPC-internal only. NEVER set to 0.0.0.0/0 — these ports have no public exposure by design."
  type        = string
  default     = ""

  validation {
    condition     = var.allowed_ingress_cidr != "0.0.0.0/0"
    error_message = "allowed_ingress_cidr must not be 0.0.0.0/0 — the serving ports are internal-only by design."
  }
}

variable "allowed_ingress_security_group_id" {
  description = "Peer security group allowed to reach :8001/:8002 (e.g. the control-plane app SG when co-located). Empty = none."
  type        = string
  default     = ""
}

# ---------------------------------------------------------------------------
# Cost / capacity.
# ---------------------------------------------------------------------------
variable "use_spot" {
  description = "Launch as a Spot instance (cheaper, interruptible). Default false = on-demand (weights persist on EBS across stop/start; a Spot reclaim terminates the box)."
  type        = bool
  default     = false
}

variable "hourly_cost_usd" {
  description = "The box's on-demand $/hr, surfaced in the env block so the control plane's measured $/query uses the real running price. g6e.12xlarge ~= 10.49."
  type        = string
  default     = "10.49"
}

# ---------------------------------------------------------------------------
# Cartridge store bucket. The onboarding worker (:8001) WRITES CAG carts here and
# the vLLM serve engine (:8002) READS them by id — the S3CartridgeStore durable
# tier. Empty = create a dedicated bucket engram-carts-<acct>-<region>; set to an
# existing bucket name to reuse the control plane's storage bucket instead.
# ---------------------------------------------------------------------------
variable "cart_bucket_name" {
  description = "S3 bucket for the durable cartridge store. Empty = create engram-carts-<acct>-<region> (versioned)."
  type        = string
  default     = ""
}

variable "cart_store_prefix" {
  description = "Key prefix under the cart bucket for cart blobs."
  type        = string
  default     = "cartridges"
}

variable "provision_bucket_name" {
  description = "S3 bucket provision.sh uploads the wheel + ml_service bundle to (SSM then runs bootstrap from it). Empty = create engram-serving-provision-<acct>-<region> (force-destroyable scratch)."
  type        = string
  default     = ""
}
