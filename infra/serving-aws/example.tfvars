# Copy to a real *.tfvars (gitignored) and adjust. All values shown are the
# defaults for the Llama-3.3-70B-FP8 tier — an empty file applies the same thing.

# --- model + GPU (change these together to pivot to another model) ---
instance_type   = "g6e.12xlarge" # 4x L40S 48GB
model_ref       = "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"
tensor_parallel = 4
model_precision = "fp8"
context_tokens  = 131072
root_volume_gb  = 300

# --- cost ---
use_spot        = false   # true = cheaper, interruptible (a reclaim terminates the box)
hourly_cost_usd = "10.49" # surfaced to the control plane for measured $/query

# --- access (ports 8001/8002 are internal-only; never 0.0.0.0/0) ---
# allowed_ingress_cidr              = "10.20.0.0/16"   # operator/tunnel subnet
# allowed_ingress_security_group_id = "sg-xxxx"        # control-plane app SG when co-located

# --- placement (default VPC unless set) ---
# vpc_id    = "vpc-xxxx"
# subnet_id = "subnet-xxxx"

# --- buckets (empty = auto-named; set to reuse the control plane's storage bucket) ---
# cart_bucket_name = "cartridge-storage-808379776072-us-east-1"
