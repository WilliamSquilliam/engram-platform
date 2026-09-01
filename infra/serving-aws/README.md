# AWS GPU serving unit

One GPU box that serves **Llama-3.3-70B-Instruct-FP8-dynamic** through the cartridge
stack: onboarding + resident-KV vLLM serving, behind the platform's swappable-unit
contract. Apply the Terraform, provision the box over SSM, smoke-test it, paste four
env values into the control plane, done.

## The swappable-unit contract

The control plane (`backend/app/config.py` + `backend/app/serving.py`) consumes exactly
**four** values. This module emits them as Terraform outputs. Any serving stack — this
one, a neocloud box, an H100 fleet — that produces these four is a drop-in replacement.

| Contract value | Output | What it is |
| --- | --- | --- |
| `ML_SERVICE_URL` | `ml_service_url` | Onboarding / train worker, `:8001` (`ml_service/app.py`) |
| `INFERENCE_SERVICE_URL` | `inference_service_url` | vLLM resident-KV serve, `:8002` (`ml_service/vllm_inference.py`) |
| `ML_AUTH_TOKEN` | `ml_auth_token` (sensitive) | Shared bearer both processes enforce |
| `MODEL_REGISTRY_JSON` | `model_registry_json` | Product tiers → `model_ref`/precision/context |

`terraform output -raw serving_unit_env` prints a ready-to-paste env block with all four
(the token line is a placeholder — fill it from `terraform output -raw ml_auth_token` so a
plain `terraform output` never leaks it).

Model and instance are **input variables**, so a pivot is a variable change, never a
product-code change. The control plane never learns the GPU type or cloud.

## What gets stood up

- One EC2 GPU instance (default `g6e.12xlarge` = 4× L40S 48GB), pinned DLAMI
  `ami-062857f1094ea90ce` (NVIDIA driver + CUDA preinstalled), 300GB gp3 root (holds the
  ~70GB model + the vLLM env + HF cache).
- IAM instance profile: SSM core (the only access path — no SSH keypair) + RW on the cart
  bucket + read on the provisioning bucket.
- S3 cart bucket `engram-carts-<acct>-us-east-1` (versioned, encrypted, private) — the
  durable cartridge store the onboarding worker writes and the serve engine reads.
- Security group: `:8001`/`:8002` reachable only from inside the VPC (+ an optional
  operator CIDR / peer SG). Never public. Egress open for the HF model pull.
- Two systemd units on the box: `engram-onboard` (`:8001`) and `engram-serve` (`:8002`),
  enabled + `Restart=on-failure`.

## Run it

```bash
cd infra/serving-aws

# 1. APPLY — creates the box, buckets, IAM, SG, and generates the ML_AUTH_TOKEN.
terraform init
terraform apply     # add -var use_spot=true for a cheaper interruptible box

# 2. PROVISION — build the wheel, bundle ml_service + bootstrap, run it over SSM.
#    Needs the sibling repo (../Engram-Smart-CAG) and `pip install build`.
bash provision.sh

# 3. SMOKE — wait for engine_ready, run compat_check, onboard one doc, query it,
#    assert a grounded answer + the metrics shape. Prints PASS/FAIL per step.
bash smoke.sh

# 4. WIRE — paste the four contract values into the control plane env.
terraform output -raw serving_unit_env      # everything except the token
terraform output -raw ml_auth_token         # the bearer (sensitive)
```

The first serve start downloads the ~70GB model to the HF cache on the box's EBS
(minutes). `:8002/health` answers immediately and flips `engine_ready` to `true` once the
engine is warm; `smoke.sh` waits for that.

## Stop / start for idle savings

The box is on-demand and the model weights persist on the EBS root volume, so stop it when
idle and start it later — no re-download, and the private IP (hence the two service URLs)
is stable across the cycle.

### Disk layout (two tiers, deliberately split)

| Volume | Size | Persistence | Holds |
|---|---|---|---|
| EBS root (gp3, `root_volume_gb`) | 300 GB default | survives stop/start | OS + venv + **HF model weights** (~70 GB) |
| NVMe instance store | ~3.8 TB on g6e.12xlarge (2×1900, RAID0) | **wiped on stop** | cart hot-mirror + registry scratch |

`engram-nvme.service` (installed by bootstrap) re-formats and mounts the instance store at
`/opt/engram/nvme` on **every boot** — after a stop/start the mirror comes back empty and
re-warms from S3 on demand, which is exactly the loss-tolerant tier the cart store is
designed for. On an instance type with no instance store the path stays a plain EBS dir
and everything still works, just with less hot-cache capacity.

```bash
ID=$(terraform output -raw instance_id)
aws ec2 stop-instances  --instance-ids "$ID" --profile Engram-Dynamics   # stop paying compute
aws ec2 start-instances --instance-ids "$ID" --profile Engram-Dynamics   # resume; systemd units auto-start
```

You keep paying for the EBS volume while stopped (small next to the GPU-hour), and the
S3 cart store, either way.

## Cost

- `g6e.12xlarge` on-demand ≈ **$10.49/hr** (4× L40S). Spot (`-var use_spot=true`) is
  cheaper but a reclaim terminates the box — use it only for throwaway runs, not the
  durable serving box whose EBS you want to keep.
- EBS: 300GB gp3 ≈ $24/mo (billed whether the box runs or is stopped).
- S3 cart store: pennies per GB-month; versioned copies expire after 30 days.

The box's real hourly cost is surfaced in `serving_unit_env` (var `hourly_cost_usd`) so the
control plane's measured $/query uses the actual running price, not a documentation constant.

## How to pivot

- **Different model / precision / GPU:** change `model_ref`, `model_precision`,
  `context_tokens`, `tensor_parallel`, and `instance_type` together (a bigger model needs a
  bigger box). Re-apply, re-provision. Nothing in product code changes — the control plane
  reads the new `model_registry_json` and calls the same two URLs.
- **Different cloud / a sibling unit:** implement another module that emits the same four
  outputs (`ml_service_url`, `inference_service_url`, `ml_auth_token`, `model_registry_json`).
  Point the control plane at its env block. The contract is the seam; the hardware behind it
  is free to change.

## Notes

- State lives in the shared S3 backend (`cartridge-tfstate-808379776072-us-east-1`) under its
  **own** key `serving-aws/terraform.tfstate`, so this unit applies/destroys independently of
  the control-plane stack. It holds the generated `ML_AUTH_TOKEN` — hence encrypted + versioned.
- An account guard refuses to apply anywhere but Engram Dynamics (808379776072).
- The pinned DLAMI was verified present on 2026-09-01. If AWS ever retires it, set
  `-var gpu_ami_id=""` to fall back to the most-recent DLAMI lookup (this **replaces** the
  box and wipes its EBS/model cache — do it deliberately).
