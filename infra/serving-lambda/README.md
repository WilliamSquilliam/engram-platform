# Lambda Cloud serving unit

A second drop-in serving unit, on cheaper GPUs. One Lambda Cloud box serves
**Command A+ W4A4** (`CohereLabs/command-a-plus-05-2026-w4a4`) on **1× B200**
(fallback **2× H100 SXM**) through the same cartridge stack the AWS unit runs, and
emits the same four contract values the control plane consumes. Launch the box,
provision it, point DNS, smoke-test it, paste four values into an env, done. Terminate
when idle and pay $0 for compute; relaunch in minutes because the model weights persist.

## The swappable-unit contract

The control plane (`backend/app/config.py` + `backend/app/serving.py`) consumes exactly
**four** values. Any serving stack that produces these four is a drop-in replacement —
the AWS unit, this Lambda unit, anything next. The control plane never learns the GPU
type or the cloud.

| Contract value | This unit's source | What it is |
| --- | --- | --- |
| `ML_SERVICE_URL` | `https://gpu-onboard.engramdynamics.org` | Onboarding / train worker, `:8001` |
| `INFERENCE_SERVICE_URL` | `https://gpu.engramdynamics.org` | vLLM resident-KV serve, `:8002` |
| `ML_AUTH_TOKEN` | `.state/ml_auth_token` (persisted once) | Shared bearer both processes enforce |
| `MODEL_REGISTRY_JSON` | one enabled `best` tier → Command A+ W4A4 | Product tiers → `model_ref`/precision/context |

`emit-env.sh` prints all four plus a ready-to-paste `envs/uat` tfvars snippet. The token
is printed only with `--show-secrets`, so a plain run never leaks it.

**Why HTTPS hostnames, not a private IP.** The AWS unit lives in the control plane's VPC
and is reached at a private IP. The Lambda box is off in another cloud, so it's reached
over public HTTPS: Caddy terminates TLS and reverse-proxies to the two services, which
bind to `127.0.0.1` only. The two hostnames A-record to the box's public IP.

## The three storage tiers

The whole point of the design: **cheap to keep, fast to bring back.** Three tiers, each
sized to what it holds and how much its loss hurts.

| Tier | Where | Survives terminate? | Holds | Cost while idle |
| --- | --- | --- | --- | --- |
| Local SSD root | `/home/ubuntu/engram` (2.75 TiB on 1× B200) | **No — wiped** | live HF weights cache, cart hot-mirror, cart registry | $0 (gone) |
| Persistent filesystem | `/lambda/nfs/engram-fs` (region-bound) | **Yes** | **seeded HF weights** (~120 GB) | ~$0.20/GB/mo |
| S3 cart bucket | `aws-support/` (`engram-carts-lambda-<acct>-us-east-1`) | **Yes** | durable cartridges (onboarded memory) | pennies/GB-mo |

**Weight seeding is what makes relaunch fast.** On boot, `engram-seed-in` rsyncs the
persistent FS's weights down to the local SSD before the engine starts. After the first
successful warm, `engram-seed-out` rsyncs the local weights up to the FS once. So the
first ever launch downloads ~120 GB; every launch after that seeds from the FS and skips
the download. The S3 cart store is durable, so onboarded memory survives too — the local
cart mirror just re-warms from S3 on demand.

## Run it

Prereqs: `LAMBDA_API_KEY` in the repo-root `.env` (scripts read it at runtime and never
print it), the AWS CLI + Terraform for `aws-support/`, `python3` with `build`
(`pip install build`), the sibling repo `../Engram-Smart-CAG`, and an SSH client. Optional:
`CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ZONE_ID` in `.env` for automatic DNS.

```bash
cd infra/serving-lambda

# 0. AWS SUPPORT — the durable cart bucket + a bucket-scoped IAM key for the box.
cd aws-support && terraform init && terraform apply && cd ..

# 1. LAUNCH — SSH key, filesystem, region+type discovery, launch, DNS, save state.
bash scripts/launch.sh
#    Prints the two A records to add if Cloudflare isn't wired. Point DNS before step 2.

# 2. PROVISION — build the wheel, generate serving.env (with the bucket-scoped AWS
#    creds), ship it, run bootstrap-lambda.sh (venv + vLLM + Caddy + systemd units).
bash scripts/provision.sh

# 3. SMOKE — over HTTPS: wait for engine_ready, compat_check (ssh), onboard one doc,
#    query it (grounded answer + metrics), describe it. PASS/FAIL per step.
bash scripts/smoke.sh

# 4. WIRE — print the four contract values + a UAT tfvars snippet.
bash scripts/emit-env.sh                 # token as a placeholder (safe to share)
bash scripts/emit-env.sh --show-secrets  # token inlined — paste into the gitignored tfvars
```

### How the four values reach the platform envs

The AWS unit publishes its four values into a terraform state that `platform-aws` reads
via `terraform_remote_state`. The Lambda unit keeps **no** serving terraform state (the
compute is on Lambda, not AWS), so it feeds the same four values through the platform
module's **override vars** instead. `emit-env.sh --show-secrets` prints exactly this
snippet — drop it into a gitignored `infra/platform-aws/envs/uat/<name>.tfvars`:

```hcl
read_serving_state             = false   # skip the serving remote-state read
ml_service_url_override         = "https://gpu-onboard.engramdynamics.org"
inference_service_url_override  = "https://gpu.engramdynamics.org"
ml_auth_token_override          = "<token from --show-secrets>"
model_registry_json_override    = "<the one-tier registry json>"
```

Same seam as the AWS unit (`ml_service_url` → `config.ML_SERVICE_URL`, etc.); only the
delivery differs (override vars vs. remote state). Nothing in product code changes.

## Relaunch cycle (idle = $0)

Lambda has no stop state — terminate is how you stop paying for compute. The design makes
that cheap and reversible:

```bash
bash scripts/terminate.sh                       # compute -> $0; local SSD dies, weights + carts persist
# ... later ...
bash scripts/launch.sh && bash scripts/provision.sh   # back up in minutes (FS weight seed + lazy cart re-warm)
```

Terminate drops the local SSD (the loss-tolerant tier), keeps the persistent FS (seeded
weights) and the S3 cart store (your memory), and keeps `.state/ml_auth_token` so the
platform env values don't churn. A relaunch gets a **new public IP**, so re-run
`launch.sh` to refresh the two A records (automatic if Cloudflare is wired).

## The B200 → 2× H100 fallback (one var)

`launch.sh` discovers instance types at runtime from Lambda's `/instance-types` and picks
the first with live capacity. `INSTANCE_TYPE_FILTER` (default `b200`) is the primary
target; if nothing matches with capacity, it falls back to `FALLBACK_FILTER`
(default `h100_sxm`, the 2× H100 SXM box). When you fall back, set `VLLM_TP=2` in
`provision.sh`'s env so vLLM shards Command A+ across the two H100s (the B200 default is
`VLLM_TP=1`). Nothing else changes — same model, same contract, same URLs.

## Cost

| Item | Cost | Billed when |
| --- | --- | --- |
| 1× B200 (primary) | **$6.99/hr** | running only |
| 2× H100 SXM (fallback) | market rate, per Lambda | running only |
| Persistent filesystem | **~$0.20/GB/mo** (~$24/mo for the ~120 GB weight seed) | always |
| S3 cart store | pennies/GB-mo; deleted versions expire after 30 days | always |

Terminated = $0 compute. You keep paying only the persistent FS and S3, both small next
to a GPU-hour. `emit-env.sh` surfaces the $6.99/hr B200 price to the control plane so its
measured $/query uses the real running price.

## Security model: firewall + TLS

Two layers, so a single misconfig doesn't expose the services:

- **Bind to localhost.** `:8001` and `:8002` bind to `127.0.0.1` only. They are never
  reachable from the network directly — Caddy is the only front door.
- **Caddy terminates TLS.** Two vhosts (`gpu` → `:8002`, `gpu-onboard` → `:8001`) with
  automatic HTTPS via Let's Encrypt HTTP-01. Every request still carries the
  `ML_AUTH_TOKEN` bearer, enforced on every route except `/health`.
- **Firewall to 22/80/443.** Open **only** SSH (provisioning), HTTP (Let's Encrypt
  HTTP-01 + the redirect), and HTTPS. Lambda firewall rulesets are **account-global**, so
  `launch.sh` sets them only when you pass `MANAGE_FIREWALL=1`; otherwise set the same
  three ports in the Lambda console. HTTP-01 needs 80/443 reachable to issue the cert.
- **Least-privilege cart access.** The box has no AWS instance-profile, so `aws-support/`
  mints an IAM user scoped to the one cart bucket (Get/Put/Delete/List) and provision.sh
  injects its access key into `serving.env`. If the key leaks, the blast radius is that
  one bucket. Rotate by tainting `aws_iam_access_key.serving` and re-provisioning.

## Files

| File | Role |
| --- | --- |
| `aws-support/` | Terraform: the S3 cart bucket + the bucket-scoped IAM user/key (state key `serving-lambda/aws-support.tfstate`) |
| `scripts/lib.sh` | Shared helpers: `.env` load, the Lambda API wrapper (key never printed), JSON reader |
| `scripts/launch.sh` | SSH key + filesystem + region/type discovery + launch + DNS + save `.state` |
| `scripts/provision.sh` | Build wheel, generate `serving.env`, ship + run `bootstrap-lambda.sh` over SSH |
| `scripts/bootstrap-lambda.sh` | Runs ON the box: venv + vLLM + three-tier storage + weight seeding + systemd + Caddy |
| `scripts/emit-env.sh` | Print the four contract values + a UAT tfvars snippet (`--show-secrets` for the token) |
| `scripts/terminate.sh` | Terminate the box (local SSD dies, weights + carts persist) |
| `scripts/smoke.sh` | HTTPS e2e: health → compat_check → onboard → query → describe |

## Notes

- The `.keys/` (SSH keypair), `.state/` (instance id/ip + the persisted token), and any
  `*.tfvars` / terraform state are gitignored — none of them ever gets committed.
- `bootstrap-lambda.sh` installs the **latest** vLLM (`>=0.26`) rather than pinning the
  AWS unit's `0.26.0`, because Blackwell (B200, SM100) needs recent CUDA wheels. If a
  fresh vLLM build regresses on Blackwell, pin a known-good version in that script.
- The scripts read `LAMBDA_API_KEY` from `.env` at runtime and route every Lambda call
  through `lib.sh`'s wrapper, which passes the bearer via a curl config on stdin so the
  key never lands in a process list or a log.
