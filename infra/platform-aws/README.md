# Engram platform on AWS — two environments (UAT + prod)

Runs the control plane (FastAPI backend) and the Next.js frontend on AWS Fargate as
**two parallel environments** — `uat` and `prod` — in the **same account and VPC** as
the GPU serving unit (`infra/serving-aws`). One shared public ALB fronts four hosts;
each env owns its own database, document bucket, secrets, and services.

```
                      Cloudflare CNAMEs (added by hand)
   app.engramdynamics.org ─┐   uat-app.engramdynamics.org ─┐
   api.engramdynamics.org ─┤   uat-api.engramdynamics.org ─┤
                           ▼                               ▼
                 ┌───────────────── one PUBLIC ALB (:443, HTTP→HTTPS) ─────────────────┐
   host rules:   │  api. → prod backend   app. → prod frontend                          │
                 │  uat-api. → uat backend   uat-app. → uat frontend                    │
                 └───────────────────────────── shared ECS cluster ─────────────────────┘
     prod:  prod-engram-backend (:8000)  prod-engram-frontend (:3000)  RDS  S3  secrets
     uat:   uat-engram-backend  (:8000)  uat-engram-frontend  (:3000)  RDS  S3  secrets
                                    │
                                    ▼  (the four contract values, via remote state)
                      GPU serving unit (infra/serving-aws) — SHARED by both envs
```

## The four stacks and their state keys

Each stack has its own key in the shared S3 backend
(`cartridge-tfstate-808379776072-us-east-1` + the `cartridge-tflock` DynamoDB lock),
so they apply and destroy independently — same pattern as `serving-aws`.

| Stack | Dir | State key | What it owns |
| --- | --- | --- | --- |
| common | `common/` | `platform-aws/common.tfstate` | ECS cluster, 2 ECR repos, public ALB (+ :80→:443 redirect, :443 listener), ACM cert (4 hosts), ALB SG |
| uat | `envs/uat/` | `platform-aws/uat.tfstate` | UAT services, target groups, host rules, RDS, S3, secrets, task IAM |
| prod | `envs/prod/` | `platform-aws/prod.tfstate` | prod services, target groups, host rules, RDS, S3, secrets, task IAM |
| module | `modules/platform-env/` | (none — used by uat + prod) | the reusable per-env definition |

The account guard (808379776072) is in every root, exactly like `serving-aws`.

## Apply order

`common` first (the env stacks read its ALB / cluster / ECR / cert via
`terraform_remote_state`), then either env.

```bash
cd infra/platform-aws

# 1. COMMON — cluster, ECR repos, ALB, ACM cert (four-host, DNS-validated).
terraform -chdir=common init
terraform -chdir=common apply
#   apply RETURNS immediately even though the cert is PENDING_VALIDATION — it does
#   not block on validation. Do the Cloudflare handoff (below) now; the cert issues
#   in minutes once the CNAMEs are live, and the :443 listener goes green then.

# 2. BUILD + PUSH the images (backend once; frontend per env).
bash build_push.sh                       # tags: backend :<sha>, frontend :<sha>-uat / :<sha>-prod

# 3. UAT — stand it up, deploy the sha, verify at uat-app.
terraform -chdir=envs/uat init
terraform -chdir=envs/uat apply
bash deploy.sh uat <sha>                  # or omit <sha> to use the current git sha

# 4. PROD — same sha, once UAT looks right.
terraform -chdir=envs/prod init
terraform -chdir=envs/prod apply
bash deploy.sh prod <sha>
```

## The Cloudflare records handoff (added by hand)

Two sets of CNAMEs are printed by the `common` stack; the operator adds them in
Cloudflare (DNS-only / grey cloud is simplest). Same manual handoff style as the SES
DKIM records.

```bash
# 1/2  cert validation — add each name→value CNAME so ACM issues the cert.
terraform -chdir=common output -json acm_validation_cname_records

# 2/2  the four host records — each host CNAMEs to the ALB DNS name.
terraform -chdir=common output -json host_cname_records
```

- Validation CNAMEs move the cert from `PENDING_VALIDATION` to `ISSUED` (minutes). The
  HTTPS listener already references the cert ARN, so it goes live automatically.
- The four host CNAMEs point `app.` / `api.` / `uat-app.` / `uat-api.` at the ALB. The
  ALB's host-header rules then route each to the right env's service.

## Build / deploy flow (image → UAT → prod)

`build_push.sh` builds the **backend once** (env-agnostic, tag = short git sha) and the
**frontend per env** (because `NEXT_PUBLIC_API_URL` is inlined at build time — the SPA
must be built pointing at `uat-api.` vs `api.`), tagged `<sha>-uat` / `<sha>-prod`.

```bash
bash build_push.sh                # backend :<sha>, frontend :<sha>-uat, :<sha>-prod
bash deploy.sh uat <sha>          # roll UAT's two services to that sha; wait for stable
#   → verify at https://uat-app.engramdynamics.org
bash deploy.sh prod <sha>         # SAME sha to prod once UAT is verified
```

`deploy.sh` reads the current (terraform-managed) task definition, swaps only the
container image to the requested tag, registers a new revision, updates the service, and
waits for steady state. Terraform still owns the task defs; pin a promoted sha into the
env's `*_image_tag` var when you want `terraform apply` to re-assert it.

## How the serving-unit values flow in

The GPU serving unit (`infra/serving-aws`) emits its four contract values as terraform
outputs into its own state (`serving-aws/terraform.tfstate`). The `platform-env` module
reads that state via `terraform_remote_state` and feeds the backend task env:

| Contract value | Backend env var | Source |
| --- | --- | --- |
| `ml_service_url` | `ML_SERVICE_URL` (:8001) | serving state output |
| `inference_service_url` | `INFERENCE_SERVICE_URL` (:8002) | serving state output |
| `ml_auth_token` | `ML_AUTH_TOKEN` (secret) | serving state output → Secrets Manager |
| `model_registry_json` | `MODEL_REGISTRY_JSON` | serving state output |
| `cart_bucket` | `CARTRIDGE_STORE_BUCKET` | serving state output |

Each value has an `*_override` var (empty by default). To point **UAT at a throwaway
ephemeral serving unit** (so a serving-side rehearsal never touches the customer GPU),
set `serving_state_key` to that unit's state or pass the four `*_override` vars in
`envs/uat/*.tfvars`. `INFERENCE_BACKEND=vllm` and `PLATFORM_ADMIN_EMAIL` /
`ALLOW_REGISTRATION=false` / `EMAIL_BACKEND=ses` are wired in the module.

Because both envs default to the **same** serving state, UAT and prod share the one GPU
box — see the guardrails.

## The default-VPC coupling (important)

The serving unit runs in the account's **default VPC** (`vpc-00f7fc0cc4895aa81`,
`172.31.0.0/16`, six public subnets). The platform envs MUST join the **same** VPC so the
Fargate tasks reach the GPU box on its private IP over the VPC. `common` therefore
defaults `vpc_id`/`subnet_ids` to the default VPC via data sources, and the env stacks
inherit that VPC from `common`. Tasks run in the public subnets with a public IP (there is
no NAT gateway in the default VPC) so they can pull from ECR / reach SES / S3; ingress is
still locked to the ALB security group only. If the serving unit is ever moved off the
default VPC, set `common`'s `vpc_id` (and `serving-aws`'s) to the same explicit VPC.

## SES (operator TODO — in flight)

Prod transactional email needs **SES production access + a verified sending domain**
(`engramdynamics.org`, from-address `will.stephenson@engramdynamics.org`). Until that is
granted the backend runs with `EMAIL_BACKEND=ses` but SES will reject sends outside the
sandbox. Verify the domain (DKIM CNAMEs — another Cloudflare handoff) and request
production access in the SES console. This is already in flight.

## Cost (both envs)

| Item | Monthly |
| --- | --- |
| Public ALB (shared) | ~$16 |
| Fargate — 2 envs × (backend 0.5 vCPU/1GB + frontend 0.25/0.5GB) | ~$30 |
| RDS — 2 × `db.t4g.micro` (single-AZ, gp3) | ~$24 |
| S3 doc buckets + ECR storage + logs | a few $ |
| **Total (both UAT + prod)** | **~$75–95/mo** |

The GPU serving box is billed separately by `serving-aws` (its own $/hr). Stopping it
when idle is the big lever; the platform envs above are the fixed, cheap always-on layer.

## Guardrails

- **UAT shares the one GPU serving unit with prod.** Do NOT run bulk onboards from UAT
  during customer hours — a large onboarding job saturates the box's onboarding worker
  (:8001) and degrades prod chat latency. Onboard in bulk off-hours, or point UAT at an
  ephemeral serving unit first.
- **Serving-side rehearsals use ephemeral spot serving units**, never the customer box.
  Stand up a throwaway `serving-aws` (spot) and point UAT at it via the `*_override`
  vars / `serving_state_key`, so a risky serving change is proven in isolation.
- **Same sha, UAT then prod.** Never build a separate prod image — promote the exact sha
  verified in UAT (frontend differs only by the baked API host).

## Notes

- All four stacks share the encrypted, versioned S3 backend + DynamoDB lock. The env
  states hold generated secrets (JWT/SESSION/INTERNAL, the DB URL) — hence encrypted and
  never committed.
- The backend image now bundles `libreoffice-writer` (`soffice`) for genuine legacy `.doc`
  → `.docx` conversion, and pins `FASTEMBED_CACHE_DIR=/data/fastembed` so the dense
  retriever's ONNX model caches to a writable dir.
- `config.validate()` fail-fasts are all satisfied by the module: strong
  JWT/SESSION/INTERNAL/ML_AUTH tokens (≥32), Postgres `DATABASE_URL`, explicit
  `CORS_ORIGINS=https://<app-host>` (no localhost, no `*`), `EMAIL_BACKEND=ses`,
  `ENV=production`.
```
