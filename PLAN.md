# Engram Platform — Inference-as-a-Service Build Plan

Status: draft v0.1 (2026-08-30). This is a living document. Open questions are marked
**[Q]** and tracked in the "Open questions" section; decisions get recorded in
"Decisions log" as we lock them.

---

## 1. The product (the pivot)

**What we're building:** a fully self-service Inference-as-a-Service platform where a
customer connects their document base, we onboard it into resident-KV **cartridges**, and
they get fast, cheap grounded answers over it — through a polished chat interface *and*
through an MCP endpoint so their own agents can query it.

**The shift:** the old go-to-market was a pip-installable vLLM plug-in
(`engram-cartridge`) sold to inference providers. The new go-to-market is **our own hosted
platform** that runs that plug-in inside our serving engine. The customer never touches
vLLM, connectors, or KV internals — they upload documents and chat.

**What stays the core IP:** the `cartridges/` library in the `Engram-Smart-CAG` repo (the
KV cartridge format, the vLLM `CartridgeKVConnector`, the cart store, onboarding, and the
compat/model-binding gates). This platform *consumes* it in the serving engine. **[Q1]**
covers how we consume it (pip dependency vs submodule vs monorepo).

**The one-line value prop (customer-facing):** connect your documents once, then chat with
them or wire them into your agents — answers that are as accurate as search-based AI, at a
fraction of the latency and cost, and they don't slow down as your library grows.

**Who it's for:** teams with large, slowly-changing, heavily-queried corpora — internal
knowledge bases, legal/regulatory libraries, product/support docs, code repositories.

---

## 2. The single biggest lever: reuse `Engram-Smart-CAG/platform/`

That directory is a complete, tested, 4-service monorepo that already does register →
create corpus → upload docs → onboard cartridges → chat/compare → expose via MCP. It is
**well past demo-grade toward production**. We fork/migrate it into this repo and
productize, rather than rebuild.

| Service | Stack | Role | Reuse verdict |
|---|---|---|---|
| `backend/` | FastAPI + SQLAlchemy 2.0 + Alembic | Control plane: auth, tenants, corpora, docs, jobs, chat, MCP-query, retrieval | **Reuse, harden** |
| `ml_service/app.py` | FastAPI + torch + `cartridges` | GPU worker: onboard (one forward pass) + HF inference + fused retrieval | **Reuse** |
| `ml_service/vllm_inference.py` | FastAPI + vLLM ≥0.26 | Inference Service: serve resident-KV carts by `doc_id` via `CartridgeKVConnector` | **Reuse (this is the product serving path)** |
| `mcp_server/` | official `mcp` SDK (FastMCP) | Per-corpus MCP tool for a tenant's agents | **Reuse, expand** |
| `frontend/` | Next.js 14 + Tailwind 3.4 + TS | Onboarding wizard, chat/compare, scale test, MCP panel | **Reuse shell, rebuild chat, add doc mgmt** |

Clean pluggable seams already exist (one env switch each): SQLite→Postgres
(`DATABASE_URL`), filesystem→S3 (`PLATFORM_STORAGE_BACKEND`), inline jobs→RQ/Redis
(`JOB_BACKEND`), local JWT→OIDC (`AUTH_BACKEND`), BM25→fused retrieval
(`RETRIEVAL_BACKEND`), HF→vLLM serving (`INFERENCE_BACKEND`). Production is the "right"
side of every switch.

---

## 3. Target architecture

```
                          app.engramdynamics.org (Next.js SPA)
                                     |
                          api.engramdynamics.org
                          ┌─────────────────────────┐
                          │  Control plane (FastAPI) │  torch-free
                          │  auth · tenants · corpora │
                          │  docs · jobs · chat · MCP │
                          │  retrieval · billing      │
                          └───────────┬──────────────┘
                        onboard/query │ (ml_client httpx seam)
                          ┌───────────┴──────────────┐
                          │   GPU plane               │
                          │  app.py     : onboard     │  cartridges lib
                          │  vllm_infer : serve carts │  CartridgeKVConnector
                          └───────────┬──────────────┘
                                      │
                          S3 cart store  +  Postgres  +  Redis
```

- **Control plane** owns identity, tenancy, the document catalog, job orchestration,
  retrieval (which `doc_id`s answer a question), chat sessions, MCP tokens, and billing.
  It never imports torch.
- **GPU plane** owns onboarding (one frozen forward pass per doc → a cartridge) and
  serving (resident-KV answer over routed `doc_id`s). The cartridge connector makes it so
  only the ~15-token question is re-prefilled, not the documents.
- **Cart store** (S3 + local hot mirror) is the durable, addressable KV tier.
- **MCP** is a first-class product surface: each corpus (or tenant) gets an MCP endpoint so
  external agents (Claude, etc.) query the same grounded inference the chat UI uses.

---

## 4. Build scope (epics)

Each epic notes what already exists (reuse), what we add, and the key files in the
reference platform.

### E1 — Auth & self-service sign-on  → **built-in email + Google (locked) · ✅ core built (2026-08-31)**
- **Exists:** email/password → HS256 JWT (PBKDF2, 600k iters), multi-tenant, Google OAuth
  (real client already provisioned), pluggable OIDC backend (`AUTH_BACKEND=oidc`) for later
  enterprise SSO. Registration gate (`ALLOW_REGISTRATION`).
  Ref: `backend/app/routers/auth.py`, `backend/app/security.py`, `backend/app/deps.py`.
- **Add:** invite/waitlist-gated signup ([Q5]/[Q7]: keep public registration off; a public
  "request access" form feeds an admin approval queue), email verification + password reset, "Continue
  with Google" in the product UI, org/workspace model, invite teammates, roles. Fix the
  committed-secret issue (rotate `JWT_SECRET`/`SESSION_SECRET`, untrack `.env.local`).
  Enterprise SSO (SAML/OIDC) deferred — the OIDC backend seam already exists for it.

### E2 — Document base management  → **upload + PDF/DOCX parsing + SharePoint + Google Drive (locked)**
- **Exists:** corpus CRUD, multipart upload (idempotent on `(corpus, filename)`), doc list,
  thorough corpus delete with invalidate→offboard→audit receipt, operator GC for orphans.
  Ref: `backend/app/routers/corpora.py`, `backend/app/routers/jobs.py`, `backend/app/storage.py`.
- **Add (the real "document base" page):**
  - **Ingestion / connectors** — today only direct file upload + a `FilesystemConnector`
    ABC; SharePoint/Confluence/Drive/S3 are stubs (`connectors/base.py`) and not wired to the
    upload router. **MVP builds two live connectors — SharePoint and Google Drive** — plus the
    connector→ingest pipeline (OAuth to the source, list/select folders, pull files, sync). The
    Google OAuth client we already have can seed Drive access scopes. **✅ Framework done
    (2026-08-31):** config-driven connector registry + `GET /connectors`; Drive/SharePoint
    scaffolds gated `available:false` until OAuth creds are set. **Remaining: the actual OAuth
    flows** (Drive scopes on the existing Google client; a new Azure AD app for SharePoint) +
    folder pick/pull/sync.
  - **Document parsing** — today plain `.txt`/`.md` only. Add PDF/DOCX/HTML extraction
    (e.g. `unstructured`), chunk/normalize before onboarding. **✅ Done (2026-08-31):**
    `parsing.py` (pypdf/python-docx/beautifulsoup4) writes an extracted-text sidecar the onboard
    path consumes; the wizard surfaces per-doc `parse_status`/`parse_error`.
  - **Per-document CRUD** — add per-doc delete/replace endpoints (only whole-corpus delete
    exists today).
  - **Change detection / re-onboard** — the mechanics exist (idempotent re-onboard under the
    same slug + auto `inference_invalidate`; a `doc_version` metadata hook on the store), but the
    *decision* of which docs are stale is not automated. Add content-hash/mtime staleness so an
    edited doc re-caches automatically.
  - **Lifecycle UI** — status per doc (queued/onboarding/ready/failed/stale), sizes, token
    counts, last-onboarded, re-onboard button, delete with the audit trail surfaced.
  - **Resumable onboarding flow (per-corpus) — [Q6] locked** — a 5-step wizard: (1) name,
    (2) add documents (upload / Google Drive / SharePoint), (3) choose model, (4) review +
    estimate (doc count, detected types, est. onboarding time/cost) then confirm, (5) onboard
    with live per-doc progress (queued→parsing→onboarding→ready, cancelable). The corpus is saved
    at step 1 and each step persists (add `Corpus.onboarding_step` + per-doc parse/onboard status
    on top of `Job.progress`), so exiting returns to the current step; onboarding runs
    server-side so a closed tab never stops it. Model step = tiers (Fast/Balanced/Best) with the
    backing model name shown; entries placeholder. See E5 for model binding + serving.
    **✅ Backend built + tested (2026-08-31):** schema + Alembic `0005`, `PATCH/GET
    /corpora/{id}/onboarding`, `GET /corpora/{id}/estimate`, `POST /corpora/{id}/onboard` (gated
    on an *available* serving tier → 409 `no_serving_engine` while tiers are placeholders, so the
    flow is fully testable with no GPU); reuses the job/progress/cancel path. 79-test suite green.
    Remaining: the frontend wizard UI + real document parsing wiring the per-doc `parse_status`.

### E3 — Chat experience (Google/Anthropic-grade)  → **chat-only MVP (locked) · ✅ built (2026-08-31)**
- **Exists:** a Gemini-style collapsible app shell (`components/AppShell.tsx`), a streaming
  **Compare** view (Smart CAG vs RAG side-by-side, SSE) that is really a *sales* surface, and
  a `POST /corpora/{id}/chat` API + `api.chat()` that aren't yet a conversational page.
  Ref: `frontend/app/corpus/[id]/(workspace)/chat`, `backend/app/routers/chat.py`.
- **Add:** a clean, single-answer **conversational chat** page that looks and feels like
  Claude/Gemini — message threads, streaming tokens, markdown + code rendering, citations back
  to source documents (the cart path grounds answers), conversation history, corpus/model
  picker, stop/regenerate, copy. Replace hand-vendored `components/ui.tsx` primitives with a
  real design system (shadcn/ui — already the noted intended swap).
- **Decision ([Q4]):** the product is chat-only. **Cut Compare / Scale-test / Costs from the
  product UI** (keep them as internal/sales tools outside the app). This shrinks the frontend
  surface to: auth, the document-base page, the onboarding flow, and chat.

### E4 — MCP as a first-class product surface
- **Exists:** a single-file FastMCP server exposing one tool `query_corpus(question, k)`,
  per-corpus, authed by `X-MCP-Token`, proxying to `POST /mcp/{id}/query`.
  Ref: `mcp_server/server.py`, `backend/app/routers/chat.py`.
- **Add:** self-service MCP endpoint provisioning in the UI (generate/rotate per-corpus or
  per-tenant tokens, copy-paste client config), a hosted remote MCP endpoint (not just stdio)
  so agents connect over the network, more tools (list documents, cite, multi-corpus query),
  rate limits + usage metering per token.

> Consumption (locked): the GPU plane installs `engram-cartridge` as a **pip dependency**.
> Add a build/publish step for the package from `Engram-Smart-CAG/cartridges/` (internal
> index or pinned VCS ref), and pin the version this platform is validated against.

### E5 — Serving engine integration (the cartridge/CAG plugin)
- **Exists:** `vllm_inference.py` serves resident carts by `doc_id` through
  `CartridgeKVConnector` via `KVTransferConfig`; onboarding builds one CAG cart per doc
  (`cag_carts_batch`); model-binding gate (HTTP 409 on mismatch), `format_version` load-guard,
  `compat_check` CLI; async batching, TP, FP8, speculative decoding supported.
  Ref: `Engram-Smart-CAG/cartridges/serve/`, `ml_service/vllm_inference.py`.
- **Add:** productionize model choice (see the enabled beta model below), GPU
  capacity/autoscaling, the multi-tenant multiplexing budget (many tenants' carts hot on shared
  GPUs — the real moat), warm/cold tiering, and health/observability. **[Q1]** decides how
  `cartridges/` is pulled in.
- **Enabled beta model (locked 2026-08-30): Cohere Command A+, served 4-bit on one
  `g6e.12xlarge`.** `CohereLabs/command-a-plus-05-2026` (Apache 2.0, 218B / 25B-active MoE, native
  grounding/citation spans). Picked for best faithful grounded fact-QA that fits the box; served
  4-bit because FP8 (~218 GB) overflows 192 GB while the lossless QAD 4-bit is ~120 GB. Carts are
  bf16 and orthogonal to weight precision, so the connector is unaffected — onboard and serve on
  the **same** checkpoint (`model_ref` binding). **Fallback: Llama-3.3-70B FP8** (~71 GB, proven
  on L40S) if the Ada 4-bit path underperforms. **Open verification:** Command A+'s lossless path
  is H100/B200-native (W4A4); on L40S/Ada expect 4-bit-weight (W4A16) — confirm parity with a
  grounded-QA A/B vs the fallback before it goes live.
- **Serving is an interchangeable interface (2026-08-31): `backend/app/serving.py` + a
  config-driven model registry.** The control plane reaches the GPU plane only over HTTP and
  names models by `model_ref`; GPU/cloud/precision live in deploy config, not code. Tiers come
  from `MODEL_REGISTRY_JSON` (placeholders until a box is chosen), exposed at `GET /models` for
  the onboarding menu; the client sends a tier `id`, the backend maps it to weights
  (`serving.model_ref_for_tier`). The Command A+ choice above is the intended registry entry to
  wire once the hardware/cloud lands — swapping the model or the box is config, not code.
  **Note (VLM):** Command A+ is a vision+text model (SigLIP2-class encoder, ~1–2 GB VRAM, does
  not affect carts); the text-only Llama-3.3-70B / Qwen2.5-72B remain first-class registry
  candidates if the vision baggage or the 4-bit-on-Ada risk isn't worth it.
- **Per-corpus model binding (from [Q6], locked):** the onboarding model is chosen per corpus
  (as a tier → model mapping) and stamped into every cart (`model_ref`;
  `cartridges/model_binding.py`, 409 on mismatch). A corpus is **pinned** to its model (changing
  it = full re-onboard). **Beta serving runs a single active model** — the selection UI is
  present but one model is enabled and one GPU engine serves it, so no cross-model routing is
  needed yet. Deferred until demand is real: the multi-model fleet (engine-per-model or on-demand
  model load) with the budget manager multiplexing carts per model. A `tier → model_ref` config
  table is the seam that lets us add/swap models without touching the UI.

### E6 — Multi-tenancy hardening (correctness, not optional)  → **✅ done (2026-08-31)**
- **Issue:** today cart slugs are **shared across tenants** — identical filenames collide to
  the same KV blob (a storage-dedup optimization for public corpora). For a real SaaS with
  customers' **private** documents this is a data-isolation problem.
- **Add:** per-tenant (or per-corpus) cart namespacing so no two tenants can ever address the
  same blob; scope retrieval, serving, invalidation, and GC to the namespace. Accept the loss
  of cross-tenant dedup in exchange for isolation.

### E7 — Self-service & monetization  → **invite-only beta first (locked)**
- **MVP:** invite/waitlist gate (E1) + **usage metering from day one** (tokens onboarded,
  queries served, storage GB, GPU-seconds) so we have real numbers before pricing. The
  `Measurement` table already captures per-query metrics — the metering foundation.
- **Deferred to a later phase:** plan tiers, quota enforcement, and Stripe billing +
  open self-serve, turned on once pricing is validated during the beta.

### E8 — Infra, deploy, domains  → **✅ built (2026-09-02); awaiting operator applies**
- **Exists (reuse):** AWS account `808379776072`, `us-east-1`, profile `Engram-Dynamics`.
  Terraform for ECS (backend/frontend/worker) + GPU EC2 (`g6e`, L40S) + RDS Postgres 16 +
  ElastiCache Redis + S3 + Secrets Manager + Budgets, currently mothballed to save spend.
  Domain `engramdynamics.org` at Cloudflare; landing site live on S3+CloudFront; a live
  invite-only Cognito pool (`us-east-1_jtvbJ65Qe`) for members.
  Ref: `Engram-Smart-CAG/infra/terraform/`, `engram-dynamics-landing/infra/`.
- **Add:** `app.engramdynamics.org` (frontend) + `api.engramdynamics.org` (backend) DNS at
  Cloudflare (the `app.`/`api.` scheme is already assumed in `.env.aws.example`), bring the
  mothballed stack up, CI/CD (CodeBuild→ECR→ECS exists), secrets via Secrets Manager.
- **Two-environment shape (2026-09-02):** the platform stack is ONE Terraform module
  instantiated twice — `env=prod` and `env=uat` — in the same account/VPC: per-env ECS services,
  DB, secrets, S3 doc prefix, and `uat-app.`/`uat-api.` Cloudflare names; both consume the SAME
  serving unit's four env values. Deploy flow: image → UAT (migrations rehearse on the UAT DB) →
  click-through at uat-app → same image to prod. Serving-side rehearsals use ephemeral spot
  serving units, never the shared box.
- **GPU serving box (locked): one `g6e.12xlarge` (4× L40S, 192 GB) in `us-east-1`.** Quota
  verified 2026-08-30 in the Engram Dynamics account: **On-Demand G&VT = 64 vCPU, Spot = 48 vCPU,
  nothing running** → one box (48 vCPU) launches now. AWS has no small H100 shape (H100 = 8×
  `p5.48xlarge`, ~$40k/mo), so L40S is the right fit and stays in-VPC. A **second serving box
  (HA / capacity) needs a Service Quotas bump to 96+ vCPU on-demand** — user pursuing. Note: quota
  ≠ capacity — use an On-Demand Capacity Reservation for a guaranteed always-on box.

### E9 — Security & compliance
- ✅ Done (with a correction): the committed `.env.local` actually held documented dev
  *placeholders* (`dev-secret-change-me`), not real secrets; the real Google creds lived in
  the gitignored `.env` (never committed) and were preserved into this repo. `.env.local` is
  now gitignored here with freshly rotated local secrets. No real exposure occurred.
- Tenant isolation review (E6), per-doc auth, MCP token scoping, audit receipts (exist),
  data-deletion guarantees (invalidate→offboard→audit exist), S3 recycle-bin/versioning
  (described in `DATA_LIFECYCLE.md` but the `deploy/byoc/` CFN is not in the working tree —
  verify before relying on the 30-day recovery guarantee).
- **AWS root credentials in use (flagged 2026-08-30):** CLI calls to the account authenticate as
  `arn:aws:iam::808379776072:root` — root access keys on a local machine. Before deploying, create
  a least-privilege IAM user/role for automation and disable/rotate the root access keys.

### E10 — Tenant "Admin Dashboard" (customer-facing; tenant-admin only)  → **✅ built (2026-08-31)**
- **Who:** a tenant's admin user(s) (`User.role = admin`); members get a reduced/no view. Add a
  `require_tenant_admin` dependency (admin-role gate on top of `get_current_user`). Shown to the
  customer as simply **"Admin Dashboard."**
- **Sections:**
  - **Usage** — queries served, documents onboarded, resident storage (GB carts), GPU-seconds;
    per corpus and aggregate, over time. Backed by the `Measurement` table (per-query metrics
    already captured) + job/onboarding records + cart-store sizes.
  - **Costs** — the tenant's *own* spend/bill and plan consumption (distinct from the cut sales
    "Costs" demo tab). Ties to E7 metering.
  - **Team & access** — list tenant users + roles, invite/remove teammates, pending invites,
    role changes (who on the team has access). Ties to E1 (org/workspace, invites, roles).
  - **Billing & payment** — payment method, invoices, plan tier + limits, subscription mgmt
    (Stripe; management lands with E7/Phase 3 — the shell + read-only plan/limits can ship first).
  - **Access tokens** — manage/rotate/revoke MCP endpoint tokens (per-corpus/tenant; ties to E4).
  - **Data & audit** — deletion receipts / audit log (exists), export/delete controls, retention.
  - **Limits & alerts** — plan quotas vs current consumption, usage-threshold alerts.
- **Build:** tenant-scoped `/admin/*` endpoints aggregating `Measurement`/jobs/storage; a frontend
  "Admin Dashboard" section in the tenant nav (admin-only). Reuses the metering foundation (E7)
  and existing audit/team primitives.

### E11 — Platform Admin console (operator-only — the founder)  → **✅ built (2026-08-31)**
- **Who:** ONLY the platform owner. Needs a new authz tier above tenant roles — a
  `platform_admin`/superuser flag on `User` (or a separate operator identity) — on a separate
  route namespace (`/platform-admin/*`) with strict checks and its own audit. Cross-tenant data
  is sensitive; lock it down and never expose it in the tenant app.
- **Sections:**
  - **Tenants** — every tenant: the users added to it (who has access), signup date, plan, status.
  - **Cost per tenant** — aggregate usage → cost per tenant (GPU-seconds, storage GB, queries),
    plus fleet totals — the operator's "who costs what" view.
  - **Invite / waitlist approvals** — the manual-approval queue for the invite-only beta
    ([Q5]/[Q7]) lives here: approve/deny access requests.
  - **Operator controls** — per-tenant limit/quota overrides, suspend/reactivate, support access
    (impersonate with audit), global health, revenue/MRR once billing lands.
- **Build:** cross-tenant read models over the same metering tables (the one place that
  intentionally spans tenants — NOT tenant-scoped), a `require_platform_admin` dep, and a separate
  frontend surface rendered only for the superuser. Keep it isolated from the tenant app (own
  routes, own nav, heavy authz + audit).

### E12 — GPU serving controls in the Platform Admin console  → **✅ built (2026-09-02)**
- **What:** start / stop / live status of the Lambda GPU serving box, from the Platform Admin tab.
  Lambda has no pause state, so Stop = terminate (compute drops to $0/hr; document bases, carts
  (S3), and model weights (persistent FS) all survive) and Start = launch a fresh box that
  **provisions itself unattended**: cloud-init `user_data` runs `self-provision.sh` off the
  persistent FS (provision.sh publishes the bundle there), so the control plane never SSHes.
- **Backend:** `lambda_cloud.py` (httpx client over the official REST API), `cloudflare_dns.py`
  (A-record upsert with drift check), `routers/gpu_admin.py` (`/platform-admin/gpu/status|start|stop`,
  all `require_platform_admin`). Status derives offline→booting→provisioning→warming→serving from
  the Lambda instance + the two health probes, and best-effort re-points the gpu/gpu-onboard DNS
  records at the new IP on every poll. Start ensures the 22/80/443 firewall, then picks type+region
  by discovery: `b200` preferred, `2x_h100_sxm` fallback, intersected with regions hosting
  `engram-fs`. Secrets (LAMBDA_API_KEY, CLOUDFLARE_*) ride the per-env Secrets Manager map; empty =
  panel hidden.
- **Frontend:** "GPU Serving" card atop the Platform Admin console — status pill, type/region/IP,
  $/hr, Start/Stop with honest confirm dialogs, adaptive 10s/60s polling.

---

## 5. Reuse-vs-build summary

**Reuse largely as-is:** control-plane skeleton, auth (JWT + Google + OIDC), tenancy model,
corpus/job orchestration with progress + cancel, the full delete/audit/GC lifecycle, the
GPU-plane onboarding + vLLM cartridge serving path, Alembic migrations, Docker/compose, the
app shell + upload wizard.

**Build new or significantly extend:** conversational chat UI (E3), external ingestion
connectors + document parsing + per-doc CRUD + change detection (E2), remote/self-service
MCP (E4), per-tenant cart namespacing (E6), metering/billing (E7), production auth self-serve
flows (E1), bring-up + DNS (E8), the tenant Admin Dashboard (E10), and the operator-only
Platform Admin console (E11).

**Known gaps to close:** `RETRIEVAL_BACKEND=pgvector` raises `NotImplementedError` (only
bm25 + fused work); no document parsing beyond text; connectors unwired; no per-doc delete;
MCP is one stdio tool; cross-tenant slug sharing; the S3 recycle-bin CFN is missing from the tree.

---

## 6. Phased roadmap

- **Phase 0 — Land the codebase. ✅ complete (2026-09-01).** Migrated `platform/` into this
  repo, repointed to the `engram-cartridge` pip dependency, untracked/rotated secrets, and
  completed the local end-to-end run-up (waitlist → approve → invite → wizard → queued onboard →
  team → both dashboards, walked in a real browser; 3 bugs found + fixed). Chat end-to-end
  remains gated on a serving box (GPU plane).
- **Phase 1 — Product chat + doc management MVP.** E3 conversational chat with citations; E2
  document-base page with per-doc CRUD, parsing (PDF/DOCX), and change-detection; E6 per-tenant
  namespacing; E1 invite-gated signup + Google. Exit: an invited user can onboard an uploaded
  corpus and chat, isolated from other tenants.
- **Phase 2 — Connectors + MCP + serving in the cloud.** E2 SharePoint + Google Drive
  connectors (the locked ingestion scope); E4 remote self-service MCP; E5/E8 bring up the GPU
  plane on AWS with the vLLM cartridge path behind `api.engramdynamics.org`; E7 usage metering;
  observability. Exit: an invited tenant connects a live source, chats, and wires an agent via
  MCP against GPU-served carts, with usage metered.
- **Phase 3 — Open self-serve.** E7 plan tiers + Stripe billing + quotas; open registration
  (lift the invite gate); enterprise SSO via the OIDC seam; autoscaling/multiplexing.
  Exit: a stranger can sign up, connect a source, pay, and use it end-to-end.

---

## 7. Open questions

**Resolved (see Decisions log):** [Q1] pip dependency · [Q2] built-in email + Google ·
[Q3] upload + PDF/DOCX parsing + SharePoint + Google Drive · [Q4] chat-only MVP ·
[Q5] invite-only beta first · [Q6] onboarding flow + model selection · [Q7] waitlist + manual admin approval.

**Deferred (not blocking the MVP build):**
- **Fill the model menu.** Beta enabled model = **Command A+** (see Decisions log). Still open:
  which models back the other Fast / Balanced / Best tiers.
- **Multi-model routing.** Engine-per-model vs on-demand model load, and how the budget manager
  multiplexes carts across models — build when a second model is enabled.
- **30B accuracy gate.** The one open research item if/when a 30B tier is offered.

---

## 7b. Backlog (the single consolidated list — 2026-09-02)

Everything queued, in one place, with its trigger. Newest wins float up; done items move to
the Decisions log.

**Before general availability (gates wider access):**
- **Billing enablement (Stripe).** Two-meter pricing decided (see Decisions log 2026-09-02):
  memory $0.03/doc/mo + inference $1.30/1k queries, onboarding FREE; platform fee per
  workspace (~$199–499/mo) as the floor; enterprise = dedicated serving instance at annual
  commit (~GPU cost ×2.5–3 + discounted meters). Build: 2 metered prices + webhook handler +
  tenant→customer mapping + Stripe Customer Portal button in the E10 dashboard; usage-report
  job reads the SAME tables the E11 console reads. Update backend pricing.py + BILLING_PLAN.md
  + estimator meters together.
- **Fair-use terms.** Free onboarding needs two clauses before GA: onboard-and-delete thrash
  (any started month bills in full) and re-onboarding churn (reasonable-sync language until
  change detection ships).
- **HttpOnly-cookie session migration** (security sweep M2): move JWTs out of Web Storage;
  requires CSRF protection with it.
- **SES production access** (operator: AWS request) — email is sandbox-gated today.
- **Per-doc delete + change detection.** Was product polish; now ALSO the free-onboarding
  guardrail (skip unchanged docs on re-sync). Priority up.

**Product (post-beta value):**
- **E4 remote/self-service MCP endpoint** (per-corpus MCP for customers' agents).
- **Drive / SharePoint OAuth connectors** (scaffolds exist, gated off; operator registers the
  OAuth apps).
- **Step-up confirmation on GPU start/stop** (security sweep L3 nicety) + short JWT expiry for
  the platform-admin account.
- **DEFAULT_MODEL_TIER env** — currently "balanced" while only "best" exists, so no Default
  badge renders in the wizard (cosmetic, one env var).

**Platform / infra:**
- **Prod env go-live**: `terraform apply envs/prod` + `deploy.sh prod <uat-proven sha>` once
  UAT has soaked.
- **B200 relaunch when capacity appears** (native NVFP4 + TP=1; carts are TP-agnostic by
  design so nothing re-onboards).
- **Research-bucket cleanup decision** (June cartridge-encoder/qasper/quality buckets + SGs —
  possibly IP-relevant, operator review).
- **AWS root-key rotation** (operator).

**ML / research:**
- **CRITICAL — multi-cart RoPE re-base in the connector scatter.** Serving MORE THAN ONE cart
  per request is position-broken: the 2nd+ carts sit at concatenated offsets while their stored
  keys keep build-time rotations — answers degenerate into repetition loops (found live on the
  Command A+ bench; single-cart answers are flawless; historical evidence only ever served one
  cart/request). Feasible because the cc codec stores keys DE-rotated: re-rotate at scatter to
  the placement position. Until it ships the fleet serves top-1 (`INFERENCE_TOPK=1`, set in the
  platform env + bench), which trades recall for correctness.
- **Chunked per-layer cart decode (restores 131k context on 160GB).** The load-path transient is
  a ~2GB/GPU whole-cart fp32 buffer; per-layer-group streaming shrinks it ~64x, letting
  GPU_MEM_UTIL 0.90 + 131k coexist with cart loads (today: 65536 ctx at 0.85 — see Decisions
  log). vLLM rejects the expandable_segments allocator with a KV connector, so this is the
  real lever.
- **recall@k A/B: doc descriptions on vs off** (DOC_DESCRIPTIONS_ENABLED flag exists; one env
  var per arm).
- **Streamed per-layer-group encode** (the 70B-class onboarding encode lever filed in the
  estimator notes; less urgent now that engine-side onboarding measured ~2.9s/doc).
- **Head-to-head Command A+ benchmark** (IN FLIGHT: harness built, awaiting 2×H100/B200
  capacity) → fills the estimator anchors + accuracy picture.

---

## 8. Decisions log

(newest first)

- **2026-09-03 — RESIDENT-QRC serve mode + dynamic top-k BUILT (behind flags, pending box validation).**
  Two additions on top of the QRC hybrid path, both OFF by default so nothing changes until proven:
  **(1) RESIDENT-QRC (`QRC_MODE=resident`)** — instead of re-prefilling docs 2..k's routed chunks as
  TEXT `context` every query (hybrid), load those chunks' KV as SPANS: the top-1 doc loads its full
  cart, each other doc loads only the token ranges its routed CHAR spans map to. The engine
  (`ml_service/vllm_inference.py`) re-tokenizes each spanned doc EXACTLY as onboarding did
  (`add_special_tokens=False`, truncated to `CAG_MAX_DOC_TOK`, `return_offsets_mapping=True`, LRU-cached
  per (cart_id, text-hash)) and maps each char span to a token range that fully covers the chars
  (`_spans_to_token_ranges`, a pure/tested fn). All loaded segments pack into ONE contiguous prefix
  (the connector contract is prefix-only), so NO text may sit between them — source attribution moved
  into the computed question turn (`_resident_user_turn`: "Above: the primary document, then excerpts
  from: <title B>; and <title C>." + the direct-answer instruction, titles control-plane-supplied). The
  engine's registry entry is the new DICT-mode list `{cart_id, src_start, src_len, dest}` (dest = the
  segment's cumulative offset in the packed prefix) — the wheel side (separate repo) consumes it.
  `QueryReq` gained `doc_spans` (per cart_id, CHAR spans into the doc's text), `doc_texts` (the served
  text so the engine can re-tokenize — it never reads storage), `doc_titles` (attribution). Guards 400:
  a span map for an id not in doc_ids, span-loading the top-1 doc, or a spanned doc with no text; an id
  whose spans all resolve empty is dropped from BOTH the prefix and the attribution (never a zero-token
  segment). Backend: `retrieval.route_chunk_spans` returns the SAME chunks `route_chunks_context`
  composes but as char spans (via `chunking.chunk_spans` indices), reusing the `chunk_vecs` cache;
  `served_texts_for`/`doc_titles_for` supply the engine's text+titles from the same cached index the
  spans index into. `total_p` still counts loaded KV as resident; computed = template + question only.
  **(2) Dynamic top-k (`RETRIEVAL_DYNAMIC_K=on`, `RETRIEVAL_DYNK_RATIO=0.6`)** — after RRF fusion keep
  doc i (beyond the top) only while its fused score >= ratio*top, capped at k, always >= 1. Scores are
  exposed cleanly via `rrf_fuse_scored` (the single fusion source; `rrf_fuse` slices its ids,
  `dynamic_k` reads its scores — no re-fusion). Bench: new `qrc_res` arm (all doc_ids + spans, no
  context) and `qrc_dyn` arm (doc set from `dynamic_k`, records per-question k, reports mean k);
  `phase_accuracy` now runs cart/qrc/qrc_res/rag_churn by default (qrc_dyn opt-in). Tests: engine
  char->token mapping over a fake tokenizer, `route_chunk_spans` selection parity + char-range validity
  + unknown-id KeyError, dynamic-k boundary/cap/>=1 math + retrieve() wiring, chat resident split
  payload shape (mocked ml_client). Full backend suite 297 passed / 4 skipped; bench selftest 0
  failures (extended to assert qrc_res spans exactly the non-top docs with no context, and qrc_dyn
  records k) — all GPU-free/network-free. `hybrid` stays the default serve mode until the resident
  span-load path is validated on the box.
- **2026-09-03 — QRC hybrid serving BUILT + VALIDATED: k=3 accuracy 8-9→13/15 at 128 gen tokens
  (14/15 at 256), onboarding cost unchanged, routing adds ~56ms/query.** Design: top-1 retrieved
  doc serves as the resident cart (proven 15/15 solo); docs 2..k contribute query-routed chunks
  (~256 tok/doc, bm25s+dense+RRF per doc — `backend/app/chunking.py`, byte-identical core for
  product AND bench) as real-token context on the same request (engine QueryReq gained `context`).
  Three iterations, each driven by reading the failing answers: **(1)** first run 11/15 — the
  'Documents:' framing made the model treat visible excerpts as its ONLY sources (it never
  consulted the resident cart) and it narrated deliberation until the token budget died;
  **(2)** reframed ('Above is the primary document… answer directly') + per-doc 'From "<title>":'
  headers → 13/15, tying RAG (13/15) with the primary doc still resident; **(3)** 256-token
  diagnostic: qrc 14/15 vs cart 9/15, rag 15/15 — the residual misses are deliberation truncation,
  NOT routing (relevant to the INFERENCE_MAX_TOKENS=96 prod default — consider raising).
  Perf defect caught by measuring: per-query chunk re-embedding cost ~8.4s/query (same class as
  the 2026-09-02 doc-level re-embed bug) → embed-once-per-doc vector caches (bench retriever +
  `_CorpusIndex.chunk_vecs`) → p50 55.6ms/query warm, one-time ~2-4s/doc first touch.
  Chunk descriptions (founder-approved OFF by default, `QRC_CHUNK_DESC`): measured 16.7 s/doc —
  6× the 2.72 s/doc onboarding headline — with 4/13 docs failing to parse at max_tokens=384, while
  13/15 routing held mostly on raw chunk text; machinery + sidecar stay (flip on when cheaper:
  bigger chunks or piggyback the doc-description call). `QRC_MODE=hybrid` is the default;
  `off` = legacy multi-cart. compare.py deliberately unchanged (demo surface, still multi-cart).
  Raw artifacts: `bench/results/accuracy_k3_qrc{3arm,_round2,_256tok}.json` (box + local).
- **2026-09-03 — Compression EXONERATED for the k=3 accuracy gap (codec A/B on the box):
  lossless carts fail composed exactly like 3-bit carts ⇒ composition interference confirmed,
  QRC stays the designed fix.** Method (`bench/codec_ab.sh`, transient systemd unit on the GPU
  box): force-rebuilt the same 13 bench carts at cc_safe (4-bit) and bf16 (lossless), same 15
  questions / retriever / engine as the baseline; restored default cc_aggr carts + clean env
  after. Results (cart substring-ok): cc_aggr 15/15 k=1, 8–9/15 k=3; cc_safe 15/15 k=1, 6/15
  k=3; **bf16 15/15 k=1, 8/15 k=3** — zero compression noise, same composed failure. Failure
  modes at bf16 k=3: the model slips into question-narration ("The user is asking…") and burns
  the 128-token budget deliberating across the 3 resident docs, plus one genuine cross-doc
  confabulation (same question that confabulated under cc_aggr). One "failure" is a judge
  artifact (case-sensitive substring: answered "railway", expected "Railway") — the strict
  count slightly understates every arm (RAG control wobbled 12–14/15 across runs the same way).
  Side-find: bf16 onboards at 1.17 s/doc vs 2.38–2.49 with the codec — encode is ~half the
  onboarding wall clock. Raw artifacts: `bench/results/accuracy_k{1,3}_{cc_safe,bf16}.json`.
- **2026-09-02/03 — Head-to-head benchmark run on the production stack (Command A+, 2×H100,
  production hybrid retriever in the loop for BOTH arms) — the concurrency advantage HOLDS, and
  the run flushed out four product-critical bugs before any customer saw them.** Method: 13 QASPER
  papers onboarded via the engine path (13 docs / 35.4s = 2.72 s/doc at ~2,350 tok/doc; 75.4 MB/doc
  in the cc store), then closed-loop sweeps at conc 1/8/32/64/128 × {cart, churned-RAG, hot-RAG} on
  the same engine, same retrieval (bm25s+dense+RRF top-k), bench_fleet methodology (p50s, warmups,
  Zipf sampling, fact spot-checks). Headline (k=3, churned-RAG = the honest production-RAG arm):
  cart serves ALL 384 requests at conc 128 at 1.63 qps while churned RAG collapses to 0.157 qps
  with 62 timeouts at a 900s budget — a ~10× throughput advantage at high concurrency, and 4.5×
  faster first-token at conc 1 (182ms vs 818ms). Both arms are KV-pool-limited on this box (218B
  weights leave ~10GB pool → ~11 concurrent k=3 requests computing; see the 65536-context entry);
  at k=1 the cart advantage is 1.6–2.3× with cart accuracy 15/15. Caveat recorded: hot-cache RAG
  wins raw qps on a 13-doc corpus because everything fits the prefix cache — an artifact that
  vanishes when corpus >> cache, which is the entire point of resident carts. Bugs found+fixed by
  the run: (1) prod retrieval re-embedded the whole corpus EVERY query (~9s/query — now built once
  per index), (2) cart loads OOM'd at GPU_MEM_UTIL 0.90 (degrade-to-blank-KV), (3) Command A+
  answers were its THINKING stream (template ignores enable_thinking; reasoning=False is the real
  knob), (4) multi-cart serving position-broken (next entry).
- **2026-09-03 — Multi-cart serving made positionally correct (wheel 0.6.1) — THREE stacked
  defects, plus two deploy-integrity fixes.** The bench (retrieval-in-the-loop ⇒ 3 carts/request,
  the first real answer-quality test of composition at scale) found multi-cart answers degenerating
  into repetition loops. Fixed in order of discovery: **(a) claim/scatter geometry mismatch**
  (pre-existing since the June composition oracle, which only proved equivalence to a reference
  sharing the flaw): the scheduler claims carts packed at EXACT token offsets but the scatter
  placed them block-rounded — cart 2+ landed in slots the engine never reads. Carts now pack
  end-to-end (scatter_tokens_into_paged start_token, mid-block starts). **(b) RoPE rebase**: stored
  keys carry build-time rotations; 2nd+ carts are now rigid-shift re-rotated to their placement
  offset (a group action — within-cart geometry preserved exactly). **(c) rotary convention**:
  Cohere family pairs even/odd indices (interleaved) where Llama half-splits; the codec's
  de/re-rotation round-trips either way (which is why single-cart never exposed it) but an applied
  offset rotation in the wrong convention corrupts keys. CARTRIDGE_ROPE_CONVENTION env, per-model,
  tests derived from the transformers formula INDEPENDENTLY of the wheel's own helpers. Deploy
  integrity: pip skips same-version wheel installs (a rebuilt-but-unbumped wheel silently never
  installed — bootstrap now --force-reinstall + version-bump discipline) and a site-packages debug
  patch survived a provision. Validated live: answer-in-third-cart correct; INFERENCE_TOPK
  restored to 3 fleet-wide. Residual (filed): composed carts are still SOLO-encoded — k=3 accuracy
  trails k=1 (verbose narration + one confabulation observed); QRC remains the designed answer for
  true multi-doc grounding.

- **2026-09-02 — Pricing simplified to TWO meters; onboarding is free.** Memory $0.03/doc/mo
  (was $0.02) + inference $1.30/1k queries; the $0.10/doc onboarding meter is gone. Why: one-time
  revenue becomes recurring (the +1¢/mo recovers onboarding cost in 1–2 months and passes the old
  one-time fee at ~10 months — customers who stay pay more, churners pay less, and retention is
  structural because onboarded docs ARE the switching cost); "free onboarding" removes the
  biggest trial objection vs RAG vendors' ingestion fees; and the story compresses to one line —
  pay for what your AI remembers and what you ask it. Guardrails paired with it (see Backlog):
  fair-use terms for thrash/re-sync, platform fee as the floor, change detection as the real
  re-onboarding fix. Payment provider decided: Stripe (metered Billing + Customer Portal + Tax;
  internal usage tables stay the source of truth). Billing remains DISABLED for now.

- **2026-09-02 — DECIDED + SHIPPED: onboarding goes THROUGH the vLLM engine (path A) — full smoke
  6/6 on the live box.** The connector gained its SAVE direction (wheel 0.5.0): a request flagged
  with `cartridge_build_cart_id` in `extra_args` never loads carts; at finish the scheduler returns
  delay-free-blocks and the worker harvests the full prompt KV out of the paged cache (same fused-
  layout math as scatter, both stride orders, chunked-prefill-proof), staging per-TP-rank shards that
  `engine_onboard.collect_rank_shards` merges back to FULL-head carts (TP-agnostic — a cart built at
  TP=2 serves at TP=1 on a future B200). `vllm_inference` gained `/onboard_cag` (schema + progress
  reporting identical to the transformers path; same store write, same cc codec) and `app.py`
  proxies to it behind `ONBOARD_VIA_ENGINE=1` — control-plane contract unchanged, transformers path
  intact for models it can load. Wins held: onboarding measured at ~2.9s for a 646-token doc on the
  serving box (no second box, no extra spend), storage identical (same codec/store), serve hot path
  untouched (every new hook early-returns for non-build requests), `/describe` works against
  engine-built carts. Validated live: smoke 6/6 — engine_ready, compat_check, engine onboard,
  grounded query (fact recalled from the cart), measured metrics shape, describe. Also fixed en
  route: stale-wheel accumulation in the provision staging dirs silently reinstalling old wheels
  (both dirs now cleaned), and smoke's corpus path being MSYS-mangled on Git Bash.
  Original finding (kept for the record): serving PASSES but the transformers onboarding path
  cannot run this model — With the venv-PATH (ninja) and /data contract-dir
  fixes, the 2×H100 box serves Command A+ W4A4: engine_ready over HTTPS, weights load from the FS
  seed in ~17s (61.5 GiB/GPU, Marlin FP4 on H100 — native NVFP4 arrives with a B200), compat_check
  passes, token auth enforced. But `/onboard_cag` cannot build carts: Command A+ is a VISION-language
  model (`Cohere2VisionConfig`) and the onboarding worker's transformers `AutoModelForCausalLM`
  forward pass rejects it — the exact GATE 2 FAIL the validation run filed (transformers can't
  forward-pass this NVFP4 MoE VLM; and dequantized bf16 needs ~224GB, more than the box's 160GB).
  Serving and onboarding are now on DIFFERENT stacks for this model. Two real paths:
  **(A) onboard through the vLLM engine** — implement the save direction in `CartridgeKVConnector`
  (`save_kv_layer` is a deliberate read-only stub today) + an engine onboard endpoint; KV is
  layout-exact by construction, one box, no new spend — but it's core-IP wheel engineering.
  **(B) decoupled spot onboarding at higher precision** — `onboard_corpus.py` was built for exactly
  this shape (spot-safe, S3 store, serve elsewhere): fix the VLM load (language tower via
  `AutoModelForImageTextToText`), run onboarding on a rented big-VRAM spot box (bf16 ≈ 224GB → 8×A100
  class, hours not months), and gate on the Phase 5.3 cross-precision equivalence A/B before mapping
  the refs in `CARTRIDGE_MODEL_EQUIV`. Ships sooner, known tools; adds spot cost per onboarding wave
  + an equivalence caveat. Recommendation: A for the product (the serving engine is the only thing
  that can run this model class end to end); B only if carts are needed this week.
- **2026-09-02 — E12 shipped: GPU start/stop/status from the Platform Admin tab, with unattended
  relaunch.** Backend wraps the official Lambda REST API (`lambda_cloud.py` + `cloudflare_dns.py` +
  `/platform-admin/gpu/*`); Stop = terminate ($0/hr, all durable tiers survive), Start = launch with
  a cloud-init `user_data` that runs `self-provision.sh` off the persistent FS (published by
  provision.sh), so a fresh box provisions itself — the control plane never SSHes. Launch-day
  hardening folded into the same pass: **(a)** Lambda's default account firewall is 22-only — a fully
  provisioned box was silently unreachable on 80/443; `MANAGE_FIREWALL` now defaults ON and the
  backend Start ensures the rules. **(b)** `systemctl reload caddy` on first provision silently kept
  serving the apt-default :80 config (no TLS ever issued); bootstrap now restarts + verifies :443.
  **(c)** the engram-cartridge `torch<2.12` ceiling downgraded torch under vLLM 0.28 (torchvision ABI
  crash at the service's lazy import, missed by the old `import vllm` check) — wheel 0.4.2 drops the
  ceiling, bootstrap installs vLLM LAST and the stack check now exercises the lazy path.
  **(d)** `VLLM_WORKER_MULTIPROC_METHOD=spawn` for TP>1; `VLLM_TP` autodetected from GPU count so one
  bundle serves the 1×B200 and 2×H100 shapes. **(e)** seed-out restarts use `--no-block` (a oneshot
  that waits for engine_ready was blocking provisioning's SSH session). Also fixed: the uat/prod env
  roots never declared `cartridge_store_bucket_override`, so the tfvars value was silently ignored
  (Terraform drops undeclared vars) — now passed through. First `gpu_2x_h100_sxm5` box drew a bad
  host (GPU fabric registration stuck "In Progress" → CUDA Error 802, links themselves healthy,
  survived reboot) — terminated and relaunched, which is exactly the recycle flow the scripts + the
  new Start button automate.
- **2026-09-02 — Lambda pivot applied: UAT is LIVE at uat-app.engramdynamics.org.** The $7,500
  Lambda credits pivot executed end-to-end this session: `serving-lambda/aws-support` applied (cart
  bucket + bucket-scoped IAM key), `platform-aws/common` + `envs/uat` applied, images `56996e5`
  deployed, login verified with the seeded operator admin (password in Secrets Manager
  `uat-engram/BOOTSTRAP_ADMIN_PASSWORD`). UAT points at the Lambda serving unit via overrides
  (`read_serving_state=false`): gpu.engramdynamics.org (serve :8002) / gpu-onboard.engramdynamics.org
  (onboard :8001), Caddy auto-HTTPS fronting 127.0.0.1-bound services, Cloudflare A records
  auto-upserted by launch.sh. Serving = Command A+ W4A4 (`best` tier, 131072 ctx) on `2x_h100_sxm5`
  (B200 preferred when capacity appears). Three-tier storage: ephemeral SSD (hot) / persistent
  `engram-fs` (weights seed, survives terminate) / S3 (durable carts). Prod env not yet applied.
- **2026-09-02 — E8 built: two-env platform Terraform (`infra/platform-aws/`).** Four stacks
  (common → uat → prod + the shared module), all validate clean, nothing applied: one public ALB +
  ACM cert (Cloudflare records output for manual add), per-env Fargate services / RDS / S3 /
  Secrets, serving-unit values via remote state, `build_push.sh` + `deploy.sh <env>` giving the
  image → UAT → prod flow. Backend image gains libreoffice-writer (.doc) + a pinned fastembed
  cache dir. ~$75–95/mo both envs. **Operator sequence to go live:** apply `common` → add ACM
  CNAMEs at Cloudflare → `build_push.sh` → apply `envs/uat` + `deploy.sh uat` → add the 4 host
  CNAMEs → verify at uat-app → `deploy.sh prod`. Prereqs also on the operator: SES production
  access; the GPU serving unit applied (envs read its state).
- **2026-09-02 — UAT is an in-account STAGE, not the sub-account; shared-store GC made
  environment-safe (shipped).** Decision: run UAT as a parallel `uat-` stack inside the prod
  account/VPC (own ECS services, own DB, own secrets/buckets, `uat-app.`/`uat-api.` domains),
  sharing the single GPU serving unit — trading hard account-level blast-radius for zero
  cross-account networking (the GPU URLs are private-VPC IPs) and ~$30–80/mo. The org's "Test
  Account" (651343918364) is the graduation path when the team grows. Consequences: (a) E8's
  platform Terraform becomes ONE module instantiated twice (`env=prod|uat`); (b) UAT cannot
  rehearse serving-side changes (different model/config) — those use ephemeral spot serving
  units (`use_spot=true`, hours not months); (c) heavy UAT onboards contend with prod latency
  (calendar rule at beta scale); (d) UAT sits inside the prod ML-plane trust boundary (shared
  ML_AUTH_TOKEN). **Prerequisite shipped:** the GC sweep is now attribution-safe under a shared
  cart store — only ids whose tenant prefix exists in the local DB are sweepable; foreign-
  environment carts and legacy un-namespaced ids are skipped and counted (`n_foreign_skipped`),
  so one environment's GC can never delete the other's carts. Suite 185 passed / 4 skipped.
- **2026-09-02 — Retrieval upgrade: LLM doc descriptions (flag, default ON) + hybrid
  retriever.** Retrieval is the measured accuracy ceiling, so two investments: (1) at
  onboarding, the inference service generates a one-sentence description per doc **against the
  resident cart** (short prefill + ≤60-token decode, batched — est. +15–20% onboarding cost) via
  a new `POST /describe`; stored on `Document.description` (Alembic `0011`), shown in the UI,
  folded into the wizard estimate, and **gated by `DOC_DESCRIPTIONS_ENABLED` (default on) so an
  off-run gives a clean A/B baseline**; best-effort — can never fail an onboarding. (2) the
  control-plane retriever is now the industry-standard hybrid: `bm25s` + `fastembed` dense
  (ONNX, torch-free) fused with RRF; descriptions indexed as retrieval metadata only; dense
  degrades to lexical on failure; `fused` (GPU) unchanged. **GPU-gated follow-ups:** wire
  `/describe` into the serving smoke test, and run the actual recall@k A/B (descriptions on vs
  off) once the box is live — the flag exists so that measurement is one env var.
- **2026-09-01 — Serving decision: Llama-3.3-70B (FP8) on one g6e.12xlarge, packaged as a
  swappable serving unit.** Command A+ has no runnable path on available hardware for now, so the
  beta serves the validated fallback: `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic`, TP=4 on the
  4× L40S box (the exact model+instance combo already measured in the evidence: fact-grid 16/24
  with cart==RAG==bf16, 17.5 cart-QPS at conc-24; codec supports Llama-3 rope scaling). **The
  swappable-unit contract:** a serving unit is anything that emits `ML_SERVICE_URL`,
  `INFERENCE_SERVICE_URL`, `ML_AUTH_TOKEN`, and `MODEL_REGISTRY_JSON` — the only four values the
  control plane consumes — so pivoting model/instance/cloud is a new unit with the same outputs,
  zero product-code change. AWS implementation lives in `infra/serving-aws/` (Terraform +
  SSM provisioning + smoke test); nothing is launched until the operator runs `terraform apply`
  (GPU spend is a human decision). Command A+ stays the upgrade path via the registry (new tier →
  onboard new corpora against it; existing corpora stay pinned to their model_ref).
  **✅ Unit built + verified (same day):** `infra/serving-aws/` — Terraform (one g6e.12xlarge,
  SSM-only access, S3 cart bucket, DLAMI pin verified live, spot toggle, account guard),
  `provision.sh` (wheel from the sibling repo → S3 bundle → SSM bootstrap → two systemd
  services), `smoke.sh` (engine health → compat_check → onboard a test doc → grounded query).
  `terraform validate` clean; the emitted `MODEL_REGISTRY_JSON` parsed through the real
  `serving.py` loader. Operator runbook in the module README. **Next human step:**
  `terraform apply` → `provision.sh` → `smoke.sh` → paste the four env values.
- **2026-09-01 — Local end-to-end run-up completed (closes Phase 0).** Booted the real stack
  (control plane on SQLite + Next.js dev; no GPU plane) and walked it in a browser as three
  users: waitlist request → platform-admin approve → `#token` fragment accept-invite → the full
  5-step wizard (upload with live parse badges: txt/html Ready, blank-pdf Failed with reason;
  model tiers; review/estimate; onboard → the 409 "queued for a model" state) → step resume →
  team invites → tenant Admin Dashboard → Platform Admin console, with role/platform-admin nav
  gates verified across users. **Three bugs found live and fixed (`feb8dbe`):** the one-time
  invite link vanished on approval (row-local state unmounted by the list refresh — lifted to
  the page); configured Google creds lit up the unimplemented Drive connector (availability now
  = module `IMPLEMENTED` flag AND creds); and all-placeholder model tiers dead-ended the wizard
  at step 3 (placeholders are now selectable — the serving gate stays at onboard's 409 — so
  setup completes into the queued state). **Not walked locally:** chat (requires the GPU plane;
  covered by the stream contract tests until the serving box exists). Suite 159 passed /
  4 skipped; tsc clean.
- **2026-09-01 — Production-readiness review + hardening pass shipped.** Three read-only Opus
  reviews (security/isolation, backend correctness, frontend contracts) over the whole codebase.
  **Fundamentals verified clean:** tenant isolation on every route, platform-admin gating, cart
  namespacing completeness, migrations vs models, storage sidecar, aggregation math, API
  contracts, no committed secrets. **17 real findings fixed** across three commits
  (`b7b92a7`, `27cc8eb`, `13e557c`+merge `50ce5d3`), the big ones: invite/reset links were
  entirely broken (`#token` fragment vs `?token` query — a cross-agent seam); every tenant was
  billed the deployment-wide query volume (fixed properly: `Measurement.tenant_id`, Alembic
  `0009`, stamped at all corpus-scoped record sites); deactivated users' JWTs kept working
  (`is_active` now enforced); prod `validate()` now refuses `EMAIL_BACKEND=none` (link leakage)
  and requires `ML_AUTH_TOKEN`; cross-tenant invite could overwrite an existing account (now
  bound to the invite's tenant); per-doc status machine redesigned (parse_status = upload-time
  fact; cancel/fail/empty-doc paths no longer stick or lie); zip-bomb/page-count parse guards;
  chat stream always terminates + stops on client disconnect; frontend: global 401 → login
  redirect, stream abort on unmount/corpus-switch, chat history capped, types aligned.
  **Deferred, documented (LOW):** `ALLOWED_HOSTS` fail-fast at public launch, per-account login
  backoff, streaming upload buffering, SSE wall-clock caps, non-stream adaptive escalation
  (deliberately stream-only). Suite: **159 passed / 4 skipped** (+21 tests); frontend build clean.
- **2026-08-31 — Shipped E10 + E11 dashboards.** Parallel agents (backend main tree / frontend
  worktree). Backend: `/auth/me` now returns `platform_admin`; E10 tenant `GET /admin/usage` +
  `/admin/billing`, E11 cross-tenant `GET /platform-admin/tenants` + `/platform-admin/usage`
  (cost-per-tenant + fleet totals); a single pricing source (`app/pricing.py`) + shared aggregation
  (`app/usage.py`); `Tenant.plan`/`status` + Alembic `0008`. Frontend: the tenant **Admin
  Dashboard** (`/admin`, role-gated — usage stat cards + recharts time series + per-corpus table +
  billing shell) and the operator **Platform Admin** console (`/platform-admin`,
  platform_admin-gated — tenants table, cost-per-tenant recharts bar chart, and the waitlist
  approve/deny UI on E1's endpoints). Charts via recharts (per the OSS preference). Merged to
  `main` (`6d2d8e2`); backend **138 passed / 4 skipped**; frontend build clean; **no integration
  seam** (contract aligned). Note: `Measurement` has no tenant_id (deployment-global serving twin),
  so query counts are a deployment-level signal; the tenant-scoped facts (corpora/docs/storage)
  are strictly `tenant_id`-filtered.
- **2026-08-31 — Shipped E1: auth self-serve + roles/invites.** Parallel agents (backend main
  tree / frontend worktree). Invite-only beta (waitlist `request-access` → platform-admin approve
  → `accept-invite`), forgot/reset-password, **two authz tiers** (tenant `admin`/`member` via
  `require_tenant_admin` + a `platform_admin` superuser via `require_platform_admin`), and a
  role-gated **Team** management surface (members/roles/invites, self-demote + last-admin guards).
  Tokens use `secrets` + SHA-256 + `hmac.compare_digest`; email gated (`EMAIL_BACKEND=none` returns
  the link in-response so every flow works with no provider). Models + Alembic `0007`. One
  integration seam fixed: aligned the frontend team page to the backend `{members, invites}` /
  `LinkResp` shapes. Merged to `main` (`c3f7907`); backend suite **128 passed / 4 skipped**;
  frontend build clean. Follow-ups: `/auth/me` doesn't yet return `platform_admin` (add when E11
  needs the platform-admin nav); email needs SES config to actually send (links returned meanwhile).
- **2026-08-31 — Shipped E2 remainder: document parsing + connector framework.** Parallel
  agents again (backend in the main tree / frontend in a worktree). Backend: `parsing.py`
  (pypdf/python-docx/beautifulsoup4 — control plane stays torch-free) wired into upload → an
  extracted-text sidecar so onboarding/retrieval consume real text; `Document.parse_error`
  (Alembic `0006`); a config-driven connector registry + `GET /connectors` with Drive/SharePoint
  gated `available:false` until OAuth creds are set (OAuth itself not built — needs app
  registration). Frontend: the wizard documents step accepts PDF/DOCX/HTML, shows per-doc
  parse-status badges + errors (self-stopping 2s poll), and renders the gated Connect
  Drive/SharePoint buttons. Merged to `main` (`5d9078a`); backend suite **113 passed / 4 skipped**;
  frontend build clean; no integration seam this time.
- **2026-08-31 — Shipped E6 + E3 + onboarding wizard frontend (parallel build).** Three chunks
  built by parallel Opus agents (E6 backend in the main tree; the frontend in an isolated git
  worktree, so their builds/tests couldn't race) and integrated: **E6 per-tenant cart namespacing**
  (cart ids now `<tenant>__<slug>` via `cart_id_for`; cross-tenant slug sharing removed,
  delete/GC tenant-scoped, no migration needed); **E3 conversational chat UI** as the primary
  surface (streamed answers via a new `POST /corpora/{id}/chat/stream` SSE endpoint, markdown +
  citations + localStorage history, Compare/Scale/Costs retired from nav); and the **resumable
  onboarding wizard UI** (consumes `/models` + the onboarding endpoints; the 409 `no_serving_engine`
  gate handled gracefully). Merged to `main` (`7e8d1bd`); backend suite **97 passed / 4 skipped**
  (one integration fix: the frontend's stream tests re-pointed to E6's namespaced ids); frontend
  build clean. Pushed.
- **2026-08-31 — Onboarding-flow backend built + tested.** The [Q6] resumable 5-step wizard
  backend is in: Corpus `onboarding_step`/`model_tier`/`model_ref` + per-doc parse/onboard status
  (Alembic `0005`), `PATCH/GET /corpora/{id}/onboarding`, `GET /corpora/{id}/estimate`,
  `POST /corpora/{id}/onboard` — the last gated on an *available* serving tier (returns 409
  `no_serving_engine` while tiers are placeholders, so the whole flow is testable with no GPU).
  Reuses the existing job/progress/cancel path via an extracted `dispatch_training`. Full backend
  suite green (79 passed, 4 pre-existing skips). Remaining for the flow: frontend wizard UI +
  document parsing.
- **2026-08-31 — Two admin dashboards added to scope (E10, E11).** (1) **Tenant "Admin
  Dashboard"** (customer-facing, shown as just "Admin Dashboard", tenant-admin-role only): usage,
  the tenant's own costs/bill, team & access (who has access, invite/remove, roles),
  payment/billing, plus MCP-token, data/audit, and plan-limit views. (2) **Platform Admin
  console** (operator-only — the founder; a new platform-superadmin authz tier above tenant
  roles): every tenant, users added per tenant, and cost per tenant, plus the invite/waitlist
  approval queue and per-tenant limit/suspend controls. **Two authz tiers:** `tenant_admin`
  (scoped) and `platform_admin` (cross-tenant, tightly gated + audited).
- **2026-08-31 — Serving backend left interchangeable; no hardware/cloud committed.** The
  cloud/box decision is deferred (pending a cloud-credit outcome), so the serving target is a
  swappable interface, not a baked-in choice. Added `backend/app/serving.py` as the single
  control-plane↔GPU-plane boundary: the control plane reaches serving only over HTTP
  (`ML_SERVICE_URL` / `INFERENCE_SERVICE_URL`) and identifies models by `model_ref` — GPU type,
  cloud, and precision are never modeled in product code. Model choice is a **config-driven tier
  registry** (`MODEL_REGISTRY_JSON`), shipping **placeholder** tiers (Fast/Balanced/Best,
  disabled) so the onboarding menu renders "coming soon" until a box is chosen; `GET /models`
  serves the menu. The prior model intent (Command A+, with Llama-3.3-70B / Qwen2.5-72B text-only
  alternatives) stands as the *candidate* to wire once hardware lands — now a registry entry +
  endpoint config, not code. **Supersedes the hardware half** of the 2026-08-30 entry below; the
  model rationale there still holds.
- **2026-08-30 — Beta model + serving box locked: Command A+ (4-bit) on one g6e.12xlarge.**
  Enabled beta model = **Cohere Command A+** (`CohereLabs/command-a-plus-05-2026`, Apache 2.0,
  218B / 25B-active MoE, native citation/grounding spans — matches the chat citation feature),
  chosen for best faithful grounded fact-QA (reasoned from the injection/compression mechanics +
  open research, not repo benchmarks: CAG makes retrieval free, so comprehension + faithfulness
  is the bottleneck, which favors a grounding-tuned model over low-active MoEs that ace recall
  but collapse on synthesis). Served **4-bit** so a frontier model fits the box — FP8 is ~218 GB;
  the lossless QAD 4-bit is ~120 GB. **Box = one `g6e.12xlarge` (4× L40S, 192 GB) on AWS**: stays
  in-VPC next to the S3 cart store, more VRAM than 2× H100 (160 GB), and AWS has no small H100
  shape (H100 = 8× `p5.48xlarge`, ~$40k/mo). **Fallback if the Ada 4-bit path underperforms:
  Llama-3.3-70B FP8** (~71 GB, no verification needed). **Quota verified in-account (2026-08-30):**
  On-Demand G&VT 64 vCPU / Spot 48, nothing running → one box available now; a **second box needs
  a Service Quotas bump to 96+ vCPU** (user pursuing). **Open task:** verify Command A+ 4-bit
  serves cleanly on L40S/Ada (native W4A4 is Hopper/Blackwell; on Ada expect 4-bit-weight/W4A16)
  via a grounded-QA A/B vs the Llama-3.3-70B fallback.
- **2026-08-30 — [Q6 drill-down] Onboarding flow + model selection locked.** Resumable
  5-step per-corpus wizard: (1) name, (2) add documents (upload / Google Drive / SharePoint),
  (3) choose model, (4) review + estimate (doc count, detected types, est. onboarding
  time/cost) then confirm, (5) onboard with live per-doc progress. The corpus is saved at step
  1 and each step persists, so exiting returns to the current step; onboarding runs server-side
  (Job) so it survives a closed tab. **Model menu = quality/speed tiers (Fast/Balanced/Best) as
  the primary choice with the backing model name shown as detail**; entries are placeholders
  until the model list is filled. **Serving = a single active model for the closed beta**
  (selection UI present, one model enabled, one GPU engine); multi-model routing deferred.
  Corpus stays pinned to its model (`model_ref`); switching = re-onboard.
- **2026-08-30 — [Q4] Chat-only MVP.** The product surface is just the Claude/Gemini-style
  conversational chat. Compare (CAG vs RAG), Scale-test, and Costs are removed from the product
  and kept only as internal/sales tools.
- **2026-08-30 — [Q6] Model choice is per-corpus, selected inside a resumable onboarding
  flow.** Not one launch model — the user picks the model at onboarding time as a step in a
  click-in "onboarding flow" that shows live progress and can be exited and resumed. The
  selectable model list is a placeholder for now (designed in the Q6 drill-down). Grain fits the
  IP: carts are model-bound (`model_ref`), so a corpus is pinned to its onboarding model
  (changing it = full re-onboard) and serving must route to an engine running that model.
- **2026-08-30 — [Q7] Invites: waitlist + manual admin approval.** Public "request access"
  form; an admin view approves each account. Simplest gate for the invite-only beta.
- **2026-08-30 — Migration executed (Phase 0 landing).** Flattened
  `Engram-Smart-CAG/platform/*` into this repo's root (dropped the redundant `platform/`
  wrapper, which also removes the stdlib-`platform` module shadowing hazard). `cartridges`
  is now consumed as the `engram-cartridge[s3,build]>=0.4.1` pip dependency (sys.path shims
  removed from `ml_service`; requirements + Dockerfiles + compose repointed). `.env.local`
  untracked and local secrets rotated; the real Google OAuth creds were preserved from the
  old gitignored `.env` into this repo's gitignored `.env`. `platform/` removed from
  `Engram-Smart-CAG` on branch `chore/remove-platform`. Bootstrap commit here on `main`
  (`c176092`). Remaining for a full local run-up: `pip install -e ../Engram-Smart-CAG`
  (until the package is published) + service deps + `npm install`.
- **2026-08-30 — [Q5] Self-service depth: invite-only beta first.** Gate signups behind
  invites/waitlist while we validate. Usage is metered from day one; Stripe billing + open
  self-serve come later. Keep `ALLOW_REGISTRATION=false` + an invite/waitlist gate on signup.
- **2026-08-30 — [Q3] Ingestion for MVP: upload + PDF/DOCX parsing + SharePoint + Google
  Drive connectors.** Bigger ingestion scope than the default — MVP includes two live external
  source connectors (SharePoint, Google Drive) on top of file upload with real document parsing.
  Wire the `Connector` ABC into the upload/ingest pipeline and build the two connectors.
- **2026-08-30 — [Q2] Auth: built-in email + Google.** Reuse the platform's JWT auth + the
  already-provisioned Google OAuth client for self-serve signup. Enterprise SSO deferred.
- **2026-08-30 — [Q1] Consume IP via pip dependency.** This repo installs `engram-cartridge`
  as a versioned pip package; the GPU plane imports it. Requires publishing/pinning the package
  from `Engram-Smart-CAG/cartridges/` (add a build/release step for it).

---

## 9. Reference file map (in `Engram-Smart-CAG`)

- Control plane routes: `platform/backend/app/routers/{auth,corpora,jobs,chat,compare,economics,metrics,audit}.py`
- Control↔GPU seam: `platform/backend/app/ml_client.py`
- GPU onboard + HF inference: `platform/ml_service/app.py`
- vLLM cart serving: `platform/ml_service/vllm_inference.py`
- Cartridge IP: `cartridges/serve/{serve_carts,vllm_cartridge_connector,compat_check}.py`, `cartridges/cart_store.py`, `cartridges/model_binding.py`
- MCP: `platform/mcp_server/server.py`
- Frontend: `platform/frontend/app/**`, `platform/frontend/components/{AppShell,ui}.tsx`
- Infra: `infra/terraform/**`; landing/DNS/auth: `engram-dynamics-landing/infra/**`
- Product/spec docs: `engram-dynamics-landing/docs/{VISION,DATA_LIFECYCLE,INTEGRATION,COST_COMPARISON}.md`
</content>
</invoke>
