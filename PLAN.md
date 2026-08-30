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

### E1 — Auth & self-service sign-on  → **built-in email + Google (locked)**
- **Exists:** email/password → HS256 JWT (PBKDF2, 600k iters), multi-tenant, Google OAuth
  (real client already provisioned), pluggable OIDC backend (`AUTH_BACKEND=oidc`) for later
  enterprise SSO. Registration gate (`ALLOW_REGISTRATION`).
  Ref: `backend/app/routers/auth.py`, `backend/app/security.py`, `backend/app/deps.py`.
- **Add:** invite/waitlist-gated signup (per the invite-only-beta decision — keep public
  registration off and admit via invite codes), email verification + password reset, "Continue
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
    Google OAuth client we already have can seed Drive access scopes.
  - **Document parsing** — today plain `.txt`/`.md` only. Add PDF/DOCX/HTML extraction
    (e.g. `unstructured`), chunk/normalize before onboarding.
  - **Per-document CRUD** — add per-doc delete/replace endpoints (only whole-corpus delete
    exists today).
  - **Change detection / re-onboard** — the mechanics exist (idempotent re-onboard under the
    same slug + auto `inference_invalidate`; a `doc_version` metadata hook on the store), but the
    *decision* of which docs are stale is not automated. Add content-hash/mtime staleness so an
    edited doc re-caches automatically.
  - **Lifecycle UI** — status per doc (queued/onboarding/ready/failed/stale), sizes, token
    counts, last-onboarded, re-onboard button, delete with the audit trail surfaced.

### E3 — Chat experience (Google/Anthropic-grade)  **[Q4 in doc, recommended default]**
- **Exists:** a Gemini-style collapsible app shell (`components/AppShell.tsx`), a streaming
  **Compare** view (Smart CAG vs RAG side-by-side, SSE) that is really a *sales* surface, and
  a `POST /corpora/{id}/chat` API + `api.chat()` that aren't yet a conversational page.
  Ref: `frontend/app/corpus/[id]/(workspace)/chat`, `backend/app/routers/chat.py`.
- **Add:** a clean, single-answer **conversational chat** page that looks and feels like
  Claude/Gemini — message threads, streaming tokens, markdown + code rendering, citations back
  to source documents (the cart path grounds answers), conversation history, corpus/model
  picker, stop/regenerate, copy. Replace hand-vendored `components/ui.tsx` primitives with a
  real design system (shadcn/ui — already the noted intended swap).
- **Recommendation:** make conversational chat the primary product surface; relocate
  Compare / Scale-test / Costs into an "Insights" or admin area (they're great for sales and
  for showing the customer their own savings, but they aren't the daily driver).

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
- **Add:** productionize model choice (Qwen3-8B validated; 30B is the prod target — 30B
  *accuracy* head-to-head is the one open research gate), GPU capacity/autoscaling, the
  multi-tenant multiplexing budget (many tenants' carts hot on shared GPUs — the real moat),
  warm/cold tiering, and health/observability. **[Q1]** decides how `cartridges/` is pulled in.

### E6 — Multi-tenancy hardening (correctness, not optional)
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

### E8 — Infra, deploy, domains
- **Exists (reuse):** AWS account `808379776072`, `us-east-1`, profile `Engram-Dynamics`.
  Terraform for ECS (backend/frontend/worker) + GPU EC2 (`g6e`, L40S) + RDS Postgres 16 +
  ElastiCache Redis + S3 + Secrets Manager + Budgets, currently mothballed to save spend.
  Domain `engramdynamics.org` at Cloudflare; landing site live on S3+CloudFront; a live
  invite-only Cognito pool (`us-east-1_jtvbJ65Qe`) for members.
  Ref: `Engram-Smart-CAG/infra/terraform/`, `engram-dynamics-landing/infra/`.
- **Add:** `app.engramdynamics.org` (frontend) + `api.engramdynamics.org` (backend) DNS at
  Cloudflare (the `app.`/`api.` scheme is already assumed in `.env.aws.example`), bring the
  mothballed stack up, CI/CD (CodeBuild→ECR→ECS exists), secrets via Secrets Manager.

### E9 — Security & compliance
- Rotate + untrack the committed dev secrets in `Engram-Smart-CAG/platform/.env.local`.
- Tenant isolation review (E6), per-doc auth, MCP token scoping, audit receipts (exist),
  data-deletion guarantees (invalidate→offboard→audit exist), S3 recycle-bin/versioning
  (described in `DATA_LIFECYCLE.md` but the `deploy/byoc/` CFN is not in the working tree —
  verify before relying on the 30-day recovery guarantee).

---

## 5. Reuse-vs-build summary

**Reuse largely as-is:** control-plane skeleton, auth (JWT + Google + OIDC), tenancy model,
corpus/job orchestration with progress + cancel, the full delete/audit/GC lifecycle, the
GPU-plane onboarding + vLLM cartridge serving path, Alembic migrations, Docker/compose, the
app shell + upload wizard.

**Build new or significantly extend:** conversational chat UI (E3), external ingestion
connectors + document parsing + per-doc CRUD + change detection (E2), remote/self-service
MCP (E4), per-tenant cart namespacing (E6), metering/billing (E7), production auth self-serve
flows (E1), bring-up + DNS (E8).

**Known gaps to close:** `RETRIEVAL_BACKEND=pgvector` raises `NotImplementedError` (only
bm25 + fused work); no document parsing beyond text; connectors unwired; no per-doc delete;
MCP is one stdio tool; cross-tenant slug sharing; the S3 recycle-bin CFN is missing from the tree.

---

## 6. Phased roadmap

- **Phase 0 — Land the codebase.** Resolve **[Q1]**; migrate `platform/` into this repo;
  get it running locally (control plane on SQLite/filesystem, GPU plane optional); rotate the
  leaked secrets. Exit: register → upload `.txt` → onboard → chat works locally.
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
[Q3] upload + PDF/DOCX parsing + SharePoint + Google Drive · [Q5] invite-only beta first.

**Still open:**
- **[Q4] Chat surface shape.** Confirm the recommendation: conversational chat is primary,
  Compare/Scale/Costs move to an Insights/admin area (vs. keeping the sales-demo tabs prominent).
- **[Q6] Model + serving path for launch.** Start on Qwen3-8B (validated) vs push to the 30B
  prod target (needs the open 30B accuracy head-to-head); GPU sizing / monthly cost tolerance
  for the always-on serving box.
- **[Q7] Invite mechanism.** Waitlist + manual admin approval, or shareable invite codes, or
  both — and who administers it during the beta.

---

## 8. Decisions log

(newest first)

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
