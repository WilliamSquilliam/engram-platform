"""Pydantic request/response shapes."""
import datetime
import re

from pydantic import BaseModel, Field, field_validator

# Pragmatic email shape check (something@something.tld, no whitespace) — enough to
# reject garbage without pulling in the email-validator dependency.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class RegisterReq(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(min_length=8, max_length=128)
    tenant_name: str = Field(min_length=1, max_length=120)

    @field_validator("email")
    @classmethod
    def _valid_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EMAIL_RE.match(v):
            raise ValueError("invalid email address")
        return v


class TokenResp(BaseModel):
    access_token: str
    token_type: str = "bearer"


# --- E1: self-serve auth + roles/invites + waitlist ------------------------

def _normalize_email(v: str) -> str:
    v = v.strip().lower()
    if not _EMAIL_RE.match(v):
        raise ValueError("invalid email address")
    return v


class RequestAccessReq(BaseModel):
    """Public 'request access' (invite-only beta waitlist)."""
    email: str = Field(max_length=254)
    name: str = Field(min_length=1, max_length=120)
    tenant_name: str = Field(min_length=1, max_length=120)
    reason: str | None = Field(default=None, max_length=2000)

    @field_validator("email")
    @classmethod
    def _valid_email(cls, v: str) -> str:
        return _normalize_email(v)


class AcceptInviteReq(BaseModel):
    """Redeem an invite/approval link: set a password and activate the account.
    `name` is required — it's the member's display name across the workspace."""
    token: str = Field(min_length=1, max_length=512)
    password: str = Field(min_length=8, max_length=128)
    name: str = Field(min_length=1, max_length=120)


class InviteInfoReq(BaseModel):
    """Look up who an invite is for (the accept page's confirmation header)."""
    token: str = Field(min_length=1, max_length=512)


class InviteInfoResp(BaseModel):
    email: str
    workspace: str


class ForgotPasswordReq(BaseModel):
    email: str = Field(max_length=254)

    @field_validator("email")
    @classmethod
    def _valid_email(cls, v: str) -> str:
        return _normalize_email(v)


class ResetPasswordReq(BaseModel):
    token: str = Field(min_length=1, max_length=512)
    password: str = Field(min_length=8, max_length=128)


class InviteCreateReq(BaseModel):
    """Tenant-admin invites a teammate into their workspace."""
    email: str = Field(max_length=254)
    role: str = Field(default="member", pattern="^(admin|member)$")

    @field_validator("email")
    @classmethod
    def _valid_email(cls, v: str) -> str:
        return _normalize_email(v)


class RoleUpdateReq(BaseModel):
    role: str = Field(pattern="^(admin|member)$")


class MemberResp(BaseModel):
    id: str
    email: str
    name: str | None = None
    role: str
    is_active: bool = True


class PendingInviteResp(BaseModel):
    id: str
    email: str
    role: str
    created_at: datetime.datetime


class MembersResp(BaseModel):
    """GET /admin/members — active members + still-pending invites for the tenant."""
    members: list[MemberResp] = []
    invites: list[PendingInviteResp] = []


class AccessRequestResp(BaseModel):
    id: str
    email: str
    name: str
    tenant_name: str
    reason: str | None = None
    status: str
    created_at: datetime.datetime


class LinkResp(BaseModel):
    """Generic response for the gated-email flows. When EMAIL_BACKEND=none the link is
    included so the flow is completable without an email provider; when a real backend
    is configured the link is emailed and these fields are omitted (None)."""
    status: str = "ok"
    # Present only when EMAIL_BACKEND=none. Named per the flow at the call site.
    invite_link: str | None = None
    reset_link: str | None = None


class UserResp(BaseModel):
    id: str
    email: str
    name: str | None = None
    tenant_id: str
    role: str
    # Cross-tenant superuser flag — the frontend gates the platform-admin nav on this.
    platform_admin: bool = False


# --- E10 tenant Admin Dashboard: usage + billing ---------------------------

class CorpusUsageResp(BaseModel):
    corpus_id: str
    name: str
    queries: int = 0
    documents: int = 0
    storage_gb: float = 0.0
    gpu_seconds: float = 0.0


class UsagePointResp(BaseModel):
    """One day of the ~30-day served-query series."""
    date: str
    queries: int = 0


class UsageResp(BaseModel):
    """GET /admin/usage — tenant-scoped usage rollup + a daily query series."""
    queries: int = 0
    documents: int = 0
    storage_gb: float = 0.0
    gpu_seconds: float = 0.0
    n_corpora: int = 0
    by_corpus: list[CorpusUsageResp] = []
    series: list[UsagePointResp] = []


class BillingResp(BaseModel):
    """GET /admin/billing — the billing shell (Stripe deferred). Plan + limits + current
    usage + an estimated cost from the pricing rate card."""
    plan: str = "beta"
    limits: dict = {}
    usage: dict = {}
    rate_card: dict = {}
    estimated_cost_usd: float = 0.0
    currency: str = "usd"
    period: str


# --- Stripe billing (dark-launched) -----------------------------------------

class BillingStatusResp(BaseModel):
    """GET /billing/status — always 200, safe when billing is disabled. `enabled` mirrors the flag;
    `rate_card` is the pricing rate card; `portal_available` tells the UI whether to offer the
    'manage billing' button."""
    enabled: bool = False
    rate_card: dict = {}
    portal_available: bool = False


class BillingPortalResp(BaseModel):
    """POST /billing/portal — the Stripe billing-portal session URL to redirect the admin to."""
    url: str


# --- E11 platform-admin console: tenants + fleet usage ----------------------

class TenantLimitsReq(BaseModel):
    """PATCH /platform-admin/tenants/{id}/limits — the 'contact us to raise it' lever. Either field
    optional; null clears the override (fall back to the config default). 0 means unlimited."""
    max_docs_override: int | None = Field(default=None, ge=0)
    max_queries_override: int | None = Field(default=None, ge=0)


class TenantLimitsResp(BaseModel):
    """A tenant's current beta-limit overrides after a PATCH."""
    tenant_id: str
    max_docs_override: int | None = None
    max_queries_override: int | None = None


class PlatformTenantResp(BaseModel):
    id: str
    name: str
    created_at: datetime.datetime
    n_users: int = 0
    n_corpora: int = 0
    plan: str = "beta"
    status: str = "active"


class PlatformTenantUsageResp(BaseModel):
    tenant_id: str
    name: str
    queries: int = 0
    documents: int = 0
    storage_gb: float = 0.0
    gpu_seconds: float = 0.0
    est_cost_usd: float = 0.0


class PlatformUsageTotalsResp(BaseModel):
    queries: int = 0
    storage_gb: float = 0.0
    gpu_seconds: float = 0.0
    est_cost_usd: float = 0.0
    n_tenants: int = 0


class PlatformUsageResp(BaseModel):
    """GET /platform-admin/usage — per-tenant cost + fleet totals (cross-tenant)."""
    tenants: list[PlatformTenantUsageResp] = []
    totals: PlatformUsageTotalsResp = PlatformUsageTotalsResp()


# --- E12 platform-admin GPU controls: Lambda Cloud serving box --------------

class GpuInstanceResp(BaseModel):
    """The one running LAMBDA_INSTANCE_NAME box (null in GpuStatusResp when offline)."""
    id: str
    name: str
    type: str
    region: str
    ip: str | None = None
    price_cents_per_hour: int | None = None


class GpuStatusResp(BaseModel):
    """GET /platform-admin/gpu/status — the box's derived lifecycle state + health probes. `enabled`
    is false / `state` "offline" / `instance` null when LAMBDA_API_KEY is unset (still a 200)."""
    enabled: bool
    # offline | booting | provisioning | warming | serving | terminating
    state: str
    instance: GpuInstanceResp | None = None
    serve_reachable: bool = False
    engine_ready: bool = False
    onboard_reachable: bool = False
    # null when Cloudflare creds are absent or the host isn't a DNS name (localhost / IP literal).
    dns_pointed: bool | None = None
    hourly_usd: float | None = None


class GpuStartResp(BaseModel):
    instance_id: str
    state: str = "booting"


class GpuStopResp(BaseModel):
    state: str = "terminating"


class CorpusCreateReq(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    source_type: str = "upload"


class CorpusResp(BaseModel):
    id: str
    name: str
    source_type: str
    status: str
    n_documents: int
    # Resumable onboarding wizard: the step the user left off at + their chosen model tier, so the
    # frontend reopens at the right place (see routers/onboarding.py).
    onboarding_step: str = "name"
    model_tier: str | None = None
    mcp_token: str | None = None
    # Populated after a successful training run (see Job._run_training).
    n_cartridges: int | None = None
    train_seconds: float | None = None
    corpus_tokens: int | None = None
    created_at: datetime.datetime


class DocumentResp(BaseModel):
    id: str
    filename: str
    size: int
    # Per-document onboarding progress surfaced on wizard resume.
    parse_status: str = "pending"
    # Short reason text extraction failed (unsupported type / encrypted / corrupt); null when
    # parsing hasn't run or succeeded. Lets the wizard show WHY a file didn't onboard.
    parse_error: str | None = None
    onboard_status: str = "pending"
    # One-sentence LLM description written at onboarding (Feature 1); null until the best-effort
    # describe pass fills it (or if descriptions are off / the pass failed). Shown as a secondary
    # line in the Documents tab when present.
    description: str | None = None


# Wizard step values, single source of truth for validation.
ONBOARDING_STEPS = ("name", "documents", "model", "review", "onboarding", "ready")


class OnboardingPatchReq(BaseModel):
    """Persist the wizard cursor and/or the chosen model tier. Both optional — the frontend PATCHes
    whichever changed as the user moves through the steps. `model_tier` is validated against the
    serving registry in the router (unknown ids are rejected there)."""
    onboarding_step: str | None = Field(default=None, pattern="^(name|documents|model|review|onboarding|ready)$")
    model_tier: str | None = None


class OnboardingStateResp(BaseModel):
    """The full resumable-onboarding snapshot: where the wizard is, the chosen tier, and per-document
    parse/onboard status so the frontend can reopen at the exact step and show per-file progress."""
    corpus_id: str
    onboarding_step: str
    status: str  # corpus lifecycle (new|training|ready|failed) — the wizard reads both
    model_tier: str | None = None
    model_ref: str | None = None
    n_documents: int
    documents: list[DocumentResp] = []


class JobResp(BaseModel):
    id: str
    corpus_id: str
    kind: str
    status: str
    detail: str
    progress: float = 0.0  # 0.0..1.0
    eta_seconds: int | None = None  # est. seconds remaining; None = unknown / not training
    created_at: datetime.datetime
    updated_at: datetime.datetime


class ProgressReq(BaseModel):
    """Worker -> control-plane training heartbeat."""
    progress: float
    eta_seconds: int | None = None
    detail: str | None = None


class GcCartsReq(BaseModel):
    """Body for the operator-invoked cart GC sweep. `confirm` MUST be explicitly true to delete —
    the default is a dry run (list orphans only)."""
    confirm: bool = False


class ChatMsg(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=4000)


class ChatReq(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    # Bounded so a single request can't compose an unbounded number of carts /
    # stuff an unbounded RAG context through the GPU.
    k: int = Field(default=3, ge=1, le=20)
    # Prior turns of THIS conversation (oldest first). They ride as small per-turn
    # prefill on top of the resident corpus KV — the model remembers the chat AND
    # the documents. Bounded: 20 turns x 4k chars.
    history: list[ChatMsg] = Field(default_factory=list, max_length=20)
    # Compare-stream only: the cart side's stream already retrieved these doc_ids; the
    # RAG side passes them back so retrieval runs ONCE per question and both sides see
    # identical evidence. Client-supplied, so they are validated against the corpus
    # (retrieval.context_for) — never interpreted as paths.
    doc_ids: list[str] | None = Field(default=None, max_length=20)
    # Optional per-request answer budget — the industry-standard shape: the client asks,
    # the server CLAMPS to the INFERENCE_MAX_TOKENS ceiling (never raises past it), and
    # the model still stops at EOS well before the cap on short answers. None = ceiling.
    max_tokens: int | None = Field(default=None, ge=1, le=4096)


class ScaleRunSaveReq(BaseModel):
    """A finished scale-test run posted by the Scale Test tab so it can be re-loaded later."""
    max_concurrency: int = Field(ge=1, le=64)
    n_queries: int = Field(default=0, ge=0, le=64)
    # Per-level series ({u, cart:{...}, rag:{...}}); stored verbatim. Bounded so a client can't
    # persist an unbounded blob.
    points: list[dict] = Field(default_factory=list, max_length=64)


class ScaleRunResp(BaseModel):
    id: str
    corpus_id: str
    max_concurrency: int
    n_queries: int
    points: list[dict]
    created_at: datetime.datetime


class AuditEventResp(BaseModel):
    id: str
    event: str
    corpus_id: str | None = None
    detail: str  # compact JSON string (see audit.record_event) — the client parses it as needed
    created_at: datetime.datetime


class SourceRef(BaseModel):
    id: str
    title: str


class ChatResp(BaseModel):
    answer: str
    used_docs: list[str]
    sources: list[SourceRef] = []


# --- source connectors (Google Drive / SharePoint) --------------------------------------------

class ConnectorAuthorizeResp(BaseModel):
    """The provider consent URL the SPA redirects itself to (we don't 302 the XHR)."""
    url: str


class ConnectionResp(BaseModel):
    """A tenant's authorized source connection (no tokens ever exposed — only the account label)."""
    id: str
    provider: str
    account_label: str
    created_at: datetime.datetime


class BrowseFolderResp(BaseModel):
    """A folder the user can drill into. `id` is an opaque provider ref the client passes straight back
    (Drive file id / "root"; SharePoint "site:<id>" or "item:<driveId>:<itemId>")."""
    id: str
    name: str


class BrowseResp(BaseModel):
    folders: list[BrowseFolderResp]
    supported_files: int  # count of importable files AT THIS LEVEL (not recursive)
    path_hint: str        # human label for where we are ("My Drive" / "SharePoint sites" / ...)


class ImportReq(BaseModel):
    """Start importing a connected source's folder into the corpus. folder_id is the opaque provider
    ref from browse; site_id is only needed for a SharePoint site's default library (optional)."""
    connection_id: str
    folder_id: str = ""
    folder_name: str = Field(default="", max_length=400)
    site_id: str | None = None


class ImportStatusResp(BaseModel):
    """Latest ImportRun for a corpus, or {"state": "none"} when nothing has been imported. Counters are
    live while state == 'running'."""
    state: str  # none|running|done|failed|limited
    id: str | None = None
    folder_name: str | None = None
    imported: int = 0
    skipped: int = 0
    failed: int = 0
    error: str | None = None
    created_at: datetime.datetime | None = None
    finished_at: datetime.datetime | None = None
