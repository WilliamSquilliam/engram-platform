"""Domain model. Tenants own corpora; corpora hold documents and produce
cartridges via training jobs. Multi-tenant isolation is enforced everywhere by
filtering on tenant_id (see routers)."""
import datetime
import uuid

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime.datetime:
    # Naive UTC (matches the existing DateTime columns) via the non-deprecated API.
    return datetime.datetime.now(datetime.UTC).replace(tzinfo=None)


class Tenant(Base):
    __tablename__ = "tenants"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=_now)


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"))
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String)
    # Workspace-level role (E1): the FIRST user of a tenant is "admin", teammates
    # invited in default to "member". require_tenant_admin gates /admin/*.
    role: Mapped[str] = mapped_column(String, default="member")
    # Cross-tenant superuser (the founder). Gates /platform-admin/* via
    # require_platform_admin. Seeded from BOOTSTRAP_ADMIN / PLATFORM_ADMIN_EMAIL.
    platform_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    # Set true once the user proves control of the address (accept-invite / reset).
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    # Soft-disable a member without deleting them (removed from a workspace, etc.).
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=_now)


class AccessRequest(Base):
    """Invite-only beta waitlist (E1). Public /auth/request-access writes a pending
    row here; a platform_admin approves it, which provisions a tenant + admin user
    and issues an accept-invite link. `status` ∈ pending|approved|denied."""
    __tablename__ = "access_requests"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String, index=True)
    name: Mapped[str] = mapped_column(String)
    tenant_name: Mapped[str] = mapped_column(String)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String, default="pending", index=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=_now)


class Invite(Base):
    """A pending invitation into a tenant (E1). Used for BOTH teammate invites (an
    admin invites an email into their workspace) and approval invites (a platform
    admin approves an access request, seeding a new tenant + its admin). Only the
    HASH of the token is stored — the raw token lives only in the link. Redeemed via
    /auth/accept-invite, which creates/activates the user and clears the invite."""
    __tablename__ = "invites"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), index=True)
    email: Mapped[str] = mapped_column(String, index=True)
    role: Mapped[str] = mapped_column(String, default="member")  # admin|member
    token_hash: Mapped[str] = mapped_column(String, index=True)  # sha256 of the raw token
    expires_at: Mapped[datetime.datetime] = mapped_column(DateTime)
    accepted_at: Mapped[datetime.datetime | None] = mapped_column(DateTime, nullable=True)
    created_by: Mapped[str | None] = mapped_column(String, nullable=True)  # inviting user id
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=_now)


class PasswordReset(Base):
    """A single-use password-reset grant (E1). Only the token HASH is stored; the raw
    token is in the reset link. Short expiry (config.PASSWORD_RESET_EXPIRE_HOURS).
    `used_at` marks it consumed so a link can't be replayed."""
    __tablename__ = "password_resets"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    token_hash: Mapped[str] = mapped_column(String, index=True)
    expires_at: Mapped[datetime.datetime] = mapped_column(DateTime)
    used_at: Mapped[datetime.datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=_now)


class Corpus(Base):
    __tablename__ = "corpora"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), index=True)
    name: Mapped[str] = mapped_column(String)
    source_type: Mapped[str] = mapped_column(String, default="upload")  # upload|sharepoint|confluence
    status: Mapped[str] = mapped_column(String, default="new")  # new|training|ready|failed
    # Wizard CURSOR for the resumable per-corpus onboarding flow — where the user is in the 5-step
    # wizard, so they can exit and reopen at the right step. Distinct from `status` (the corpus
    # lifecycle): a corpus can sit at step "review" (status "new") or be mid-onboard (status
    # "training"). Values: name|documents|model|review|onboarding|ready.
    onboarding_step: Mapped[str] = mapped_column(String, default="name")
    # Chosen model tier id (serving.tiers()); nullable until the user reaches the "model" step.
    model_tier: Mapped[str | None] = mapped_column(String, nullable=True)
    # Resolved weights id the corpus is PINNED to (serving.model_ref_for_tier), set from the tier at
    # onboard start so carts are stamped to a fixed model. Nullable until onboarding starts.
    model_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    mcp_token: Mapped[str | None] = mapped_column(String, nullable=True)
    # Populated by the last successful training run; feeds the cost/break-even view.
    n_cartridges: Mapped[int | None] = mapped_column(Integer, nullable=True)
    train_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    corpus_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=_now)

    documents: Mapped[list["Document"]] = relationship(back_populates="corpus", cascade="all, delete-orphan")
    jobs: Mapped[list["Job"]] = relationship(back_populates="corpus", cascade="all, delete-orphan")
    scale_runs: Mapped[list["ScaleRun"]] = relationship(back_populates="corpus", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    corpus_id: Mapped[str] = mapped_column(ForeignKey("corpora.id"), index=True)
    filename: Mapped[str] = mapped_column(String)
    storage_key: Mapped[str] = mapped_column(String)
    size: Mapped[int] = mapped_column(Integer, default=0)
    # Per-document onboarding progress the wizard surfaces on resume (a doc can still be parsing /
    # onboarding when the user reopens). Advanced by the onboard worker; independent of corpus.status.
    parse_status: Mapped[str] = mapped_column(String, default="pending")  # pending|parsing|parsed|failed
    # Short human-readable reason a document failed text extraction (unsupported type, encrypted,
    # corrupt); null when parsing hasn't run or succeeded. Surfaced in DocumentResp so the wizard
    # can tell the user WHY a file didn't onboard.
    parse_error: Mapped[str | None] = mapped_column(String, nullable=True)
    onboard_status: Mapped[str] = mapped_column(String, default="pending")  # pending|onboarding|ready|failed
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=_now)

    corpus: Mapped["Corpus"] = relationship(back_populates="documents")


class ScaleRun(Base):
    """A completed fleet scale-test run, saved so the Scale Test tab can re-load past runs and
    populate their finished numbers. `points` is the per-level series the SSE ramp produced
    (list of {u, cart:{qps,ttft,lat,...}, rag:{...}}), stored verbatim as JSON."""
    __tablename__ = "scale_runs"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    corpus_id: Mapped[str] = mapped_column(ForeignKey("corpora.id"), index=True)
    max_concurrency: Mapped[int] = mapped_column(Integer)
    n_queries: Mapped[int] = mapped_column(Integer, default=0)
    points: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=_now)

    corpus: Mapped["Corpus"] = relationship(back_populates="scale_runs")


class Measurement(Base):
    """One measured per-query metric from the live serving path (see app/measurements.py). Each row
    is ONE side of a head-to-head: `side` is "cart" (the resident-KV cartridge path) or "rag" (the
    re-prefill baseline), both clocked on the same vLLM engine. Durable twin of the process-local ring
    buffer, so the demo's measured aggregate + the /metrics/savings lifetime totals survive restarts.

    Columns mirror the metrics dict record() receives from the Inference Service (latency_ms, ttft_ms,
    prompt_tokens, resident_kv_tokens, gen_tokens, decode_tps, confidence); cost_per_query is the
    length-normalized $/query computed at record time so lifetime aggregates don't re-derive pricing."""
    __tablename__ = "measurements"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=_now, index=True)
    side: Mapped[str] = mapped_column(String, index=True)  # "cart" | "rag"
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    ttft_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    resident_kv_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gen_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    decode_tps: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_per_query: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Deployment labels captured at record time (model / instance the number was measured on).
    model_label: Mapped[str | None] = mapped_column(String, nullable=True)
    instance_label: Mapped[str | None] = mapped_column(String, nullable=True)


class AuditEvent(Base):
    """Append-only lifecycle receipt. Every data-affecting action (corpus.delete, carts.gc,
    carts.offboard_failed) writes one row so a deletion is PROVABLE after the fact — the record a
    security reviewer asks for when the privacy pitch ('deleting a memory removes the document from
    serving') is challenged. `detail` holds compact JSON (which cart ids were deleted / retained as
    shared / which ML-plane calls failed), so the row is self-contained without joining live state
    that the delete already removed. tenant_id is indexed (the /audit read filters on it) and NON-null
    for tenant-scoped events; the store-wide GC sweep isn't owned by a tenant, so it writes the
    "_system" sentinel rather than a real tenant id."""
    __tablename__ = "audit_events"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String, index=True)  # "_system" for operator/GC events
    user_id: Mapped[str | None] = mapped_column(String, nullable=True)  # None for non-user (worker/GC) actions
    event: Mapped[str] = mapped_column(String, index=True)  # e.g. "corpus.delete" | "carts.gc" | "carts.offboard_failed"
    corpus_id: Mapped[str | None] = mapped_column(String, nullable=True)  # nullable: not every event is corpus-scoped
    detail: Mapped[str] = mapped_column(Text, default="")  # compact JSON payload (cart ids, errors, counts)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=_now, index=True)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    corpus_id: Mapped[str] = mapped_column(ForeignKey("corpora.id"), index=True)
    kind: Mapped[str] = mapped_column(String, default="train")
    status: Mapped[str] = mapped_column(String, default="pending")  # pending|running|succeeded|failed|canceled
    # Set by the user's "cancel training" action; the worker polls this via its
    # progress-heartbeat response and aborts cooperatively (see routers/jobs.py).
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    detail: Mapped[str] = mapped_column(Text, default="")
    progress: Mapped[float] = mapped_column(Float, default=0.0)  # 0.0..1.0, updated by worker heartbeats
    eta_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)  # est. time remaining; None = unknown
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=_now, onupdate=_now)

    corpus: Mapped["Corpus"] = relationship(back_populates="jobs")
