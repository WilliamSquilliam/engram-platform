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


class UserResp(BaseModel):
    id: str
    email: str
    tenant_id: str
    role: str


class CorpusCreateReq(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    source_type: str = "upload"


class CorpusResp(BaseModel):
    id: str
    name: str
    source_type: str
    status: str
    n_documents: int
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
