"""Chat (JWT-authenticated, for the web UI) and the MCP query endpoint
(token-authenticated, for the tenant's own LLM). Both go through the same ML
service /query."""
import secrets

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from .. import config, measurements, ml_client, retrieval
from ..deps import get_current_user, get_db
from ..models import Corpus, User
from ..ratelimit import limiter
from ..schemas import ChatReq, ChatResp
from ..storage import storage

router = APIRouter(tags=["chat"])


def _answer(corpus: Corpus, req: ChatReq) -> ChatResp:
    if corpus.status != "ready":
        raise HTTPException(400, f"Corpus not ready (status={corpus.status})")
    if config.INFERENCE_BACKEND == "vllm":
        # Resident-KV serving: control plane retrieves the cart doc_ids, the Inference Service serves
        # them (no per-query document prefill). Retrieval lives here (C2 split), not on the GPU.
        # Conversation history rides as small per-turn prefill on top of the resident corpus KV.
        if req.doc_ids:
            # Session-pinned carts: a follow-up turn echoes the doc_ids the first turn resolved,
            # so we skip the 221.7ms/query retrieval and reuse them verbatim. The ids are
            # client-supplied — validate membership against the corpus (KeyError -> 400) without
            # reading any doc text (the vLLM serve path takes doc_ids, never raw context).
            doc_ids = list(req.doc_ids)
            try:
                retrieval.validate_doc_ids(corpus.id, doc_ids)
            except KeyError as e:
                raise HTTPException(400, str(e)) from e
        else:
            doc_ids = retrieval.retrieve(corpus.id, req.question, req.k)
            if not doc_ids:
                raise HTTPException(404, "no documents to retrieve for this corpus")
        result = ml_client.inference_query(doc_ids, req.question, config.INFERENCE_MAX_TOKENS,
                                           history=[m.model_dump() for m in req.history])
        measurements.record(result.get("metrics"), None)  # feed the demo's measured aggregate
        return ChatResp(answer=result["answer"], used_docs=doc_ids,
                        sources=retrieval.doc_sources(corpus.id, doc_ids))
    # default: the HF ml_service (retrieves + generates internally). req.doc_ids is ignored on this
    # path: the hf ml_service does its OWN retrieval end-to-end and takes no doc_ids, so there is no
    # seam to pin here — it returns its own used_docs, which the client echoes.
    # (The vllm path above is where pinning saves the separate control-plane retrieval trip.)
    result = ml_client.query(str(storage.corpus_dir(corpus.id)), req.question, req.k)
    return ChatResp(answer=result["answer"], used_docs=result["used_docs"],
                    sources=retrieval.doc_sources(corpus.id, result["used_docs"]))


@router.post("/corpora/{corpus_id}/chat", response_model=ChatResp)
@limiter.limit("30/minute")  # one L40S behind this — a single tenant must not saturate it
def chat(request: Request, corpus_id: str, req: ChatReq, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    corpus = db.get(Corpus, corpus_id)
    if corpus is None or corpus.tenant_id != user.tenant_id:
        raise HTTPException(404, "Corpus not found")
    return _answer(corpus, req)


@router.post("/mcp/{corpus_id}/query", response_model=ChatResp)
@limiter.limit("30/minute")  # same GPU behind this as /chat — a leaked corpus token must not saturate it
def mcp_query(
    request: Request,
    corpus_id: str,
    req: ChatReq,
    x_mcp_token: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    """Called by the MCP Gateway. Auth is the per-corpus MCP token, not a user JWT,
    so a tenant can wire their LLM to a corpus without a user session. Rate-bucketed
    by client address (no JWT here — the limiter's key func falls back to XFF/remote)."""
    corpus = db.get(Corpus, corpus_id)
    if (corpus is None or not corpus.mcp_token
            or not secrets.compare_digest(corpus.mcp_token, x_mcp_token or "")):
        raise HTTPException(401, "Invalid MCP token")
    return _answer(corpus, req)
