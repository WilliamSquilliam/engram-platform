"""Chat (JWT-authenticated, for the web UI) and the MCP query endpoint
(token-authenticated, for the tenant's own LLM). Both go through the same ML
service /query.

Two shapes: the non-streaming /chat (one JSON answer, also used by the MCP path)
and the token-streaming /chat/stream (SSE) the conversational chat UI consumes so
answers render token-by-token. Both are tenant-scoped + JWT-authed identically."""
import json
import secrets

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
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


# --- Streaming chat (SSE) -------------------------------------------------------------------------
# The conversational chat UI (E3) consumes this so the answer renders token-by-token. Shape mirrors
# compare/stream's cart side so the frontend's SSE reader is uniform:
#   head  -> {"head": true, "used_docs": [...], "sources": [...]}   (once, first)
#   delta -> {"delta": "text"}                                       (many, the model writing)
#   done  -> {"done": true, "metrics": {...}}                        (once, MEASURED numbers)
#   error -> {"error": "..."}                                        (in-band; head already went 200)
# The resident-KV cart path is the product surface here (INFERENCE_BACKEND=vllm). Like compare's
# cart side it also runs the ADAPTIVE ROUTER: if the cart-alone answer is under-confident it escalates
# to the RAG backup (full retrieved context on the SAME engine) and flags it in-band, so the user
# always sees when the backup fired.
def _forward(url: str, payload: dict, hold: dict):
    """Forward an ml-service SSE stream's delta lines; stash its final metrics in hold['m'] and
    SWALLOW its 'done' frame so the caller emits one synthesized terminal frame."""
    with httpx.stream("POST", url, json=payload, timeout=300.0,
                      headers=ml_client._ml_headers()) as r:  # ML-plane shared-token auth (off by default)
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.strip():
                continue
            if '"done"' in line:
                try:
                    hold["m"] = json.loads(line.removeprefix("data: ")).get("metrics")
                except ValueError:
                    pass
                return
            yield line + "\n\n"


@router.post("/corpora/{corpus_id}/chat/stream")
@limiter.limit("30/minute")  # one GPU generation per question; single-tenant DoS guard, same as /chat
def chat_stream(
    request: Request,
    corpus_id: str,
    req: ChatReq,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Token-streaming chat (SSE): head event with sources, delta events as the model writes, then a
    done event with measured metrics. Tenant-scoped + JWT-authed exactly like /chat. Needs a ready
    corpus on the vLLM backend (the streaming serve path); the HF path has no token stream."""
    corpus = db.get(Corpus, corpus_id)
    if corpus is None or corpus.tenant_id != user.tenant_id:
        raise HTTPException(404, "Corpus not found")
    if corpus.status != "ready" or config.INFERENCE_BACKEND != "vllm":
        raise HTTPException(400, "streaming chat needs a ready corpus on the vLLM backend")
    history = [m.model_dump() for m in req.history]

    # Resolve doc_ids ONCE. A follow-up turn echoes the first turn's doc_ids so retrieval is skipped
    # and the same evidence is reused; a first turn retrieves. Client-supplied ids are validated
    # against the corpus (unknown ids -> 400) before any GPU work.
    if req.doc_ids:
        doc_ids = list(req.doc_ids)
        try:
            retrieval.validate_doc_ids(corpus.id, doc_ids)
        except KeyError as e:
            raise HTTPException(400, str(e)) from e
    else:
        doc_ids = retrieval.retrieve(corpus.id, req.question, req.k)
        if not doc_ids:
            raise HTTPException(404, "no documents to retrieve for this corpus")
    sources = retrieval.doc_sources(corpus.id, doc_ids)

    _theta = config.ADAPTIVE_THETA
    theta = float(_theta) if _theta not in (None, "") else None
    url = f"{config.INFERENCE_SERVICE_URL}/query_stream"
    payload = {"doc_ids": doc_ids, "question": req.question,
               "max_tokens": config.INFERENCE_MAX_TOKENS, "history": history}

    def gen():
        # head carries used_docs so the UI can PIN them on follow-up turns (skip re-retrieval) and
        # render the source list immediately, before the first token lands.
        yield ("data: " + json.dumps({"head": True, "used_docs": doc_ids,
                                      "sources": sources}) + "\n\n")
        metrics_final = None
        tier = "cartridge"
        escalated = False
        try:
            hold = {"m": None}
            yield from _forward(url, payload, hold)          # cart-alone answer
            metrics_final = hold["m"]
            conf = (metrics_final or {}).get("confidence")
            # ADAPTIVE ROUTER: under-confident cart answer -> escalate to the RAG backup (full
            # retrieved context on the SAME engine) and flag it in-band so the answer shown is the
            # backup's and the user knows the cart was unsure.
            if theta is not None and conf is not None and conf < theta:
                escalated, tier = True, "rag-backup"
                context = retrieval.context_for(corpus.id, doc_ids)
                yield ("data: " + json.dumps({"escalate": True, "confidence": conf,
                                              "theta": theta}) + "\n\n")
                yield ("data: " + json.dumps({"delta": "\n\n_(cartridge was unsure — verifying "
                                              "with the full documents)_\n\n"}) + "\n\n")
                bhold = {"m": None}
                yield from _forward(f"{config.INFERENCE_SERVICE_URL}/rag_query_stream",
                                    {"context": context, "question": req.question,
                                     "max_tokens": config.INFERENCE_MAX_TOKENS, "history": history},
                                    bhold)
                if bhold["m"]:
                    metrics_final = bhold["m"]               # shown answer = backup; report ITS metrics
        except httpx.HTTPError as e:
            # head already went out 200 -> report the GPU-side failure IN-BAND, not a silent truncate.
            yield ("data: " + json.dumps({"error": f"inference service failed ({type(e).__name__})"})
                   + "\n\n")
            return
        except KeyError as e:            # context_for got an id not in the corpus
            yield ("data: " + json.dumps({"error": str(e)}) + "\n\n")
            return
        if metrics_final is not None:
            metrics_final = {**metrics_final, "tier": tier, "escalated": escalated}
            measurements.record(metrics_final, None)          # feed the demo's measured aggregate
            yield ("data: " + json.dumps({"done": True, "metrics": metrics_final}) + "\n\n")

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


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
