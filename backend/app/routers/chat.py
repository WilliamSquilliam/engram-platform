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

from .. import config, limits, measurements, ml_client, retrieval, usage
from ..deps import get_current_user, get_db
from ..models import Corpus, Tenant, User
from ..ratelimit import limiter
from ..schemas import ChatReq, ChatResp
from ..storage import storage

router = APIRouter(tags=["chat"])


def _enforce_query_limit(db: Session, corpus: Corpus) -> None:
    """Beta monthly-query limit (invisible until hit): 429 BEFORE the engine is called if this
    workspace has already reached its cap for the current month. Scoped to the corpus's owning tenant
    (the billing/attribution key). No-op when the cap is 0 (unlimited)."""
    tenant = db.get(Tenant, corpus.tenant_id)
    if tenant is not None:
        limits.check_query_limit(tenant, usage.tenant_query_count_this_month(db, corpus.tenant_id))


def _hybrid_split(corpus_id: str, question: str, doc_ids: list[str]) -> tuple[list[str], str]:
    """QRC hybrid serving split. In hybrid mode with >1 retrieved doc, keep the TOP-1 doc as the
    resident cart and route the OTHER docs' answer-bearing chunks into a small real-token `context`:
    returns ([doc_ids[0]], routed_context). QRC_MODE=off (or a single doc) is the legacy multi-cart
    serve — the full doc_ids list, empty context. The caller still reports ALL doc_ids as used_docs /
    sources (the evidence the user sees is unchanged; only the serving mechanism differs)."""
    if config.QRC_MODE == "hybrid" and len(doc_ids) > 1:
        context = retrieval.route_chunks_context(corpus_id, question, doc_ids[1:])
        return [doc_ids[0]], context
    return doc_ids, ""


def _answer(corpus: Corpus, req: ChatReq, db: Session) -> ChatResp:
    if corpus.status != "ready":
        raise HTTPException(400, f"Document base not ready (status={corpus.status})")
    _enforce_query_limit(db, corpus)
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
                raise HTTPException(404, "no documents to retrieve for this document base")
        # QRC hybrid split: serve the top-1 cart + routed chunks of docs 2..k as `context` (hybrid), or
        # the full doc_ids list with no context (off). used_docs/sources still report ALL doc_ids below.
        serve_ids, context = _hybrid_split(corpus.id, req.question, doc_ids)
        result = ml_client.inference_query(serve_ids, req.question, config.INFERENCE_MAX_TOKENS,
                                           history=[m.model_dump() for m in req.history],
                                           context=context)
        # tier="cartridge": the non-stream serve path always answers cart-alone (adaptive escalation
        # is deliberately STREAM-ONLY — see chat_stream — so a one-shot answer never silently swaps to
        # the RAG backup). tenant_id attributes this query to the corpus's tenant for per-tenant billing.
        metrics = result.get("metrics")
        if metrics is not None:
            metrics = {**metrics, "tier": "cartridge"}
        measurements.record(metrics, None, tenant_id=corpus.tenant_id)
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
        raise HTTPException(404, "Document base not found")
    return _answer(corpus, req, db)


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
        raise HTTPException(404, "Document base not found")
    if corpus.status != "ready" or config.INFERENCE_BACKEND != "vllm":
        raise HTTPException(400, "streaming chat needs a ready document base on the vLLM backend")
    # Beta monthly-query limit, BEFORE any retrieval / GPU work (same gate as the non-stream path).
    _enforce_query_limit(db, corpus)
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
            raise HTTPException(404, "no documents to retrieve for this document base")
    sources = retrieval.doc_sources(corpus.id, doc_ids)

    # QRC hybrid split: serve the top-1 cart + routed chunks of docs 2..k as `context` (hybrid), or the
    # full doc_ids list with no context (off). The head frame + sources below still report ALL doc_ids —
    # the evidence the user sees is unchanged; only the serve payload's ids/context differ.
    serve_ids, context = _hybrid_split(corpus.id, req.question, doc_ids)

    _theta = config.ADAPTIVE_THETA
    theta = float(_theta) if _theta not in (None, "") else None
    url = f"{config.INFERENCE_SERVICE_URL}/query_stream"
    payload = {"doc_ids": serve_ids, "question": req.question,
               "max_tokens": config.INFERENCE_MAX_TOKENS, "history": history, "context": context}

    async def gen():
        # head carries used_docs so the UI can PIN them on follow-up turns (skip re-retrieval) and
        # render the source list immediately, before the first token lands.
        yield ("data: " + json.dumps({"head": True, "used_docs": doc_ids,
                                      "sources": sources}) + "\n\n")
        metrics_final = None
        tier = "cartridge"
        escalated = False
        done_emitted = False  # did a REAL terminal frame (done or error) go out? drives the finally guard
        try:
            hold = {"m": None}
            # Forward the cart-alone stream. Check for client disconnect between frames: if the browser
            # went away, stop pumping the GPU stream (closing _forward exits its httpx.stream context)
            # instead of decoding tokens no one will read.
            for frame in _forward(url, payload, hold):
                if await request.is_disconnected():
                    return
                yield frame
            metrics_final = hold["m"]
            conf = (metrics_final or {}).get("confidence")
            # ADAPTIVE ROUTER: under-confident cart answer -> escalate to the RAG backup (full
            # retrieved context on the SAME engine) and flag it in-band so the answer shown is the
            # backup's and the user knows the cart was unsure. Skip the (extra GPU) escalation entirely
            # if the client already disconnected.
            if theta is not None and conf is not None and conf < theta:
                if await request.is_disconnected():
                    return
                escalated, tier = True, "rag-backup"
                context = retrieval.context_for(corpus.id, doc_ids)
                yield ("data: " + json.dumps({"escalate": True, "confidence": conf,
                                              "theta": theta}) + "\n\n")
                yield ("data: " + json.dumps({"delta": "\n\n_(cartridge was unsure — verifying "
                                              "with the full documents)_\n\n"}) + "\n\n")
                bhold = {"m": None}
                for frame in _forward(f"{config.INFERENCE_SERVICE_URL}/rag_query_stream",
                                      {"context": context, "question": req.question,
                                       "max_tokens": config.INFERENCE_MAX_TOKENS, "history": history},
                                      bhold):
                    if await request.is_disconnected():
                        return
                    yield frame
                if bhold["m"]:
                    metrics_final = bhold["m"]               # shown answer = backup; report ITS metrics
            if metrics_final is not None:
                metrics_final = {**metrics_final, "tier": tier, "escalated": escalated}
                # tenant_id attributes this served query to the corpus's tenant for per-tenant billing.
                measurements.record(metrics_final, None, tenant_id=corpus.tenant_id)
                done_emitted = True
                yield ("data: " + json.dumps({"done": True, "metrics": metrics_final}) + "\n\n")
        except httpx.HTTPError as e:
            # head already went out 200 -> report the GPU-side failure IN-BAND, not a silent truncate.
            yield ("data: " + json.dumps({"error": f"inference service failed ({type(e).__name__})"})
                   + "\n\n")
            done_emitted = True  # an in-band error is a terminal frame; don't also synthesize a done
            return
        except KeyError as e:            # context_for got an id not in the corpus
            yield ("data: " + json.dumps({"error": str(e)}) + "\n\n")
            done_emitted = True
            return
        finally:
            # A client must NEVER hang waiting for a terminal frame. If we didn't produce a real done
            # (or an in-band error) — e.g. the ml-service returned no metrics — synthesize a terminal
            # done with null metrics so the reader always resolves. (On disconnect we return early
            # WITHOUT emitting, since there's no client left to receive it.)
            if not done_emitted and not await request.is_disconnected():
                yield ("data: " + json.dumps({"done": True, "metrics": None}) + "\n\n")

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
    return _answer(corpus, req, db)
