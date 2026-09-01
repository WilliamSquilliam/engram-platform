"""Side-by-side strategy compare for the workspace 'Compare' view: answer one question two ways over
the same corpus — Engram Smart CAG (the product: the adaptive cartridge router) vs RAG (the baseline). RAG
is the only realistic head-to-head baseline.

The ML service runs both on the local base model and returns REAL latency + token counts; here we
attach modeled $/query — RAG at frontier prices, the cartridge path at its multiplexed marginal — so
the numbers tie out with the /demo page (see metrics.py). Latency is measured same-hardware; $ is the
business model.
"""
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .. import config, measurements, metrics, ml_client, retrieval
from ..deps import get_current_user, get_db
from ..models import Corpus, ScaleRun, User
from ..ratelimit import limiter
from ..schemas import ChatReq, ScaleRunResp, ScaleRunSaveReq
from ..storage import storage

router = APIRouter(tags=["compare"])

_LABELS = {"everyday": "Engram Smart CAG", "rag": "RAG"}
_ORDER = ("everyday", "rag")


def _ratio(alt: float | None, base: float | None) -> float | None:
    """How many times `base` (Engram Smart CAG) beats `alt` (lower cost/latency = better)."""
    if not alt or not base or base <= 0:
        return None
    return round(alt / base, 1)


def _price(m: dict | None) -> float | None:
    """$/query, length-normalized (measured TTFT + standard answer at measured decode rate);
    falls back to raw-latency pricing when the components are missing."""
    if not m:
        return None
    return (metrics.price_normalized(m.get("ttft_ms"), m.get("decode_tps"))
            or (metrics.price_from_latency(m["latency_ms"]) if m.get("latency_ms") else None))


def _compare_vllm(corpus, req: ChatReq, n: int, side: str = "both") -> dict:
    """Measured head-to-head on the PRODUCTION serve path: the control plane retrieves doc_ids + the
    same text RAG gets, then the vLLM engine serves both the resident-KV cart answer and the live RAG
    baseline. Every number here is measured on the deployment (latency, prefill tokens); $ is measured
    latency x the box's real $/hr. Recorded so the /demo page reflects real usage.

    `side` lets the UI run the two answers SEQUENTIALLY (cart first, render, then rag) instead of
    sitting through both generations before anything appears."""
    if req.doc_ids:
        # Pinned carts: the caller (or the cart half of a sequential pair) already resolved these
        # doc_ids, so retrieval runs ONCE per question — reuse them and rebuild the SAME text for the
        # RAG side. Mirrors the stream rag-side: client-supplied ids, so context_for validates them
        # against the corpus (unknown ids -> 400) rather than silently serving a smaller context.
        doc_ids = list(req.doc_ids)
        try:
            context = retrieval.context_for(corpus.id, doc_ids)
        except KeyError as e:
            raise HTTPException(400, str(e)) from e
    else:
        doc_ids, context = retrieval.retrieve_context(corpus.id, req.question, req.k)
        if not doc_ids:
            raise HTTPException(404, "no documents to retrieve for this document base")
    sources = retrieval.doc_sources(corpus.id, doc_ids)
    history = [m.model_dump() for m in req.history]

    enriched = []
    cm = rm = None
    if side in ("both", "cart"):
        cart = ml_client.inference_query(doc_ids, req.question, config.INFERENCE_MAX_TOKENS,
                                         history=history)
        cm = cart.get("metrics", {})
        cart_q = _price(cm)
        enriched.append(
            {"key": "everyday", "label": _LABELS["everyday"], "answer": cart.get("answer"),
             "latency_ms": cm.get("latency_ms"), "prompt_tokens": cm.get("prompt_tokens"),
             "cart_tokens": cm.get("resident_kv_tokens"), "gen_tokens": cm.get("gen_tokens"),
             "used_docs": doc_ids, "sources": sources, "feasible": True,
             "measured": cm.get("measured", True),
             "tier": "cartridge", "cost_per_query": round(cart_q, 6) if cart_q else None,
             "cost_per_month": round(cart_q * n, 2) if cart_q else None,
             "note": "resident KV — only the question is re-prefilled"})
    if side in ("both", "rag"):
        rag = ml_client.inference_rag(context, req.question, config.INFERENCE_MAX_TOKENS,
                                      history=history)
        rm = rag.get("metrics", {})
        rag_q = _price(rm)
        enriched.append(
            {"key": "rag", "label": _LABELS["rag"], "answer": rag.get("answer"),
             "latency_ms": rm.get("latency_ms"), "prompt_tokens": rm.get("prompt_tokens"),
             "gen_tokens": rm.get("gen_tokens"), "used_docs": doc_ids, "sources": sources,
             "feasible": True,
             "measured": rm.get("measured", True), "cost_per_query": round(rag_q, 6) if rag_q else None,
             "cost_per_month": round(rag_q * n, 2) if rag_q else None,
             "note": "re-prefills the retrieved context every query"})
    # Feed the demo's measured aggregate: one full record when both sides ran, cart-only otherwise
    # (rag-only calls are the second half of a sequential pair the cart call already recorded).
    # tenant_id attributes the query to the corpus's tenant for per-tenant billing.
    if side == "both":
        measurements.record(cm, rm, tenant_id=corpus.tenant_id)
    elif side == "cart":
        measurements.record(cm, None, tenant_id=corpus.tenant_id)

    summary = None
    if cm is not None and rm is not None:
        cart_q = _price(cm)
        rag_q = _price(rm)
        summary = {"cheaper_than_rag_x": _ratio(rag_q, cart_q),
                   "faster_than_rag_x": _ratio(rm.get("latency_ms"), cm.get("latency_ms"))}
    return {
        "strategies": enriched,
        "summary": summary,
        "corpus_tokens": corpus.corpus_tokens or 0,
        "queries_per_month": n, "k": req.k, "measured": True, "side": side,
        "measured_on": {"model": measurements.MODEL_LABEL, "instance": measurements.INSTANCE_LABEL},
    }


@router.post("/corpora/{corpus_id}/compare/stream")
@limiter.limit("30/minute")  # 2 GPU generations per question; cheap single-tenant DoS guard
def compare_stream(
    request: Request,
    corpus_id: str,
    req: ChatReq,
    side: str = Query("cart", pattern="^(cart|rag)$"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Token-streaming version of one compare side (SSE): a `head` event with sources, `delta`
    events as the model writes, then `summary` with measured metrics + $/query. The UI streams the
    Smart CAG side first, then RAG — modern-chatbot feel with the same honest measurement."""
    corpus = db.get(Corpus, corpus_id)
    if corpus is None or corpus.tenant_id != user.tenant_id:
        raise HTTPException(404, "Document base not found")
    if corpus.status != "ready" or config.INFERENCE_BACKEND != "vllm":
        raise HTTPException(400, "streaming compare needs a ready document base on the vLLM backend")
    history = [m.model_dump() for m in req.history]
    if side == "cart":
        # Resident-KV side: needs doc_ids only (never the raw text) — retrieval.retrieve
        # is one call; on the fused backend that's the question's ONLY GPU retrieval trip.
        doc_ids = retrieval.retrieve(corpus.id, req.question, req.k)
        if not doc_ids:
            raise HTTPException(404, "no documents to retrieve for this document base")
        url = f"{config.INFERENCE_SERVICE_URL}/query_stream"
        payload = {"doc_ids": doc_ids, "question": req.question,
                   "max_tokens": config.INFERENCE_MAX_TOKENS, "history": history}
    else:
        if req.doc_ids:
            # Reuse the cart side's retrieval: one retrieval per question, and both sides
            # see identical evidence by construction. The ids are client-echoed, so
            # context_for validates them against the corpus (unknown ids -> 400).
            doc_ids = list(req.doc_ids)
            try:
                context = retrieval.context_for(corpus.id, doc_ids)
            except KeyError as e:
                raise HTTPException(400, str(e)) from e
        else:  # rag side called standalone (direct API use) — retrieve as before
            doc_ids, context = retrieval.retrieve_context(corpus.id, req.question, req.k)
            if not doc_ids:
                raise HTTPException(404, "no documents to retrieve for this document base")
        url = f"{config.INFERENCE_SERVICE_URL}/rag_query_stream"
        payload = {"context": context, "question": req.question,
                   "max_tokens": config.INFERENCE_MAX_TOKENS, "history": history}
    sources = retrieval.doc_sources(corpus.id, doc_ids)
    _theta = config.ADAPTIVE_THETA
    theta = float(_theta) if _theta not in (None, "") else None

    def _forward(u: str, pl: dict, hold: dict):
        """Forward an ml-service SSE stream's delta lines; stash its final metrics in hold['m'] and
        SWALLOW its 'done' frame so the caller emits one synthesized terminal frame (carrying tier)."""
        with httpx.stream("POST", u, json=pl, timeout=300.0,
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

    def gen():
        yield ("data: " + json.dumps({"head": True, "side": side, "used_docs": doc_ids,
                                      "sources": sources}) + "\n\n")
        metrics_final = None
        conf = None
        tier = "cartridge" if side == "cart" else "rag"
        escalated = False
        try:
            hold = {"m": None}
            yield from _forward(url, payload, hold)      # cart-alone (or the rag baseline side)
            metrics_final = hold["m"]
            conf = (metrics_final or {}).get("confidence")
            # ADAPTIVE ROUTER (cart side only): if the cart-alone answer is under-confident, escalate
            # to the RAG backup — full retrieved context on the SAME engine — and flag it in-band so
            # the user always sees when the backup fired (the answer shown then IS the backup).
            if side == "cart" and theta is not None and conf is not None and conf < theta:
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
                    metrics_final = bhold["m"]           # shown answer = backup; report ITS measured cost
        except httpx.HTTPError as e:
            # head already went out with 200 -> report GPU-side failure IN-BAND, not a silent truncate.
            yield ("data: " + json.dumps({"error": f"inference service failed "
                                          f"({type(e).__name__})", "side": side}) + "\n\n")
            return
        except KeyError as e:            # context_for got an id not in the corpus
            yield ("data: " + json.dumps({"error": str(e), "side": side}) + "\n\n")
            return
        if metrics_final is not None:
            metrics_final = {**metrics_final, "tier": tier, "escalated": escalated,
                             "confidence": conf if side == "cart" else None}
            # one synthesized terminal frame (the UI finalizes + badges the tier off this)
            yield ("data: " + json.dumps({"done": True, "metrics": metrics_final}) + "\n\n")
            measurements.record(metrics_final if side == "cart" else None,
                                metrics_final if side == "rag" else None,
                                tenant_id=corpus.tenant_id)
            cost = _price(metrics_final)
            # measured_on rides the summary so the streaming UI labels the numbers as MEASURED
            # on this deployment — without it the footer falls back to the "modeled on the local
            # base model" text, which is wrong on the GPU stack where everything is measured.
            yield ("data: " + json.dumps({"summary": True, "side": side, "tier": tier,
                                          "escalated": escalated,
                                          "confidence": conf if side == "cart" else None,
                                          "cost_per_query": round(cost, 6) if cost else None,
                                          "measured_on": {"model": measurements.MODEL_LABEL,
                                                          "instance": measurements.INSTANCE_LABEL}})
                   + "\n\n")

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/corpora/{corpus_id}/compare")
@limiter.limit("30/minute")
def compare(
    request: Request,
    corpus_id: str,
    req: ChatReq,
    queries_per_month: int = Query(100_000, ge=1, le=100_000_000),
    side: str = Query("both", pattern="^(both|cart|rag)$"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    corpus = db.get(Corpus, corpus_id)
    if corpus is None or corpus.tenant_id != user.tenant_id:
        raise HTTPException(404, "Document base not found")
    if corpus.status != "ready":
        raise HTTPException(400, f"Document base not ready (status={corpus.status})")

    # Production serve path: measured head-to-head on the live vLLM engine (cart vs RAG baseline).
    if config.INFERENCE_BACKEND == "vllm":
        return _compare_vllm(corpus, req, queries_per_month, side=side)

    # HF path (local dev): the ml_service runs both and returns measured latency + token counts.
    out = ml_client.compare(str(storage.corpus_dir(corpus_id)), req.question, req.k)
    results = out.get("results", {})
    corpus_tokens = out.get("corpus_tokens") or corpus.corpus_tokens or 0
    n = queries_per_month
    cart_q = metrics.price_everyday()

    enriched = []
    for key in _ORDER:
        r = results.get(key, {})
        note = r.get("note")
        cost: float | None
        if key == "everyday":
            # Engram Smart CAG = the adaptive router: cartridge tier = the flat read-once marginal (0 raw
            # tokens); on the queries it escalates, RAG-priced on only the raw tokens it spent.
            cost = (cart_q if r.get("raw_tokens", 0) == 0
                    else metrics.price_rag(r.get("raw_tokens", 0), r.get("gen_tokens", 0), n))
        else:  # rag — re-prefills the retrieved docs every query
            cost = metrics.price_rag(r.get("prompt_tokens", 0), r.get("gen_tokens", 0), n)
        enriched.append({
            "key": key,
            "label": _LABELS[key],
            "answer": r.get("answer"),
            "latency_ms": r.get("latency_ms"),
            "prompt_tokens": r.get("prompt_tokens"),
            "gen_tokens": r.get("gen_tokens"),
            "feasible": r.get("feasible", True),
            "measured": r.get("measured", r.get("feasible", True)),
            "used_docs": r.get("used_docs"),
            # Engram Smart CAG's adaptive routing readout (None for RAG): which tier it picked + how sure.
            "cart_tokens": r.get("cart_tokens"),
            "tier": r.get("tier"),
            "confidence": r.get("confidence"),
            "raw_tokens": r.get("raw_tokens"),
            "cost_per_query": round(cost, 6) if cost is not None else None,
            "cost_per_month": round(cost * n, 2) if cost is not None else None,
            "note": note,
        })

    by = {e["key"]: e for e in enriched}
    ev = by["everyday"]  # = the adaptive cartridge router (the product)
    summary = {
        "cheaper_than_rag_x": _ratio(by["rag"]["cost_per_query"], ev["cost_per_query"]),
        "faster_than_rag_x": _ratio(by["rag"]["latency_ms"], ev["latency_ms"]),
    }
    return {
        "strategies": enriched,
        "summary": summary,
        "corpus_tokens": corpus_tokens,
        "queries_per_month": n,
        "k": out.get("k"),
    }


# --- Live scale test: a real server-side concurrency ramp against the vLLM engine ------------------
# The browser can't generate true concurrency (per-host connection cap + the SSM tunnel bottleneck),
# so the load is driven HERE, inside the VPC, straight at the Inference Service. For each concurrency
# level we fire N in-flight cart queries (resident-KV inject) and N RAG queries (re-prefill), read the
# server-measured ttft/latency, and stream one frame per level so the UI chart fills in live. RAG runs
# in the multi-tenant CHURNED-cache regime (a per-request nonce defeats the engine's prefix cache, the
# way other tenants' traffic would), which is the fleet reality the test is about.
def _scale_levels(maxc: int) -> list[int]:
    """Concurrency ramp for a run: start at the user's max and halve down to 1 — e.g. 24 -> [1, 3, 6,
    12, 24] — so higher-concurrency fleet tests run fewer, lighter levels and finish faster."""
    levels: list[int] = []
    c = max(1, maxc)
    while True:
        levels.append(c)
        if c == 1:
            break
        c //= 2
    return sorted(set(levels))


class ScaleTestReq(BaseModel):
    queries: list[str] = Field(default_factory=list, max_length=32)
    max_concurrency: int = Field(default=24, ge=1, le=24)


def _pct(vals: list[float], q: float) -> float | None:
    v = sorted(x for x in vals if x is not None)
    return round(v[min(len(v) - 1, int(q * len(v)))], 1) if v else None


def _run_level(arm: str, conc: int, reqs: int, payloads: list[dict], max_tokens: int) -> dict:
    """Fire `reqs` requests at concurrency `conc` for one arm against the STREAMING serve path and
    measure time-to-first-token + end-to-end latency harness-side. We stream (not the batch /query
    endpoints) because only the token stream exposes TTFT — the headline metric. Returns the measured
    aggregate for this level."""
    tasks = [payloads[i % len(payloads)] for i in range(reqs)]

    def fire(p: dict):
        if arm == "cart":
            u = f"{config.INFERENCE_SERVICE_URL}/query_stream"
            body = {"doc_ids": p["doc_ids"], "question": p["q"],
                    "max_tokens": max_tokens, "history": []}
        else:  # churned-cache RAG: unique prefix -> engine prefix-cache miss -> cold re-prefill
            ctx = f"[session {uuid.uuid4().hex[:8]}]\n{p['context']}"
            u = f"{config.INFERENCE_SERVICE_URL}/rag_query_stream"
            body = {"context": ctx, "question": p["q"],
                    "max_tokens": max_tokens, "history": []}
        t = time.perf_counter()
        ttft = None
        try:
            with httpx.stream("POST", u, json=body, timeout=300.0,
                              headers=ml_client._ml_headers()) as r:  # ML-plane shared-token auth (off by default)
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line.strip():
                        continue
                    if ttft is None and '"delta"' in line:   # first token out
                        ttft = (time.perf_counter() - t) * 1000.0
                    if '"done"' in line:
                        break
        except Exception:  # noqa: BLE001 — a dropped request just doesn't count toward throughput
            return None, None
        return ttft, (time.perf_counter() - t) * 1000.0      # (ttft_ms, latency_ms)

    t0 = time.perf_counter()
    ttfts: list[float] = []
    lats: list[float] = []
    with ThreadPoolExecutor(max_workers=conc) as ex:
        for tt, la in ex.map(fire, tasks):
            if la is not None:                    # completed request counts toward throughput
                lats.append(la)
                if tt is not None:
                    ttfts.append(tt)
    wall = time.perf_counter() - t0
    return {"qps": round(len(lats) / wall, 2) if wall > 0 else 0.0, "ok": len(lats), "reqs": reqs,
            "ttft": _pct(ttfts, 0.5), "ttft_p95": _pct(ttfts, 0.95), "lat": _pct(lats, 0.5)}


@router.post("/corpora/{corpus_id}/scale-test/stream")
@limiter.limit("6/minute")  # heavy: dozens of GPU generations per run
def scale_test_stream(
    request: Request,
    corpus_id: str,
    req: ScaleTestReq,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Stream a real concurrency ramp (SSE). Per level: {level, cart:{qps,ttft,lat,...}, rag:{...}}."""
    corpus = db.get(Corpus, corpus_id)
    if corpus is None or corpus.tenant_id != user.tenant_id:
        raise HTTPException(404, "Document base not found")
    if corpus.status != "ready" or config.INFERENCE_BACKEND != "vllm":
        raise HTTPException(400, "scale test needs a ready document base on the vLLM backend")
    queries = [q.strip() for q in (req.queries or []) if q.strip()][:16]
    if not queries:
        raise HTTPException(400, "provide at least one query")
    maxc = max(1, min(req.max_concurrency, 24))
    max_tokens = config.INFERENCE_MAX_TOKENS

    def gen():
        # retrieve each query's doc_ids + text ONCE (not per level), so the ramp measures serving.
        payloads: list[dict] = []
        for q in queries:
            try:
                ids, ctx = retrieval.retrieve_context(corpus.id, q, config.INFERENCE_TOPK)
            except Exception:  # noqa: BLE001
                continue
            if ids:
                payloads.append({"q": q, "doc_ids": ids, "context": ctx})
        if not payloads:
            yield "data: " + json.dumps({"error": "retrieval failed for the workload"}) + "\n\n"
            return
        levels = _scale_levels(maxc)
        yield "data: " + json.dumps({"start": True, "queries": len(payloads), "levels": levels}) + "\n\n"
        for c in levels:
            reqs = max(c * 2, 4)
            cart = _run_level("cart", c, reqs, payloads, max_tokens)
            rag = _run_level("rag", c, reqs, payloads, max_tokens)
            yield "data: " + json.dumps({"level": c, "cart": cart, "rag": rag}) + "\n\n"
        yield "data: " + json.dumps({"done": True}) + "\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --- Saved scale-test runs: persist a finished run so the tab can re-load past runs -----------------
_SCALE_RUNS_KEEP = 20  # most recent per corpus; older ones are pruned on save


def _scale_run_resp(r: ScaleRun) -> dict:
    return {"id": r.id, "corpus_id": r.corpus_id, "max_concurrency": r.max_concurrency,
            "n_queries": r.n_queries, "points": r.points or [], "created_at": r.created_at}


@router.post("/corpora/{corpus_id}/scale-runs", response_model=ScaleRunResp)
@limiter.limit("30/minute")
def save_scale_run(
    request: Request,
    corpus_id: str,
    body: ScaleRunSaveReq,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Persist a finished scale-test run (the tab POSTs its accumulated per-level points here on done)."""
    corpus = db.get(Corpus, corpus_id)
    if corpus is None or corpus.tenant_id != user.tenant_id:
        raise HTTPException(404, "Document base not found")
    if not body.points:
        raise HTTPException(400, "no points to save")
    run = ScaleRun(corpus_id=corpus_id, max_concurrency=body.max_concurrency,
                   n_queries=body.n_queries, points=body.points)
    db.add(run)
    # Keep only the most recent runs per corpus so saved history can't grow without bound.
    old = (db.query(ScaleRun).filter(ScaleRun.corpus_id == corpus_id)
           .order_by(ScaleRun.created_at.desc()).offset(_SCALE_RUNS_KEEP - 1).all())
    for r in old:
        db.delete(r)
    db.commit()
    db.refresh(run)
    return _scale_run_resp(run)


@router.get("/corpora/{corpus_id}/scale-runs", response_model=list[ScaleRunResp])
def list_scale_runs(
    corpus_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Past scale-test runs for this corpus, newest first (points included — runs are small)."""
    corpus = db.get(Corpus, corpus_id)
    if corpus is None or corpus.tenant_id != user.tenant_id:
        raise HTTPException(404, "Document base not found")
    runs = (db.query(ScaleRun).filter(ScaleRun.corpus_id == corpus_id)
            .order_by(ScaleRun.created_at.desc()).limit(_SCALE_RUNS_KEEP).all())
    return [_scale_run_resp(r) for r in runs]
