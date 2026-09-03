"""GPU-free, network-free self-test for the head-to-head harness.

`python bench/headtohead.py --selftest` runs this. It stands up an in-process Starlette stub that
serves FAKE SSE /query_stream + /rag_query_stream + /onboard_cag in the EXACT event shapes the real
vLLM Inference Service emits ({'delta': text} fragments, then {'done': True, 'metrics': {...}}), and
drives it through the SAME ServeClient / run_cell code path as a live run — but over httpx's
ASGITransport, so there is NO socket and NO network, and NO bm25s/fastembed (the retriever is
stubbed). It then asserts:
  * the stats math (qps = ok/wall, p50/p95 percentiles) is computed and shaped right;
  * the closed loop never exceeds `conc` in-flight requests (the stub records peak concurrency);
  * the nonce-churn plumbing works (rag_churn requests carry a per-request '[req-<uuid>]' prefix
    in the context; rag_hot requests do NOT);
  * the anchor-row output schema matches the estimator's per_model_anchor row exactly.

Exits 0 on success, 1 on any assertion failure — so it is a CI gate that needs no GPU box.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

# A path that will never exist — used to prove load_chunk_descs tolerates a missing sidecar.
BENCH_DIR_MISSING = Path(__file__).resolve().parent / "no_such_qrc_chunks_sidecar.json"


# --- stub retriever (no bm25s / fastembed) ---------------------------------------------------
class _StubRetriever:
    """Returns the first `k` doc_ids and a joined context — enough to exercise the request path and
    the churn-nonce plumbing without importing the real (heavy) retriever. route_chunks_context is
    stubbed too (the qrc arm calls it) — it returns a joined slice of the non-top docs' text so the
    qrc request carries a non-empty context, without pulling in bm25s/fastembed/chunking."""

    def __init__(self):
        self.doc_ids = [f"stubdoc_{i}" for i in range(5)]
        self.texts = {d: f"Stub document {d}. The stub fact is 42." for d in self.doc_ids}
        # Fake fused scores (descending) so dynamic_k has something to cut: the top doc scores 1.0, each
        # next 0.5x the last, so at ratio 0.6 only the top survives (a stable, checkable dynamic set).
        self._scores = {d: 1.0 * (0.5 ** i) for i, d in enumerate(self.doc_ids)}

    # `_by_id` mirrors HybridRetriever's id->text map — the resident arm reads it for doc_texts/titles.
    @property
    def _by_id(self):
        return self.texts

    def retrieve_context(self, question: str, k: int):
        ids = self.doc_ids[:k]
        return ids, "\n\n".join(self.texts[d] for d in ids)

    def dynamic_k(self, question, k_max, ratio):
        # Mirror HybridRetriever.dynamic_k over the fake scores: keep the top, then each doc while its
        # score >= ratio * top, capped at k_max, always >= 1.
        threshold = ratio * self._scores[self.doc_ids[0]]
        kept = [self.doc_ids[0]]
        for d in self.doc_ids[1:]:
            if len(kept) >= k_max:
                break
            if self._scores[d] >= threshold:
                kept.append(d)
            else:
                break
        return kept

    def route_chunks_context(self, question, doc_ids, budget_tokens=None, descs_by_doc=None):
        # Mirror HybridRetriever.route_chunks_context's contract shape: empty doc_ids -> '' (qrc
        # degenerates to cart at topk=1), else the non-top docs' text joined as the routed context.
        if not doc_ids:
            return ""
        return "\n\n".join(self.texts.get(d, "") for d in doc_ids)

    def route_chunk_spans(self, question, doc_ids, budget_tokens=None, descs_by_doc=None):
        # Mirror HybridRetriever.route_chunk_spans: {doc_id: [[start,end)]} for the non-top docs. Empty
        # doc_ids -> {} (resident degenerates to a pure top-1 cart serve). One span covering the whole
        # stub text is enough to exercise the payload plumbing.
        return {d: [[0, len(self.texts.get(d, ""))]] for d in doc_ids if self.texts.get(d)}


# --- stub SSE server (Starlette ASGI app, driven in-process via httpx ASGITransport) ---------
def _build_stub_app():
    """A Starlette app mirroring the service's SSE contract. Tracks peak concurrency (to prove the
    client's closed loop is bounded) and whether each RAG request's context carried a churn nonce."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Route

    state = {"in_flight": 0, "peak": 0, "rag_churn_nonce_seen": 0, "rag_hot_nonce_seen": 0,
             "onboard_docs": 0}
    lock = asyncio.Lock()
    nonce_re = re.compile(r"^\[req-[0-9a-f]{32}\]\n")

    async def _enter():
        async with lock:
            state["in_flight"] += 1
            state["peak"] = max(state["peak"], state["in_flight"])

    async def _leave():
        async with lock:
            state["in_flight"] -= 1

    async def _sse(n_frags: int):
        """Yield `n_frags` {'delta'} fragments then a {'done', metrics} frame — the service shape.
        The answer text contains '42' so the harness's spot-check (expect_substring) can pass."""
        # A per-fragment sleep makes the closed loop observable (peak in-flight must equal conc,
        # never exceed it) AND keeps the cell wall well above the 0.1s wall_s rounding quantum, so
        # the qps==ok/wall check isn't fooled by rounding at sub-100ms walls.
        yield b"data: " + json.dumps({"delta": "The answer is "}).encode() + b"\n\n"
        await asyncio.sleep(0.05)
        for _ in range(max(0, n_frags - 1)):
            yield b"data: " + json.dumps({"delta": "42 "}).encode() + b"\n\n"
            await asyncio.sleep(0.02)
        metrics = {"latency_ms": 10.0, "ttft_ms": 2.0, "prompt_tokens": 7,
                   "resident_kv_tokens": 0, "gen_tokens": n_frags, "measured": True}
        yield b"data: " + json.dumps({"done": True, "metrics": metrics}).encode() + b"\n\n"

    async def query_stream(request):
        await _enter()
        body = await request.json()
        # QRC hybrid: /query_stream accepts an optional `context` (routed real-text chunks of the
        # non-top docs) alongside doc_ids; QRC RESIDENT instead sends `doc_spans` (per non-top doc, the
        # loadable char spans) + doc_texts/doc_titles and NO context. Record per request: ctx flag,
        # doc_ids count, and the spanned-id set — so the selftest can assert qrc_res spans EXACTLY the
        # non-top docs with no context, and qrc records its per-question k (doc_ids length).
        state.setdefault("qs_ctx_flags", []).append(bool(body.get("context", "")))
        state.setdefault("qs_ndocs", []).append(len(body.get("doc_ids", [])))
        state.setdefault("qs_doc_ids", []).append(list(body.get("doc_ids", [])))
        state.setdefault("qs_span_ids", []).append(sorted((body.get("doc_spans") or {}).keys()))
        try:
            n = min(int(body.get("max_tokens", 8)), 8)
            return StreamingResponse(_sse(n), media_type="text/event-stream",
                                     background=_Bg(_leave))
        except Exception:  # noqa: BLE001
            await _leave()
            raise

    async def rag_query_stream(request):
        await _enter()
        body = await request.json()
        ctx = body.get("context", "")
        # Churn plumbing check: rag_churn prefixes '[req-<uuid>]\n'; rag_hot does not. We can't see
        # the arm name here, so we just count how many requests carried a nonce — the selftest
        # asserts the split by running the two arms in separate cells and reading the deltas.
        if nonce_re.match(ctx):
            state["_last_had_nonce"] = True
        else:
            state["_last_had_nonce"] = False
        state.setdefault("nonce_flags", []).append(bool(nonce_re.match(ctx)))
        try:
            n = min(int(body.get("max_tokens", 8)), 8)
            return StreamingResponse(_sse(n), media_type="text/event-stream",
                                     background=_Bg(_leave))
        except Exception:  # noqa: BLE001
            await _leave()
            raise

    async def onboard_cag(request):
        body = await request.json()
        docs = body.get("docs", [])
        state["onboard_docs"] += len(docs)
        return JSONResponse({"n_cartridges": len(docs), "n_built": len(docs),
                             "corpus_tokens": sum(len(d.get("text", "")) for d in docs),
                             "cart_seconds": 0.0, "train_seconds": 0.0, "errors": {},
                             "canceled": False, "method": "stub"})

    app = Starlette(routes=[
        Route("/query_stream", query_stream, methods=["POST"]),
        Route("/rag_query_stream", rag_query_stream, methods=["POST"]),
        Route("/onboard_cag", onboard_cag, methods=["POST"]),
    ])
    app.state.stub = state
    return app, state


class _Bg:
    """Starlette BackgroundTask-compatible callable that decrements in-flight AFTER the stream body
    is fully sent — so peak concurrency reflects the real overlap window."""

    def __init__(self, fn):
        self.fn = fn

    async def __call__(self):
        await self.fn()


def _asgi_client(app):
    """An httpx.AsyncClient bound to the ASGI app via ASGITransport — in-process, no socket."""
    import httpx
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://stub", timeout=30.0)


async def _run() -> int:
    from bench import headtohead as h2h

    app, state = _build_stub_app()
    client = _asgi_client(app)
    serve = h2h.ServeClient("http://stub", "http://stub", timeout_s=30.0, client=client)
    retriever = _StubRetriever()
    questions = [{"q": "what is the stub fact?", "expect_substring": "42", "doc_id": "stubdoc_0"},
                 {"q": "another question?", "expect_substring": "42", "doc_id": "stubdoc_1"}]

    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)
            print(f"[selftest] FAIL: {msg}")
        else:
            print(f"[selftest] ok: {msg}")

    # --- onboard plumbing ---
    docs = [{"doc_id": "d0", "text": "hello world"}, {"doc_id": "d1", "text": "second doc"}]
    resp = await serve.onboard("/data/bench_corpus", docs)
    check(resp["n_cartridges"] == 2, "onboard returns n_cartridges for the batch")
    check(state["onboard_docs"] == 2, "onboard_cag received the docs")

    # --- tiny sweep: conc 1,2, 6 reqs per cell, all arms (cart, qrc, qrc_res, qrc_dyn, rag_churn,
    # rag_hot) — the QRC family covers the hybrid-text, resident-span, and dynamic-k paths.
    all_cells: list[dict] = []
    for conc in (1, 2):
        state["peak"] = 0  # reset the concurrency high-water mark per cell group
        for arm in ("cart", "qrc", "qrc_res", "qrc_dyn", "rag_churn", "rag_hot"):
            state["nonce_flags"] = []
            state["qs_ctx_flags"] = []
            state["qs_ndocs"] = []
            state["qs_doc_ids"] = []
            state["qs_span_ids"] = []
            cell = await h2h.run_cell(serve, retriever, arm, conc, questions,
                                      gen_tokens=6, topk=3, zipf_s=1.1, seed=conc,
                                      reqs_override=6, dynk_ratio=0.6)
            all_cells.append(cell)
            # closed-loop bound: peak in-flight during this cell must never exceed conc.
            check(state["peak"] <= conc,
                  f"closed loop bounded at conc={conc} (peak seen {state['peak']})")
            # nonce plumbing: churn arm sends a nonce on every request; hot arm sends none.
            if arm == "rag_churn":
                check(state["nonce_flags"] and all(state["nonce_flags"]),
                      "rag_churn context carries a per-request nonce (prefix cache can never hit)")
            if arm == "rag_hot":
                check(state["nonce_flags"] and not any(state["nonce_flags"]),
                      "rag_hot context carries NO nonce (prefix cache free to hit)")
            # qrc plumbing (topk=3): each qrc request hits /query_stream with EXACTLY one resident
            # cart (the top doc) and a NON-empty routed context (docs 2..k's chunks). The cart arm
            # hits /query_stream with the doc_ids and NO context.
            if arm == "qrc":
                check(state["qs_ctx_flags"] and all(state["qs_ctx_flags"]),
                      "qrc /query_stream carries routed context (docs 2..k) at topk=3")
                check(state["qs_ndocs"] and all(n == 1 for n in state["qs_ndocs"]),
                      "qrc serves exactly ONE resident cart (doc_ids=[top]) per request")
            if arm == "cart":
                check(state["qs_ctx_flags"] and not any(state["qs_ctx_flags"]),
                      "cart /query_stream carries NO context (resident-KV only)")
            # qrc_res plumbing (topk=3): the resident arm serves ALL doc_ids (top full-cart + docs 2..k
            # as spans) with NO context, and doc_spans covers EXACTLY the non-top docs (stubdoc_1/2).
            if arm == "qrc_res":
                non_top = sorted(retriever.doc_ids[1:3])
                check(state["qs_ctx_flags"] and not any(state["qs_ctx_flags"]),
                      "qrc_res /query_stream carries NO context (docs 2..k load as KV spans)")
                check(state["qs_ndocs"] and all(n == 3 for n in state["qs_ndocs"]),
                      "qrc_res serves ALL doc_ids (top full-cart + docs 2..k as spans) at topk=3")
                check(state["qs_span_ids"] and all(s == non_top for s in state["qs_span_ids"]),
                      "qrc_res sends doc_spans for EXACTLY the non-top docs (docs 2..k)")
            # qrc_dyn plumbing: the arm records a per-question k, and at ratio 0.6 the stub keeps only
            # the top doc (score 1.0; next 0.5 < 0.6), so k=1 and no spans go out (nothing to load).
            if arm == "qrc_dyn":
                check("mean_k" in cell, "qrc_dyn cell records mean_k (per-question docs kept)")
                check(cell.get("mean_k") == 1.0,
                      f"qrc_dyn keeps k=1 at ratio 0.6 over the stub scores (mean_k={cell.get('mean_k')})")
                check(state["qs_span_ids"] and all(s == [] for s in state["qs_span_ids"]),
                      "qrc_dyn sends NO spans when dynamic-k keeps only the top doc")
            # stats math + row schema.
            check(cell["reqs"] == 6, f"reqs honored (6) for the selftest cell arm={arm}")
            check(cell["ok"] == 6, f"all 6 requests ok for arm={arm} conc={conc}")
            check(cell["errors"] == 0, f"no errors for arm={arm} conc={conc}")
            # qps is ok/wall computed on the TRUE wall; the row exposes wall_s rounded to 1 decimal.
            # Recomputing ok/wall_s can only match within the rounding quantum: wall_s is within
            # +/-0.05s of the true wall, so qps and ok/wall_s bracket the same true value. Assert
            # they agree within the band that +/-0.05s on wall implies (exact, not a guessed pct).
            if cell["wall_s"] > 0:
                w = cell["wall_s"]
                lo = cell["ok"] / (w + 0.05)
                hi = cell["ok"] / max(w - 0.05, 1e-6)
                check(lo - 1e-6 <= cell["qps"] <= hi + 1e-6,
                      f"qps == ok/true_wall for arm={arm} conc={conc} "
                      f"(qps={cell['qps']}, band [{lo:.3f},{hi:.3f}] from wall_s={w})")
            check(cell["ttft_p50"] >= 0 and cell["e2e_p50"] >= cell["ttft_p50"],
                  f"e2e_p50 >= ttft_p50 for arm={arm} conc={conc}")
            check(cell["ttft_p95"] >= cell["ttft_p50"],
                  f"p95 >= p50 for arm={arm} conc={conc}")
            check(cell["spot_check"] == "4/4",
                  f"fact spot-check passes 4/4 (answer contains '42') for arm={arm} conc={conc}")

    # --- percentile helper correctness (independent of the server) ---
    check(h2h.pctl([10.0], 50) == 10.0, "pctl of a singleton returns it")
    check(h2h.pctl([1.0, 2.0, 3.0, 4.0], 50) == 2.5,
          "pctl inclusive-method p50 of [1,2,3,4] == 2.5")
    check(h2h.pctl([], 50) == 0.0, "pctl of empty list is 0.0")

    # --- anchor-row schema ---
    rows = h2h.anchor_rows(all_cells)
    check(len(rows) == 2, "one anchor row per conc (2)")
    expected_keys = {"model", "instance", "conc", "cart_qps", "cart_ttft_ms", "cart_e2e_ms",
                     "rag_qps", "rag_ttft_ms", "rag_e2e_ms"}
    check(all(set(r) == expected_keys for r in rows),
          "anchor row schema == per_model_anchor exactly")
    check(all(r["model"] == "Command-A-Plus" and r["instance"] == "lambda_2x_h100_sxm5"
              for r in rows), "anchor rows carry the estimator model/instance identity")
    # anchor uses the rag_CHURN arm, never rag_hot (qrc is a separate arm, also out of the anchor).
    churn = {c["conc"]: c for c in all_cells if c["arm"] == "rag_churn"}
    check(all(r["rag_ttft_ms"] == churn[r["conc"]]["ttft_p50"] for r in rows),
          "anchor rag_* comes from the rag_churn arm's p50 (anchor-consistent)")

    # --- qrc degenerates to the cart arm at topk=1 (empty routed context) ---
    state["qs_ctx_flags"] = []
    q1 = await h2h.one_request(serve, retriever, "qrc", questions[0], gen_tokens=6, topk=1)
    check(q1["error"] is None and state["qs_ctx_flags"] == [False],
          "qrc at topk=1 sends NO context (degenerates to the cart arm)")

    # --- chunkdesc parse path (the shared QRC core, no GPU) ---
    chunking = h2h._load_chunking()
    doc_text = ("Alpha beta gamma delta. " * 20 + "The stub fact is 42. ") + ("Epsilon zeta. " * 20)
    n_chunks = len(chunking.chunk_spans(doc_text))
    prompt = chunking.chunk_desc_prompt(doc_text)
    check("chunk" in prompt.lower() and str(n_chunks) in prompt,
          "chunk_desc_prompt lists every chunk by ordinal")
    # A well-formed numbered reply parses to one desc per chunk...
    good_reply = "\n".join(f"{i + 1}. describes chunk {i + 1}" for i in range(n_chunks))
    descs = chunking.parse_chunk_descs(good_reply, n_chunks)
    check(len(descs) == n_chunks and all(descs),
          "parse_chunk_descs maps a well-formed numbered reply to one desc per chunk")
    # ...and a garbled reply degrades to empty descs (never fatal — routing falls back to text).
    bad = chunking.parse_chunk_descs("(model produced no numbered lines)", n_chunks)
    check(len(bad) == n_chunks and not any(bad),
          "parse_chunk_descs degrades a garbled reply to all-empty descs (never fatal)")
    # The sidecar loader tolerates a missing file (qrc then routes on chunk text alone).
    check(h2h.load_chunk_descs(BENCH_DIR_MISSING) is None,
          "load_chunk_descs returns None for a missing sidecar (qrc routes on chunk text)")

    await client.aclose()

    print(f"\n[selftest] {len(failures)} failure(s).")
    if failures:
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[selftest] PASS — stats math, closed-loop concurrency, nonce-churn plumbing, and "
          "output schema all verified with NO GPU and NO network.")
    return 0


def run_selftest() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(run_selftest())
