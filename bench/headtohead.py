"""Head-to-head benchmark harness: Engram cart serving vs traditional RAG.

SAME model, SAME vLLM engine, SAME production-grade retrieval, on the live GPU box (localhost).
This driver runs ON the box in a light venv and drives the running vLLM Inference Service
(ml_service/vllm_inference.py) over HTTP: /onboard_cag, /query_stream (cart arm),
/rag_query_stream (RAG arm). It ports bench_fleet.py's MEASUREMENT DEFINITIONS exactly so the
numbers seed the cost estimator's per-model anchor rows on the same footing as the established
fleet probe.

Arms (bench_fleet parity):
  cart      — retrieve top-k doc_ids (timed, INSIDE the request e2e) then /query_stream with
              those doc_ids: the resident-KV serve path, no per-query document prefill.
  qrc       — hybrid QRC serving: SAME retrieval, then the TOP doc serves as the resident cart and
              docs 2..k route their query-selected real-text CHUNKS in as `context` on /query_stream
              (doc_ids=[top], context=routed chunks). This is the fix for k=3 multi-cart interference:
              one proven-solo resident cart + small real-token context instead of composed carts. At
              topk=1 the routed context is empty, so qrc degenerates to the cart arm. Optionally uses
              the chunkdesc sidecar (qrc_chunks.json) to sharpen chunk routing.
  rag_churn — SAME retrieval, then /rag_query_stream with the top-k docs' full text as `context`,
              prefixed with a per-request unique nonce line "[req-<uuid>]\\n". The RagReq endpoint
              has NO cache-salt field (checked: ml_service/vllm_inference.py RagReq = context,
              question, max_tokens, history — salt is only the process-level SERVE_CACHE_SALT env,
              applied to BOTH arms if set, so it can't force per-request churn). The nonce forces
              honest churn: real RAG at production corpus scale re-prefills every query because the
              corpus >> KV pool, so the prefix cache never hits. This is the estimator anchor's
              "churned-prefix-cache RAG" and mirrors bench_fleet's rag_churn (cache_salt=rid) arm.
  rag_hot   — same as rag_churn but NO nonce (prefix cache free to hit); reported so the claim
              survives the "but prefix caching!" objection with data.

bench_fleet measurement definitions ported verbatim:
  * closed loop at exactly `conc` in-flight via asyncio.Semaphore;
  * requests per cell = max(32, 3*conc);
  * a DISCARDED warmup of min(conc, 8) requests per cell before measurement;
  * TTFT = first streamed token (first SSE 'delta'); e2e = final ('done');
  * greedy decoding (server temperature=0.0), fixed --gen-tokens both arms;
  * Zipf(s) question sampling seeded per cell (reproducible);
  * a 4-request fact spot-check per cell (answer must contain expect_substring) so we never
    measure confabulation speed;
  * the ANCHOR-row statistic is p50 (median) for ttft/e2e and wall-clock qps — matched to
    bench_fleet's _excel_rows, which emits ttft_p50 / lat_p50 (NOT the mean).

Phases:  onboard | chunkdesc | sweep | accuracy   (plus --selftest, GPU-free, network-free).
  chunkdesc builds the QRC chunk-description sidecar (qrc_chunks.json) — one cart-resident generation
  per corpus doc describing its chunks — measuring the onboarding-cost delta of descriptions.

Config (env/flags): ML_AUTH_TOKEN (env, never printed), --serve-url / --onboard-url,
  --gen-tokens (128), --topk (3), --seed, --conc-list, --arms.

Install hint for the box venv:
  python -m pip install httpx bm25s fastembed numpy
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import signal
import statistics
import sys
import time
import uuid
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results"
# The chunk-description sidecar the chunkdesc phase writes and the qrc arm reads: {doc_id: [desc,...]}
# parallel to chunking.chunk_spans(doc_text). Optional — qrc routes on chunk text alone when absent.
CHUNK_DESCS_PATH = BENCH_DIR / "qrc_chunks.json"


def _load_chunking():
    """The shared QRC core (backend/app/chunking.py), imported robustly: a bare `import chunking`
    on the box (where it is synced beside the bench) or, in the repo layout, loaded from backend/app
    BY FILE PATH — deliberately not via sys.path, because that dir holds app modules (email.py,
    config.py) that would SHADOW stdlib/other packages. Deferred into a function so a phase that never
    touches QRC (onboard/sweep-without-qrc) doesn't pay the import — and so the module imports cleanly
    on a box that hasn't synced chunking.py yet."""
    try:
        import chunking  # box layout: chunking.py beside the bench
        return chunking
    except ImportError:
        import importlib.util as ilu
        path = BENCH_DIR.parent / "backend" / "app" / "chunking.py"
        spec = ilu.spec_from_file_location("chunking", path)
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def load_chunk_descs(path: Path = CHUNK_DESCS_PATH) -> dict | None:
    """The chunk-description sidecar {doc_id: [desc,...]} if present, else None (qrc then routes on
    chunk text alone — a missing/partial sidecar can only fail to help routing, never break it)."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[bench] WARNING: could not read chunk-desc sidecar {path}: {exc} — "
              f"qrc will route on chunk text alone", flush=True)
        return None


# ------------------------------------------------------------------ defaults (bench_fleet parity)
DEFAULT_CONC = [1, 8, 32, 64, 128]
DEFAULT_ARMS = ["cart", "rag_churn", "rag_hot"]
# Anchor rows use the rag_CHURN arm (anchor-consistent); rag_hot stays in the raw JSON only.
ANCHOR_RAG_ARM = "rag_churn"
DEFAULT_GEN_TOKENS = 128
DEFAULT_TOPK = 3
DEFAULT_ZIPF_S = 1.1
# rag at conc 128 can take MINUTES (the 30B anchor showed 68s e2e; command-a-plus is bigger) —
# a generous per-request timeout so a slow churned prefill is counted, not aborted.
DEFAULT_REQ_TIMEOUT_S = 360.0
# Estimator anchor identity (from the brief — the cost estimator's per_model_anchor row schema).
ANCHOR_MODEL = "Command-A-Plus"
ANCHOR_INSTANCE = "lambda_2x_h100_sxm5"

# Between arms/cells: a short settle sleep + a drain, so one cell's tail requests don't bleed into
# the next cell's wall clock (bench_fleet drains via gather; we add an explicit settle for the HTTP
# path where the server's queue can lag the client's gather completion).
SETTLE_S = 1.0

# Per-arm seed offsets so each (arm, conc) cell samples a reproducible-but-distinct question
# stream (bench_fleet._ARM_SEED).
_ARM_SEED = {"cart": 1, "rag_churn": 2, "rag_hot": 3, "qrc": 4}


# ------------------------------------------------------------------ stats (bench_fleet parity)
def pctl(xs: list[float], p: float) -> float:
    """method='inclusive' percentile — interpolates within [min,max] (the default 'exclusive'
    method extrapolates BEYOND the observed extremes, nonsense for latency tails). Verbatim
    bench_fleet._pctl."""
    if not xs:
        return 0.0
    if len(xs) < 2:
        return round(xs[0], 1)
    return round(statistics.quantiles(xs, n=100, method="inclusive")[int(p) - 1], 1)


def zipf_pick(rng: random.Random, n: int, s: float) -> int:
    """Zipf(s) index over [0,n) — bench_fleet._zipf_pick verbatim (a few docs get most traffic,
    the realistic skew)."""
    w = [1.0 / (k + 1) ** s for k in range(n)]
    return rng.choices(range(n), weights=w, k=1)[0]


# ------------------------------------------------------------------ corpus / questions I/O
def load_corpus(path: Path) -> tuple[list[str], list[str]]:
    """(doc_ids, texts) from corpus.jsonl. Order preserved (the retriever indexes this order)."""
    doc_ids: list[str] = []
    texts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        doc_ids.append(rec["doc_id"])
        texts.append(rec["text"])
    return doc_ids, texts


def load_questions(path: Path) -> list[dict]:
    """Answerable question slots (q non-empty) from questions.json. Blank TEMPLATE slots are
    skipped with a warning — a sweep/accuracy run needs real questions to spot-check, and a blank
    'q' would measure nothing. onboard needs no questions."""
    qs = json.loads(path.read_text(encoding="utf-8"))
    filled = [q for q in qs if q.get("q", "").strip()]
    blank = len(qs) - len(filled)
    if blank:
        print(f"[bench] WARNING: {blank} question slot(s) are BLANK templates and were skipped "
              f"(run prep_corpus.py's note: hand-write them). Using {len(filled)} real question(s).",
              flush=True)
    if not filled:
        raise SystemExit("no answerable questions in questions.json — hand-write them first "
                         "(see prep_corpus.py's loud note), then re-run.")
    return filled


# ------------------------------------------------------------------ HTTP client
def _auth_headers() -> dict:
    """Bearer ML_AUTH_TOKEN on every route (the vLLM service enforces it when set). The token is
    read from env and NEVER printed/logged/persisted."""
    tok = os.environ.get("ML_AUTH_TOKEN", "")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


class ServeClient:
    """Thin async httpx client for the vLLM Inference Service. Streams the SSE arms and measures
    TTFT (first 'delta') / e2e (final 'done') — the same event shapes the service emits
    (_stream_generate: {'delta': text} per fragment, then {'done': True, 'metrics': {...}})."""

    def __init__(self, serve_url: str, onboard_url: str, timeout_s: float, client=None):
        import httpx
        self._httpx = httpx
        self.serve_url = serve_url.rstrip("/")
        self.onboard_url = onboard_url.rstrip("/")
        # A generous read timeout for the churned-RAG tail; connect/write stay short.
        self._timeout = httpx.Timeout(timeout_s, connect=10.0)
        # `client` is injectable so --selftest can pass an ASGITransport-backed AsyncClient (drives
        # a Starlette stub in-process — NO socket, NO network). Production builds a real one.
        self._client = client or httpx.AsyncClient(timeout=self._timeout, headers=_auth_headers())

    async def aclose(self) -> None:
        await self._client.aclose()

    async def onboard(self, corpus_dir: str, docs: list[dict]) -> dict:
        """POST /onboard_cag (non-stream). Returns the service's response dict
        (n_cartridges, train_seconds, cart_seconds, corpus_tokens, errors, ...)."""
        r = await self._client.post(f"{self.onboard_url}/onboard_cag",
                                    json={"corpus_dir": corpus_dir, "docs": docs})
        r.raise_for_status()
        return r.json()

    async def stream(self, path: str, payload: dict) -> dict:
        """Drive one SSE request. Returns {ttft_ms, e2e_ms, text, metrics, error}. TTFT = time to
        the first non-empty 'delta'; e2e = time to the final 'done' (or last byte). Any transport
        error is captured (counted, not fatal) per the robustness requirement."""
        url = f"{self.serve_url}{path}"
        t0 = time.perf_counter()
        t_first: float | None = None
        text_parts: list[str] = []
        metrics: dict | None = None
        error: str | None = None
        try:
            async with self._client.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")[:200]
                    return {"ttft_ms": None, "e2e_ms": (time.perf_counter() - t0) * 1000,
                            "text": "", "metrics": None,
                            "error": f"HTTP {resp.status_code}: {body}"}
                async for raw in resp.aiter_lines():
                    if not raw.startswith("data:"):
                        continue
                    try:
                        evt = json.loads(raw[len("data:"):].strip())
                    except ValueError:
                        continue
                    if "delta" in evt:
                        if t_first is None and evt["delta"]:
                            t_first = time.perf_counter()
                        text_parts.append(evt["delta"])
                    elif evt.get("error"):
                        error = f"{evt.get('error')}: {evt.get('reason') or evt.get('detail')}"
                    elif evt.get("done"):
                        metrics = evt.get("metrics")
        except Exception as exc:  # noqa: BLE001 — network/timeout failures are counted, not fatal
            error = f"{type(exc).__name__}: {exc}"
        t1 = time.perf_counter()
        return {"ttft_ms": (t_first - t0) * 1000 if t_first else None,
                "e2e_ms": (t1 - t0) * 1000, "text": "".join(text_parts),
                "metrics": metrics, "error": error}


# ------------------------------------------------------------------ one request (arm-specific)
async def one_request(client: ServeClient, retriever, arm: str, q: dict, gen_tokens: int,
                      topk: int, descs_by_doc: dict | None = None) -> dict:
    """One measured request for `arm`. Retrieval is TIMED and counted INSIDE the request e2e (the
    honest end-to-end a user sees: retrieve then generate) AND reported separately as retrieval_ms.
    cart -> /query_stream with the retrieved doc_ids; qrc -> /query_stream with the TOP doc as the
    resident cart plus docs[1:]' query-routed real-text chunks as `context` (the hybrid serve path);
    rag_churn/rag_hot -> /rag_query_stream with the retrieved docs' text as context (churn arm
    prefixes a per-request nonce so the prefix cache can never hit). `descs_by_doc` (qrc only) is the
    onboarding chunk-description sidecar {doc_id: [desc,...]}, folded into chunk INDEX text to sharpen
    routing (None -> route on chunk text alone). Returns a per-request record with
    ttft/e2e/retrieval_ms/text/expect/error."""
    t_retr = time.perf_counter()
    doc_ids, context = retriever.retrieve_context(q["q"], topk)
    if arm == "qrc":
        # The TOP doc serves as the resident cart; the rest route their answer-bearing chunks in as
        # real-token context. At topk=1 doc_ids[1:] is empty -> context '' -> the qrc arm degenerates
        # to the pure cart arm (identical request), which is the intended edge behavior.
        context = retriever.route_chunks_context(q["q"], doc_ids[1:], descs_by_doc=descs_by_doc)
    retrieval_ms = (time.perf_counter() - t_retr) * 1000
    if arm == "cart":
        res = await client.stream("/query_stream", {
            "doc_ids": doc_ids, "question": q["q"], "max_tokens": gen_tokens})
    elif arm == "qrc":
        res = await client.stream("/query_stream", {
            "doc_ids": [doc_ids[0]], "context": context, "question": q["q"],
            "max_tokens": gen_tokens})
    else:
        if arm == "rag_churn":
            # NO per-request cache-salt field exists on RagReq (checked), so force honest churn
            # with a per-request unique nonce line — the prefix cache can never hit (mirrors
            # bench_fleet's rag_churn cache_salt=rid).
            context = f"[req-{uuid.uuid4().hex}]\n{context}"
        res = await client.stream("/rag_query_stream", {
            "context": context, "question": q["q"], "max_tokens": gen_tokens})
    # The retrieval wall is part of the user-visible e2e; add it so e2e reflects retrieve+generate.
    e2e_ms = (res["e2e_ms"] or 0.0) + retrieval_ms
    ttft_ms = (res["ttft_ms"] + retrieval_ms) if res["ttft_ms"] is not None else None
    return {"arm": arm, "ttft_ms": ttft_ms, "e2e_ms": e2e_ms, "retrieval_ms": retrieval_ms,
            "text": res["text"], "expect": q.get("expect_substring", ""),
            "doc_ids": doc_ids, "error": res["error"]}


# ------------------------------------------------------------------ one cell (closed loop)
async def run_cell(client: ServeClient, retriever, arm: str, conc: int, questions: list[dict],
                   gen_tokens: int, topk: int, zipf_s: float, seed: int,
                   reqs_override: int | None = None, descs_by_doc: dict | None = None) -> dict:
    """One (arm, conc) measurement cell: closed loop at exactly `conc` in-flight, reqs=max(32,3*conc),
    a discarded warmup of min(conc,8), Zipf(s) question sampling seeded per cell. Emits the per-cell
    row (qps from wall over completed reqs, ttft/e2e p50 + p95, retrieval_ms p50, errors, and the
    4-request fact spot-check). Verbatim port of bench_fleet._run_cell adapted to the HTTP arms.
    reqs_override is ONLY for --selftest (a tiny cell); production always uses max(32,3*conc).
    descs_by_doc (qrc arm only) is the chunk-description sidecar passed into routing."""
    reqs = reqs_override if reqs_override is not None else max(32, 3 * conc)
    rng = random.Random(seed)
    sem = asyncio.Semaphore(conc)
    results: list[dict] = []

    async def worker(measured: bool) -> None:
        async with sem:
            q = questions[zipf_pick(rng, len(questions), zipf_s)]
            r = await one_request(client, retriever, arm, q, gen_tokens, topk, descs_by_doc)
            if measured:
                results.append(r)

    # Discarded warmup (min(conc,8)) — brings the server's per-arm state warm before timing.
    await asyncio.gather(*(worker(False) for _ in range(min(conc, 8))))
    t0 = time.perf_counter()
    await asyncio.gather(*(worker(True) for _ in range(reqs)))
    wall = time.perf_counter() - t0

    ok = [r for r in results if r["error"] is None and r["ttft_ms"] is not None]
    errors = [r for r in results if r["error"] is not None]
    # 4-request fact spot-check: on the FIRST 4 ok results, the answer must contain expect_substring
    # (skip requests whose question has no expect string). Guards against measuring confabulation.
    spot = [r for r in ok if r["expect"]][:4]
    spot_pass = sum(1 for r in spot if r["expect"] in r["text"])
    row = {
        "arm": arm, "conc": conc, "reqs": reqs, "ok": len(ok), "errors": len(errors),
        "wall_s": round(wall, 1),
        "qps": round(len(ok) / wall, 3) if wall > 0 and ok else 0.0,
        "ttft_p50": pctl([r["ttft_ms"] for r in ok], 50),
        "ttft_p95": pctl([r["ttft_ms"] for r in ok], 95),
        "e2e_p50": pctl([r["e2e_ms"] for r in ok], 50),
        "e2e_p95": pctl([r["e2e_ms"] for r in ok], 95),
        "retrieval_ms_p50": pctl([r["retrieval_ms"] for r in ok], 50),
        "spot_check": f"{spot_pass}/{len(spot)}",
        "error_samples": [r["error"] for r in errors[:3]],
    }
    print(f"[bench:cell] {json.dumps({k: row[k] for k in row if k != 'error_samples'})}",
          flush=True)
    if errors:
        print(f"[bench:cell]   {len(errors)} error(s), first: {row['error_samples'][:1]}",
              flush=True)
    return row


# ------------------------------------------------------------------ anchor rows
def anchor_rows(cells: list[dict]) -> list[dict]:
    """The estimator's per_model_anchor rows: one per conc, cart vs the rag_CHURN arm (anchor-
    consistent), using the p50 statistic bench_fleet's _excel_rows emits. rag_hot stays out of the
    anchor (it's in the raw JSON). Schema exactly per the brief."""
    by = {(c["arm"], c["conc"]): c for c in cells}
    out: list[dict] = []
    for conc in sorted({c["conc"] for c in cells}):
        cart = by.get(("cart", conc))
        rag = by.get((ANCHOR_RAG_ARM, conc))
        if not (cart and rag):
            continue
        out.append({
            "model": ANCHOR_MODEL, "instance": ANCHOR_INSTANCE, "conc": conc,
            "cart_qps": cart["qps"], "cart_ttft_ms": cart["ttft_p50"], "cart_e2e_ms": cart["e2e_p50"],
            "rag_qps": rag["qps"], "rag_ttft_ms": rag["ttft_p50"], "rag_e2e_ms": rag["e2e_p50"],
        })
    return out


# ------------------------------------------------------------------ retriever build
def build_retriever(doc_ids: list[str], texts: list[str], dense: bool, cache_dir: str | None):
    """The production-parity hybrid retriever over the corpus (bm25s + fastembed + RRF)."""
    from bench.retriever import HybridRetriever
    return HybridRetriever(doc_ids, texts, dense=dense, cache_dir=cache_dir)


# ------------------------------------------------------------------ phases
async def phase_onboard(args) -> None:
    """POST the corpus to /onboard_cag in batches of ~4, timing per batch; report total s, s/doc,
    corpus_tokens, and the response's per-doc fields. Re-running re-onboards (idempotent by doc_id
    on the server unless FORCE_REONBOARD; fine either way)."""
    doc_ids, texts = load_corpus(BENCH_DIR / "corpus.jsonl")
    docs = [{"doc_id": d, "text": t} for d, t in zip(doc_ids, texts, strict=True)]
    client = ServeClient(args.serve_url, args.onboard_url, args.req_timeout)
    batch = max(1, args.onboard_batch)
    totals = {"n_cartridges": 0, "n_built": 0, "corpus_tokens": 0, "errors": {}}
    t0 = time.perf_counter()
    try:
        for i in range(0, len(docs), batch):
            chunk = docs[i:i + batch]
            tb = time.perf_counter()
            resp = await client.onboard(args.corpus_dir, chunk)
            el = time.perf_counter() - tb
            print(f"[bench:onboard] batch {i // batch + 1} "
                  f"({len(chunk)} docs) {el:.1f}s: "
                  f"n_cartridges={resp.get('n_cartridges')} n_built={resp.get('n_built')} "
                  f"corpus_tokens={resp.get('corpus_tokens')} "
                  f"cart_seconds={resp.get('cart_seconds')} errors={resp.get('errors')}",
                  flush=True)
            for k in ("n_cartridges", "n_built", "corpus_tokens"):
                totals[k] += resp.get(k, 0) or 0
            totals["errors"].update(resp.get("errors") or {})
    finally:
        await client.aclose()
    wall = time.perf_counter() - t0
    n = len(docs)
    print(f"[bench:onboard] DONE {n} docs in {wall:.1f}s ({wall / n:.2f} s/doc), "
          f"corpus_tokens={totals['corpus_tokens']}, built={totals['n_built']}, "
          f"errors={len(totals['errors'])}", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "onboard.json").write_text(json.dumps(
        {"docs": n, "wall_s": round(wall, 1), "s_per_doc": round(wall / n, 3), **totals},
        indent=2), encoding="utf-8")


async def phase_chunkdesc(args) -> None:
    """Build the QRC chunk-description sidecar — the onboarding-cost delta of descriptions.
    For each corpus doc, ONE cart-resident generation describes every chunk: the doc's OWN cart
    answers about its OWN content (doc_ids=[doc_id]), so the model reads the full document from its
    resident KV, not just the snippet. chunking.chunk_desc_prompt builds the numbered-chunk prompt;
    chunking.parse_chunk_descs turns the reply into a descs list parallel to chunk_spans(text). A
    parse failure degrades that doc to all-'' descs (never fatal — routing falls back to chunk text).
    Writes BENCH_DIR/qrc_chunks.json = {doc_id: [desc,...]} and prints per-doc timing + a DONE line
    with total wall and s/doc so the description onboarding cost is measured, not guessed."""
    chunking = _load_chunking()
    doc_ids, texts = load_corpus(BENCH_DIR / "corpus.jsonl")
    client = ServeClient(args.serve_url, args.onboard_url, args.req_timeout)
    out: dict[str, list[str]] = {}
    parse_fails = 0
    t0 = time.perf_counter()
    try:
        for doc_id, text in zip(doc_ids, texts, strict=True):
            n_chunks = len(chunking.chunk_spans(text))
            prompt = chunking.chunk_desc_prompt(text)
            td = time.perf_counter()
            # Cart-resident: the doc's own cart answers about its own content. max_tokens=384 gives
            # room for one short line per chunk (the prompt asks for exactly that).
            res = await client.stream("/query_stream", {
                "doc_ids": [doc_id], "question": prompt, "max_tokens": 384})
            el = time.perf_counter() - td
            if res["error"]:
                # A serve error -> empty descs for this doc (routing still works on chunk text).
                out[doc_id] = [""] * n_chunks
                parse_fails += 1
                print(f"[bench:chunkdesc] {doc_id}: {n_chunks} chunks {el:.1f}s ERROR "
                      f"{res['error']} — empty descs", flush=True)
                continue
            descs = chunking.parse_chunk_descs(res["text"], n_chunks)
            out[doc_id] = descs
            filled = sum(1 for d in descs if d)
            if filled == 0 and n_chunks:
                parse_fails += 1
            print(f"[bench:chunkdesc] {doc_id}: {n_chunks} chunks {el:.1f}s "
                  f"({filled}/{n_chunks} described)", flush=True)
    finally:
        await client.aclose()
    wall = time.perf_counter() - t0
    CHUNK_DESCS_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    n = len(doc_ids)
    print(f"[bench:chunkdesc] DONE {n} docs in {wall:.1f}s ({wall / n:.2f} s/doc), "
          f"{parse_fails} doc(s) with no parsed descriptions -> {CHUNK_DESCS_PATH}", flush=True)


async def phase_sweep(args, stop: asyncio.Event) -> None:
    """The concurrency grid: for each conc, for each arm, one closed-loop cell. Persists after every
    cell (a mid-sweep death must not strand finished cells), writes sweep.json (raw, incl. rag_hot),
    and prints the ready-to-paste anchor block (cart vs rag_churn, p50)."""
    doc_ids, texts = load_corpus(BENCH_DIR / "corpus.jsonl")
    questions = load_questions(BENCH_DIR / "questions.json")
    retriever = build_retriever(doc_ids, texts, dense=not args.no_dense, cache_dir=args.cache_dir)
    client = ServeClient(args.serve_url, args.onboard_url, args.req_timeout)
    # The qrc arm folds the chunk-description sidecar into routing when it exists (built by the
    # chunkdesc phase); absent, it routes on chunk text alone.
    descs_by_doc = load_chunk_descs() if "qrc" in args.arms else None
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "sweep.json"
    cells: list[dict] = []
    try:
        for conc in args.conc_list:
            for arm in args.arms:
                if stop.is_set():
                    print("[bench] stop requested — writing partial results and exiting.",
                          flush=True)
                    raise KeyboardInterrupt
                cells.append(await run_cell(
                    client, retriever, arm, conc, questions, args.gen_tokens, args.topk,
                    args.zipf_s, seed=conc * 1000 + _ARM_SEED.get(arm, 9),
                    descs_by_doc=descs_by_doc if arm == "qrc" else None))
                # Persist after EVERY cell (partial-safe), then settle + drain before the next.
                _write_sweep(out_path, cells, args, partial=True)
                await asyncio.sleep(SETTLE_S)
    except KeyboardInterrupt:
        pass
    finally:
        await client.aclose()
    _write_sweep(out_path, cells, args, partial=False)
    rows = anchor_rows(cells)
    print("\n[bench] READY-TO-PASTE per_model_anchor rows (cart vs rag_churn, p50):", flush=True)
    print(json.dumps(rows, indent=2), flush=True)
    print(f"\n[bench] raw cells (incl. rag_hot) -> {out_path}", flush=True)


def _write_sweep(path: Path, cells: list[dict], args, partial: bool) -> None:
    path.write_text(json.dumps({
        "cells": cells,
        "anchor_rows": anchor_rows(cells),
        "partial": partial,
        "config": {"model": ANCHOR_MODEL, "instance": ANCHOR_INSTANCE,
                   "gen_tokens": args.gen_tokens, "topk": args.topk, "zipf_s": args.zipf_s,
                   "conc_list": args.conc_list, "arms": args.arms,
                   "anchor_rag_arm": ANCHOR_RAG_ARM},
    }, indent=2), encoding="utf-8")


async def phase_accuracy(args) -> None:
    """Every question once per arm at conc=1 — cart, qrc, rag_churn — full answers preserved for the
    operator's manual judgment. The qrc arm folds the chunk-description sidecar (qrc_chunks.json)
    into routing when present, else routes on chunk text alone. Dumps accuracy.json:
      [{doc_id, q, expect_substring, cart_answer/cart_ok, qrc_answer/qrc_ok, rag_answer/rag_ok, ...}]."""
    doc_ids, texts = load_corpus(BENCH_DIR / "corpus.jsonl")
    questions = load_questions(BENCH_DIR / "questions.json")
    retriever = build_retriever(doc_ids, texts, dense=not args.no_dense, cache_dir=args.cache_dir)
    client = ServeClient(args.serve_url, args.onboard_url, args.req_timeout)
    descs_by_doc = load_chunk_descs()  # None -> qrc routes on chunk text alone
    rows: list[dict] = []
    try:
        for q in questions:
            cart = await one_request(client, retriever, "cart", q, args.gen_tokens, args.topk)
            qrc = await one_request(client, retriever, "qrc", q, args.gen_tokens, args.topk,
                                    descs_by_doc)
            rag = await one_request(client, retriever, "rag_churn", q, args.gen_tokens, args.topk)
            exp = q.get("expect_substring", "")
            rows.append({
                "doc_id": q.get("doc_id"), "q": q["q"], "expect_substring": exp,
                "cart_answer": cart["text"], "qrc_answer": qrc["text"], "rag_answer": rag["text"],
                "cart_ok": bool(exp) and exp in cart["text"],
                "qrc_ok": bool(exp) and exp in qrc["text"],
                "rag_ok": bool(exp) and exp in rag["text"],
                "cart_error": cart["error"], "qrc_error": qrc["error"], "rag_error": rag["error"],
            })
            print(f"[bench:accuracy] {q.get('doc_id')}: cart_ok={rows[-1]['cart_ok']} "
                  f"qrc_ok={rows[-1]['qrc_ok']} rag_ok={rows[-1]['rag_ok']}", flush=True)
    finally:
        await client.aclose()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "accuracy.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    cart_ok = sum(1 for r in rows if r["cart_ok"])
    qrc_ok = sum(1 for r in rows if r["qrc_ok"])
    rag_ok = sum(1 for r in rows if r["rag_ok"])
    print(f"[bench:accuracy] DONE {len(rows)} questions: cart {cart_ok}/{len(rows)} substring-ok, "
          f"qrc {qrc_ok}/{len(rows)} substring-ok, rag {rag_ok}/{len(rows)} substring-ok "
          f"(full answers in results/accuracy.json for manual judgment)", flush=True)


# ------------------------------------------------------------------ argparse / main
def _parse_conc(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", nargs="?", choices=["onboard", "chunkdesc", "sweep", "accuracy"],
                    help="which phase to run (omit with --selftest)")
    ap.add_argument("--selftest", action="store_true",
                    help="run the GPU-free, network-free self-test and exit")
    ap.add_argument("--serve-url", default=os.environ.get("BENCH_SERVE_URL", "http://127.0.0.1:8002"),
                    help="vLLM Inference Service base URL (/query_stream, /rag_query_stream)")
    ap.add_argument("--onboard-url", default=os.environ.get("BENCH_ONBOARD_URL",
                                                            "http://127.0.0.1:8001"),
                    help="onboarding base URL (/onboard_cag)")
    ap.add_argument("--corpus-dir", default=os.environ.get("BENCH_CORPUS_DIR", "/data/bench_corpus"),
                    help="corpus_dir passed to /onboard_cag (must be inside the box's data roots)")
    ap.add_argument("--conc-list", type=_parse_conc, default=DEFAULT_CONC,
                    help="comma-separated concurrency tiers (default 1,8,32,64,128)")
    ap.add_argument("--arms", type=lambda s: [x for x in s.split(",") if x], default=DEFAULT_ARMS,
                    help="comma-separated arms (default cart,rag_churn,rag_hot)")
    ap.add_argument("--gen-tokens", type=int, default=DEFAULT_GEN_TOKENS)
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    ap.add_argument("--seed", type=int, default=0, help="base seed offset added to every cell seed")
    ap.add_argument("--zipf-s", type=float, default=DEFAULT_ZIPF_S)
    ap.add_argument("--onboard-batch", type=int, default=4, help="docs per /onboard_cag call")
    ap.add_argument("--req-timeout", type=float, default=DEFAULT_REQ_TIMEOUT_S,
                    help="per-request timeout seconds (rag at conc 128 can take minutes)")
    ap.add_argument("--no-dense", action="store_true",
                    help="disable the fastembed dense stage (bm25s lexical-only retrieval)")
    ap.add_argument("--cache-dir", default=os.environ.get("BENCH_FASTEMBED_CACHE", None),
                    help="fastembed ONNX model cache dir")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        from bench.selftest import run_selftest
        return run_selftest()
    if not args.phase:
        print("error: a phase (onboard|chunkdesc|sweep|accuracy) or --selftest is required",
              file=sys.stderr)
        return 2
    # Fold the base --seed into the per-cell seeds so a run is reproducible AND re-seedable.
    if args.seed:
        for arm in list(_ARM_SEED):
            _ARM_SEED[arm] += args.seed
    print("[bench] install hint (box venv):  python -m pip install httpx bm25s fastembed numpy",
          flush=True)

    stop = asyncio.Event()

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        # Graceful ctrl-c: set the stop Event so the sweep writes partial results and exits.
        try:
            loop.add_signal_handler(signal.SIGINT, stop.set)
        except (NotImplementedError, RuntimeError):
            pass  # Windows / no-signal contexts: KeyboardInterrupt still propagates.
        if args.phase == "onboard":
            await phase_onboard(args)
        elif args.phase == "chunkdesc":
            await phase_chunkdesc(args)
        elif args.phase == "sweep":
            await phase_sweep(args, stop)
        elif args.phase == "accuracy":
            await phase_accuracy(args)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("[bench] interrupted — partial results (if any) are on disk.", flush=True)
        return 130
    return 0


if __name__ == "__main__":
    # Make `python bench/headtohead.py` work without a package install: put the repo root on
    # sys.path so `from bench.retriever import ...` resolves (the bench dir's parent).
    sys.path.insert(0, str(BENCH_DIR.parent))
    raise SystemExit(main())
