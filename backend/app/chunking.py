"""Query-routed chunk selection (QRC) — the shared core both consumers import.

The k=3 accuracy A/B (2026-09-03, PLAN decisions log) proved composed solo-encoded
carts interfere REGARDLESS of compression (lossless bf16 fails identically), so the
hybrid serve mode keeps ONE resident cart (the top-ranked doc — the proven 15/15
path) and routes the OTHER retrieved docs' answer-bearing chunks in as small
real-token context. This module is the routing step: deterministic chunking plus
per-doc hybrid (bm25s + optional dense) chunk ranking under a token budget.

SHARED between the control plane (backend.app.retrieval) and the benchmark harness
(bench/retriever.py syncs this exact file to the GPU box), so the bench measures
the byte-identical selection the product runs. Therefore: pure module — no app
imports, no torch, bm25s/numpy imported lazily inside functions.

Chunk descriptions: a short LLM line per chunk may ride along as retrieval
METADATA (folded into the chunk's index text, mirroring the doc-description
design) — it is never part of the SERVED text; the document's own words are what
get prefilled.
"""
from __future__ import annotations

# One source of truth for the routing shape. ~4 chars/token is the standard
# whitespace-English heuristic (no tokenizer dependency — the control plane is
# torch-free and the exact count doesn't matter, only a consistent budget).
CHUNK_TOKENS = 96          # target chunk size (route granularity)
CHUNK_BUDGET_TOKENS = 256  # selected tokens per routed doc (~2-3 chunks)
CHARS_PER_TOKEN = 4
_RRF_K = 60                # same Cormack et al. constant as doc-level retrieval

# Joins for the composed context: selected chunks of one doc in DOCUMENT ORDER
# (the ellipsis marks an elision so the model knows text was skipped), docs
# separated by a blank line — the same doc separator the RAG path uses.
_CHUNK_JOIN = "\n[…]\n"
_DOC_JOIN = "\n\n"


def chunk_spans(text: str, chunk_tokens: int = CHUNK_TOKENS) -> list[tuple[int, int]]:
    """Deterministic char spans covering `text`: ~chunk_tokens*CHARS_PER_TOKEN chars
    each, cut at a whitespace boundary (searching back up to 25% of the window so a
    word is never split; a whitespace-free run longer than that cuts hard). A tiny
    tail (< 25% of a chunk) merges into the previous span. Same text -> same spans,
    always — the onboarding-time description sidecar and query-time routing both
    call this, so their chunk ordinals must never drift."""
    n = len(text)
    if n == 0:
        return []
    window = max(1, chunk_tokens * CHARS_PER_TOKEN)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + window, n)
        if end < n:
            # Prefer to end on whitespace: scan back from the hard cut.
            back = text.rfind(" ", start + int(window * 0.75), end)
            nl = text.rfind("\n", start + int(window * 0.75), end)
            cut = max(back, nl)
            if cut > start:
                end = cut + 1
        spans.append((start, end))
        start = end
    # Merge a runt tail into its predecessor so no chunk is too small to rank.
    if len(spans) >= 2 and (spans[-1][1] - spans[-1][0]) < window // 4:
        spans[-2] = (spans[-2][0], spans[-1][1])
        spans.pop()
    return spans


def chunk_texts(text: str, chunk_tokens: int = CHUNK_TOKENS) -> list[str]:
    """The chunks themselves (convenience over chunk_spans)."""
    return [text[a:b] for a, b in chunk_spans(text, chunk_tokens)]


def _lexical_chunk_rank(question: str, index_texts: list[str]) -> list[int]:
    """Chunk indices ranked by bm25s over their index texts — full list, dropped
    (zero-token) chunks appended in index order for determinism. Mirrors the doc-level
    _lexical_ranking contract."""
    import bm25s
    tokens = bm25s.tokenize(index_texts, stopwords="en", show_progress=False)
    bm = bm25s.BM25()
    bm.index(tokens, show_progress=False)
    q = bm25s.tokenize(question, stopwords="en", show_progress=False)
    idx, _scores = bm.retrieve(q, k=len(index_texts), show_progress=False)
    ranked = [int(i) for i in idx[0]]
    seen = set(ranked)
    ranked += [i for i in range(len(index_texts)) if i not in seen]
    return ranked


def _dense_chunk_rank(question: str, index_texts: list[str], embedder) -> list[int] | None:
    """Chunk indices ranked by dense cosine to the question, or None when no embedder
    (caller fuses lexical-only — the same graceful degrade as doc retrieval)."""
    if embedder is None:
        return None
    try:
        import numpy as np
        vecs = np.array(list(embedder.embed(index_texts)), dtype=np.float32)
        qv = np.array(list(embedder.embed([question]))[0], dtype=np.float32)
        vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
        qv /= (np.linalg.norm(qv) + 1e-12)
        sims = vecs @ qv
        return sorted(range(len(index_texts)), key=lambda i: (-float(sims[i]), i))
    except Exception:  # noqa: BLE001 — dense degrades, lexical still routes
        return None


def _rrf_indices(ranked_lists: list[list[int]], rrf_k: int = _RRF_K) -> list[int]:
    """RRF over chunk-index rankings (int twin of retrieval.rrf_fuse; index ascending
    as the deterministic tie-break)."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, i in enumerate(ranked):
            scores[i] = scores.get(i, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores, key=lambda i: (-scores[i], i))


def route_chunks(question: str, docs: list[dict], embedder=None,
                 chunk_tokens: int = CHUNK_TOKENS,
                 budget_tokens: int = CHUNK_BUDGET_TOKENS) -> list[dict]:
    """Per routed doc, the answer-bearing chunks for `question` under a token budget.

    docs: [{"doc_id": str, "text": str, "descs": list[str] | None}] — descs, when
    present, are parallel to chunk_spans(text, chunk_tokens) and are folded into each
    chunk's INDEX text only (retrieval metadata, mirroring doc descriptions).

    Selection is PER DOC (each routed doc gets its own budget and always
    contributes): rank the doc's chunks lexically (bm25s) + densely (when an
    embedder is given), RRF-fuse, then take top chunks until ~budget_tokens.
    Selected chunks are re-assembled in DOCUMENT ORDER with an elision marker.

    Returns [{"doc_id", "text", "chunk_indices"}] in the input doc order. A doc at
    or under budget passes through whole (its full text IS the selection)."""
    out: list[dict] = []
    budget_chars = budget_tokens * CHARS_PER_TOKEN
    for doc in docs:
        text = doc["text"]
        spans = chunk_spans(text, chunk_tokens)
        if not spans:
            continue
        if len(text) <= budget_chars or len(spans) == 1:
            out.append({"doc_id": doc["doc_id"], "text": text,
                        "chunk_indices": list(range(len(spans)))})
            continue
        chunks = [text[a:b] for a, b in spans]
        descs = doc.get("descs") or []
        index_texts = [
            (f"{descs[i]}\n{c}" if i < len(descs) and descs[i] else c)
            for i, c in enumerate(chunks)
        ]
        lexical = _lexical_chunk_rank(question, index_texts)
        dense = _dense_chunk_rank(question, index_texts, embedder)
        fused = _rrf_indices([lexical] if dense is None else [lexical, dense])
        picked: list[int] = []
        used = 0
        for i in fused:
            picked.append(i)
            used += spans[i][1] - spans[i][0]
            if used >= budget_chars:
                break
        picked.sort()  # document order — the model reads a coherent (elided) doc
        parts: list[str] = []
        prev = None
        for i in picked:
            if prev is not None and i != prev + 1:
                parts.append(_CHUNK_JOIN)
            elif prev is not None:
                parts.append("")  # adjacent chunks: seamless
            parts.append(chunks[i])
            prev = i
        out.append({"doc_id": doc["doc_id"], "text": "".join(parts),
                    "chunk_indices": picked})
    return out


def compose_context(routed: list[dict]) -> str:
    """route_chunks output -> the single context string the serve path prefills."""
    return _DOC_JOIN.join(r["text"] for r in routed if r["text"])


def chunk_desc_prompt(text: str, chunk_tokens: int = CHUNK_TOKENS) -> str:
    """The cart-resident description prompt: ONE generation per doc yields a line per
    chunk. Each chunk is identified by ordinal + its first words (the model has the
    full document resident via its cart, so it describes from content, not from the
    snippet alone). Output contract: numbered lines, parsed by parse_chunk_descs."""
    firsts = []
    for i, c in enumerate(chunk_texts(text, chunk_tokens)):
        head = " ".join(c.split()[:8])
        firsts.append(f"{i + 1}. …{head}…")
    listing = "\n".join(firsts)
    return (
        "This document is split into numbered chunks; each is identified below by its "
        "first words. For EVERY chunk, write exactly one line: the chunk number, a "
        "period, then a short description (under 15 words) of what facts that part of "
        "the document contains. No other text.\n\n" + listing
    )


def parse_chunk_descs(reply: str, n_chunks: int) -> list[str]:
    """Parse 'N. description' lines -> a descs list of length n_chunks ('' where the
    model skipped/garbled a line — routing then falls back to the chunk's own text,
    so a bad description can only fail to help, never hurt)."""
    import re
    descs = [""] * n_chunks
    for line in reply.splitlines():
        m = re.match(r"\s*(\d+)\s*[.):-]\s+(.{3,})", line)
        if m:
            i = int(m.group(1)) - 1
            if 0 <= i < n_chunks and not descs[i]:
                descs[i] = m.group(2).strip()
    return descs
