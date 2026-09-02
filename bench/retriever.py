"""Production-parity retriever for the head-to-head harness.

HARD FAIRNESS REQUIREMENT (from the operator): BOTH arms — cart and RAG — retrieve with the
SAME production-grade pipeline per query. This module mirrors backend/app/retrieval.py's `hybrid`
backend EXACTLY so the benchmark's retrieval is identical to what the live control plane runs:
  * Lexical: bm25s (stopwords="en"), full ranked list, dropped docs appended in doc_id order.
  * Dense:   fastembed TextEmbedding("BAAI/bge-small-en-v1.5", ONNX, no torch), cosine via
             L2-normalized dot product, deterministic doc_id tie-break; degrades to lexical-only
             if fastembed is unavailable (one warning, never raises) — same contract as prod.
  * Fusion:  Reciprocal Rank Fusion, k=60 (Cormack et al.), doc_id-ascending tie-break.
  * top-k:   INFERENCE_TOPK default 3.
The functions below are line-for-line the same algorithm as retrieval.py's rrf_fuse /
_lexical_ranking / _dense_ranking / _hybrid_rank; kept as an embedded copy (not an import) because
retrieval.py drags the whole backend app package (db, storage, models) that this box-local bench
must not need. Divergence from prod retrieval would silently unfair the comparison, so the parity
is asserted by comment at each step against retrieval.py's implementation.

Imports of bm25s/fastembed/numpy are LAZY so headtohead.py --selftest runs on a machine with none
of them installed (the selftest injects a stub retriever instead).
"""
from __future__ import annotations

import logging
import re
import threading

logger = logging.getLogger("bench.retriever")

# Reciprocal Rank Fusion constant — the standard k=60 (retrieval.py._RRF_K). Dampens how much any
# one ranked list's top positions dominate the fused score.
RRF_K = 60
# Dense embedding model — byte-identical to config.RETRIEVAL_DENSE_MODEL default (bge-small, ONNX).
DENSE_MODEL = "BAAI/bge-small-en-v1.5"

_WORD = re.compile(r"[A-Za-z0-9]+")


def rrf_fuse(ranked_lists: list[list[str]], k: int, rrf_k: int = RRF_K) -> list[str]:
    """Reciprocal Rank Fusion over several ranked doc_id lists -> fused top-k. Each list contributes
    1/(rrf_k + rank) (rank 0-based); ties break on doc_id ascending (deterministic). Verbatim
    parity with retrieval.py.rrf_fuse."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    fused = sorted(scores, key=lambda d: (-scores[d], d))
    return fused[:k]


def lexical_ranking(query: str, doc_ids: list[str], texts: list[str]) -> list[str]:
    """doc_ids ranked by bm25s over their texts — EVERY doc_id in score order, dropped docs appended
    in doc_id order (bm25s can drop zero-token docs). Parity with retrieval.py._lexical_ranking."""
    import bm25s
    corpus_tokens = bm25s.tokenize(texts, stopwords="en", show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    query_tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)
    idx, _scores = retriever.retrieve(query_tokens, k=len(doc_ids), show_progress=False)
    ranked = [doc_ids[i] for i in idx[0]]
    seen = set(ranked)
    ranked += sorted(d for d in doc_ids if d not in seen)
    return ranked


# ---- fastembed dense stage (lazily-built singleton, graceful degrade) -----------------------
_embedder = None
_embedder_lock = threading.Lock()
_embedder_failed = False


def _dense_embedder(cache_dir: str | None):
    """fastembed embedder as a lazily-built singleton, or None when unavailable. An import/load
    failure logs ONE warning, latches, and returns None so retrieval degrades to lexical-only
    forever after — never an exception to the caller. Parity with retrieval.py._dense_embedder."""
    global _embedder, _embedder_failed
    if _embedder_failed:
        return None
    if _embedder is not None:
        return _embedder
    with _embedder_lock:
        if _embedder is not None:
            return _embedder
        if _embedder_failed:
            return None
        try:
            from fastembed import TextEmbedding
            kw = {"model_name": DENSE_MODEL}
            if cache_dir:
                kw["cache_dir"] = cache_dir
            _embedder = TextEmbedding(**kw)
            return _embedder
        except Exception as exc:  # noqa: BLE001 — degrade to lexical-only, never crash retrieval
            _embedder_failed = True
            logger.warning("dense retrieval disabled (fastembed unavailable): %s — "
                           "running hybrid lexical-only", exc)
            return None


def dense_ranking(query: str, doc_ids: list[str], texts: list[str],
                  cache_dir: str | None = None) -> list[str] | None:
    """doc_ids ranked by dense cosine similarity to `query`, or None when the dense stage is
    off/unavailable (caller then fuses lexical-only). Cosine via L2-normalized dot; deterministic
    doc_id tie-break. Parity with retrieval.py._dense_ranking."""
    embedder = _dense_embedder(cache_dir)
    if embedder is None:
        return None
    try:
        import numpy as np
        doc_vecs = np.array(list(embedder.embed(texts)), dtype=np.float32)
        q_vec = np.array(list(embedder.embed([query]))[0], dtype=np.float32)
        doc_vecs /= (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-12)
        q_vec /= (np.linalg.norm(q_vec) + 1e-12)
        sims = doc_vecs @ q_vec
        order = sorted(range(len(doc_ids)), key=lambda i: (-float(sims[i]), doc_ids[i]))
        return [doc_ids[i] for i in order]
    except Exception as exc:  # noqa: BLE001 — a bad embed run degrades to lexical-only this query
        logger.warning("dense ranking failed this query, using lexical-only: %s", exc)
        return None


class HybridRetriever:
    """The corpus's built hybrid index + top-k retrieval. Built ONCE over the corpus and reused
    across every query/cell (bm25s re-indexes per query as in prod, but the embedder singleton and
    the doc set are held). retrieve() returns the top-k doc_ids exactly as retrieval.retrieve()
    would; retrieve_context() returns (ids, concatenated text) exactly as retrieval.retrieve_context()
    — the SAME text both arms see (honest apples-to-apples)."""

    def __init__(self, doc_ids: list[str], texts: list[str], dense: bool = True,
                 cache_dir: str | None = None):
        # Parallel lists, ordered — the production index is likewise (doc_ids, index_texts).
        # Here index text == served text (the benchmark has no filename/description metadata layer;
        # prod prepends those to the INDEX text only, never to the served/context text).
        self.doc_ids = list(doc_ids)
        self.texts = list(texts)
        self._by_id = dict(zip(self.doc_ids, self.texts, strict=True))
        self.dense = dense
        self.cache_dir = cache_dir

    def retrieve(self, question: str, k: int) -> list[str]:
        """Top-k doc_ids for `question` (fused bm25s + dense via RRF). Parity with
        retrieval._hybrid_rank -> rrf_fuse."""
        if not self.doc_ids:
            return []
        lexical = lexical_ranking(question, self.doc_ids, self.texts)
        dense = dense_ranking(question, self.doc_ids, self.texts, self.cache_dir) \
            if self.dense else None
        ranked_lists = [lexical] if dense is None else [lexical, dense]
        return rrf_fuse(ranked_lists, k)

    def retrieve_context(self, question: str, k: int) -> tuple[list[str], str]:
        """(top-k doc_ids, their concatenated text) — the cart arm gets the ids, the RAG arm gets
        the SAME text as `context`. Joined with '\\n\\n' exactly as retrieval.retrieve_context."""
        ids = self.retrieve(question, k)
        return ids, "\n\n".join(self._by_id.get(d, "") for d in ids)
