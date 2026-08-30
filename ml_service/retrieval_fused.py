"""Fused retrieval for the demo/product control plane — the 1k-benchmark's WINNING retriever
(sweep v2 `fused+rerank`, measured recall@5 0.462 -> 0.550 on the QASPER haystack), generalized
from papers (title/abstract/body) to arbitrary documents:

  head        the document's first ~HEAD_CHARS — how a document introduces itself. The sweep's
              core insight: questions echo a doc's self-introduction, and whole-body matching
              dilutes that signal (title_abs generalized).
  candidates  RRF( BM25(head), BM25(whole), dense(head), dense(ctx_chunk) ) — two lexical + two
              semantic opinions, fused by rank (scores aren't comparable; ranks are).
  rerank      cross-encoder over the top `pool` candidates on (question, head) — the measured
              single biggest lever.

Lives in the ML service (GPU) because the reranker is impractical on the CPU control plane; the
backend selects it with RETRIEVAL_BACKEND=fused and calls POST /retrieve. Built once per corpus at
onboarding, embeddings persisted under `<corpus_dir>/fused_index.pt` (weights_only-loadable), so a
container restart reloads instead of re-embedding. Encoders are injectable for CPU-only unit tests.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

_WORD = re.compile(r"[a-z0-9]+")
HEAD_CHARS = 1200          # the "self-introduction" window (title+abstract analog)
CHUNK_WORDS, CHUNK_OVERLAP = 200, 40

# Query/document prompts per embedder (Qwen3-Embedding is instruction-tuned; others need none).
_EMBED_PROMPTS = {
    "Qwen/Qwen3-Embedding-0.6B": {
        "q": "Instruct: Given a question, retrieve the document that answers it\nQuery: ", "d": ""},
}


def _toks(s: str) -> list[str]:
    return _WORD.findall(s.lower())


def _chunk_words(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    w = text.split()
    out, i = [], 0
    while i < len(w):
        out.append(" ".join(w[i:i + size]))
        i += size - overlap
    return out


class BM25:
    """Postings-list BM25 (the lab-tested lexical core)."""

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.n = k1, b, len(docs_tokens)
        self.dl = [len(d) for d in docs_tokens]
        self.avgdl = (sum(self.dl) / self.n) if self.n else 0.0
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs_tokens):
            for t, f in Counter(d).items():
                self.post[t].append((i, f))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.post.items()}

    def scores(self, q_tokens: list[str]) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for t in set(q_tokens):
            post = self.post.get(t)
            if not post:
                continue
            idf = self.idf[t]
            for i, f in post:
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / max(1e-9, self.avgdl))
                out[i] += idf * f * (self.k1 + 1) / denom
        return out


def _doc_ranks(bm: BM25, owners: list[str], q_tokens: list[str], topn: int = 50) -> list[str]:
    """Doc score = max over its units (max-pool); returns ranked doc_ids."""
    best: dict[str, float] = {}
    for i, s in bm.scores(q_tokens).items():
        d = owners[i]
        if s > best.get(d, -1):
            best[d] = s
    return [d for d, _ in sorted(best.items(), key=lambda x: x[1], reverse=True)[:topn]]


def _rrf(rank_lists: list[list[str]], k: int = 60, topn: int = 50) -> list[str]:
    score: dict[str, float] = defaultdict(float)
    for rl in rank_lists:
        for r, d in enumerate(rl):
            score[d] += 1.0 / (k + r + 1)
    return [d for d, _ in sorted(score.items(), key=lambda x: x[1], reverse=True)[:topn]]


class FusedIndex:
    """Per-corpus fused retrieval index. `embed_fn(texts)->Tensor[n,d]` and
    `rerank_fn(pairs)->list[float]` are injectable (tests run CPU-only fakes); production uses
    sentence-transformers models built by `load_encoders()`."""

    def __init__(self, docs: dict[str, str], embed_fn=None, rerank_fn=None, q_prompt: str = ""):
        self.doc_ids = list(docs)
        self.head = {d: t[:HEAD_CHARS] for d, t in docs.items()}
        self.embed_fn, self.rerank_fn, self.q_prompt = embed_fn, rerank_fn, q_prompt
        # lexical (rebuilt cheaply at load; only embeddings are worth persisting)
        self.bm_head = (BM25([_toks(self.head[d]) for d in self.doc_ids]), list(self.doc_ids))
        whole_t, whole_o = [docs[d] for d in self.doc_ids], list(self.doc_ids)
        self.bm_whole = (BM25([_toks(t) for t in whole_t]), whole_o)
        # dense units: head + head-situated chunks (contextual retrieval, the lab's ctx_chunk)
        self.unit_texts: list[str] = [self.head[d] for d in self.doc_ids]
        self.unit_owner: list[str] = list(self.doc_ids)
        for d in self.doc_ids:
            for c in _chunk_words(docs[d]):
                self.unit_texts.append(self.head[d][:200] + ". " + c)
                self.unit_owner.append(d)
        self.mat: torch.Tensor | None = None  # (n_units, dim), normalized — set by embed()/load()

    def embed(self) -> None:
        self.mat = self.embed_fn(self.unit_texts)

    def _dense_ranks(self, qv: torch.Tensor, topn: int = 100) -> list[str]:
        order = torch.argsort(self.mat @ qv, descending=True).tolist()
        best: dict[str, int] = {}
        for i in order:
            d = self.unit_owner[i]
            if d not in best:
                best[d] = 1
            if len(best) >= topn:
                break
        return list(best)

    def retrieve(self, question: str, k: int = 3, pool: int = 50) -> list[str]:
        qt = _toks(question)
        # Overlap the two CPU BM25 scans with the GPU query-embed: the embed is dispatched on a
        # worker thread (it releases the GIL in torch/CUDA) while this thread runs the lexical
        # scoring, then we join. Ranking is unaffected — the same three rank lists are assembled in
        # the same fixed order (head, whole, dense) and fed to RRF exactly as before.
        embed_future = None
        if self.mat is not None:
            ex = ThreadPoolExecutor(max_workers=1)
            embed_future = ex.submit(self._embed_query, question)
        try:
            lists = [_doc_ranks(*self.bm_head, qt), _doc_ranks(*self.bm_whole, qt)]
            if embed_future is not None:
                lists.append(self._dense_ranks(embed_future.result()))
        finally:
            if embed_future is not None:
                ex.shutdown(wait=True)
        cand = _rrf(lists, topn=pool)[:pool]
        if self.rerank_fn is not None and cand:
            # One batched predict over ALL pairs (batch_size=pool) — see rerank_fn in load_encoders,
            # which passes the pool size so the cross-encoder doesn't silently re-chunk to 32.
            scores = self.rerank_fn([(question, self.head[d]) for d in cand])
            cand = [d for d, _ in sorted(zip(cand, scores, strict=True),
                                         key=lambda x: x[1], reverse=True)]
        return cand[:k]

    def _embed_query(self, question: str) -> torch.Tensor:
        return self.embed_fn([self.q_prompt + question])[0].to(self.mat.device)

    # ---- persistence: embeddings + the texts needed to rebuild lexical state -----------------
    def save(self, path: str | Path) -> None:
        torch.save({"doc_ids": self.doc_ids, "head": self.head,
                    "unit_texts": self.unit_texts, "unit_owner": self.unit_owner,
                    "mat": self.mat.cpu() if self.mat is not None else None,
                    "q_prompt": self.q_prompt}, path)

    @classmethod
    def load(cls, path: str | Path, docs: dict[str, str], embed_fn=None, rerank_fn=None):
        d = torch.load(path, map_location="cpu", weights_only=True)
        idx = cls(docs, embed_fn=embed_fn, rerank_fn=rerank_fn, q_prompt=d["q_prompt"])
        if d["mat"] is not None:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            idx.mat = d["mat"].to(dev)
        return idx


def load_encoders(embedder: str | None = None, reranker: str | None = None,
                  device: str | None = None):
    """(embed_fn, rerank_fn, q_prompt) from sentence-transformers, fp16 (the sweep's exact
    models). Imported lazily so the module (and its unit tests) work without the dependency.
    Model ids fall back to env RETR_EMBEDDER / RETR_RERANKER (sweep-exact defaults) so a box can
    swap encoders without a code change; app.py passes them explicitly, callers with no args get
    the same values via env."""
    import os

    from sentence_transformers import CrossEncoder, SentenceTransformer
    embedder = embedder or os.environ.get("RETR_EMBEDDER", "Qwen/Qwen3-Embedding-0.6B")
    reranker = reranker or os.environ.get("RETR_RERANKER", "BAAI/bge-reranker-v2-m3")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    st = SentenceTransformer(embedder, device=device, trust_remote_code=True,
                             model_kwargs={"torch_dtype": "float16"})
    rer = CrossEncoder(reranker, device=device, trust_remote_code=True,
                       model_kwargs={"torch_dtype": "float16"})
    prompts = _EMBED_PROMPTS.get(embedder, {"q": "", "d": ""})

    def embed_fn(texts: list[str]) -> torch.Tensor:
        return st.encode(texts, batch_size=32, normalize_embeddings=True,
                         convert_to_tensor=True, show_progress_bar=False).to(device)

    def rerank_fn(pairs: list[tuple[str, str]]) -> list[float]:
        # batch_size = the full pool so predict() runs ONE forward over every pair instead of
        # silently re-chunking to its default 32 (the pool is ~50; a single batch is the win here).
        return [float(s) for s in rer.predict(pairs, batch_size=max(1, len(pairs)),
                                              show_progress_bar=False)]

    return embed_fn, rerank_fn, prompts["q"]
