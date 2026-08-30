"""Control-plane retrieval: pick which cartridge(s) (doc_ids) answer a query, so the lean vLLM
Inference Service can be handed doc_ids instead of a whole corpus (per the C2 split: retrieval is the
control plane's job; serving is the GPU's).

Today: a small **pure-python BM25** over the corpus's document texts — zero dependencies, runs in the
API process, fine for the demo / modest corpora. `RETRIEVAL_BACKEND=pgvector` is the architecture
target: a vector index swapped behind the same `retrieve()` signature (the rest of the app only
depends on retrieve()).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from . import config, ml_client
from .storage import storage

_WORD = re.compile(r"[A-Za-z0-9]+")


def doc_id_for(rel_path: str) -> str:
    """Filename -> cartridge id (slug), matching the id used at onboarding (one cart per doc):
    'handbook/ch1/intro.md' -> 'handbook_ch1_intro'. Single source of truth (jobs.py imports this)."""
    name = Path(rel_path).name
    stem = rel_path[: -(len(name) - name.rfind("."))] if "." in name else rel_path
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_") or "doc"


def _tokens(s: str) -> list[str]:
    return _WORD.findall(s.lower())


def bm25_rank(query: str, docs: list[tuple[str, list[str]]], k: int,
              k1: float = 1.5, b: float = 0.75) -> list[str]:
    """Classic BM25 over a small in-memory corpus. `docs` = [(doc_id, tokens)]; returns the top-k
    doc_ids by score (ties broken by original order). Pure-python, no deps."""
    n = len(docs)
    if n == 0:
        return []
    lengths = [len(t) for _, t in docs]
    avgdl = (sum(lengths) / n) or 1.0
    df: Counter[str] = Counter()
    for _, toks in docs:
        df.update(set(toks))
    q = _tokens(query)
    scored: list[tuple[float, int, str]] = []
    for i, (did, toks) in enumerate(docs):
        tf = Counter(toks)
        s = 0.0
        for w in q:
            f = tf.get(w, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * lengths[i] / avgdl))
        scored.append((s, -i, did))            # -i keeps original order on ties
    scored.sort(reverse=True)
    return [did for s, _, did in scored[:k] if s > 0] or [docs[0][0]]


def _corpus_docs(corpus_id: str) -> tuple[list[tuple[str, list[str]]], dict[str, str]]:
    """(bm25 input, {doc_id: text}) over a corpus's non-empty documents — the shared basis for both
    retrieve() and retrieve_context()."""
    docs: list[tuple[str, list[str]]] = []
    texts: dict[str, str] = {}
    for fn in storage.list_doc_filenames(corpus_id):
        text = storage.read_text(corpus_id, fn)
        if text.strip():
            did = doc_id_for(fn)
            docs.append((did, _tokens(text)))
            texts[did] = text
    return docs, texts


def _corpus_dir_str(corpus_id: str) -> str:
    return str(storage.corpus_dir(corpus_id))


def retrieve(corpus_id: str, question: str, k: int | None = None) -> list[str]:
    """Top-k cartridge doc_ids for `question` over `corpus_id`'s documents. The doc_ids match the
    cart ids the onboarding worker stored, so the Inference Service can serve them directly.

    Backends: `bm25` (zero-dep, in-process — fine to a few hundred docs) or `fused` (the
    1k-benchmark's BM25+dense+rerank pipeline, served from the GPU box — the at-scale path)."""
    k = k or config.INFERENCE_TOPK
    if config.RETRIEVAL_BACKEND == "fused":
        return ml_client.retrieve_fused(_corpus_dir_str(corpus_id), question, k)
    if config.RETRIEVAL_BACKEND == "pgvector":  # seam: vector index (RDS) — not wired here yet
        raise NotImplementedError("RETRIEVAL_BACKEND=pgvector is a documented seam; use 'bm25' or 'fused'")
    docs, _ = _corpus_docs(corpus_id)
    return bm25_rank(question, docs, k)


def retrieve_context(corpus_id: str, question: str, k: int | None = None) -> tuple[list[str], str]:
    """Top-k doc_ids AND their concatenated text. The doc_ids feed the resident-KV cart serve path; the
    SAME text is the RAG baseline's re-prefilled context — so the head-to-head compare sees identical
    evidence on both sides (honest apples-to-apples)."""
    k = k or config.INFERENCE_TOPK
    if config.RETRIEVAL_BACKEND == "fused":
        ids = ml_client.retrieve_fused(_corpus_dir_str(corpus_id), question, k)
        return ids, "\n\n".join(_doc_text(corpus_id, d) for d in ids)
    docs, texts = _corpus_docs(corpus_id)
    ids = bm25_rank(question, docs, k)
    return ids, "\n\n".join(texts.get(d, "") for d in ids)


def validate_doc_ids(corpus_id: str, doc_ids: list[str]) -> None:
    """Membership check for client-echoed doc_ids (session pinning): unknown ids raise KeyError.
    Same contract as context_for's validation but WITHOUT reading any document text — the vLLM
    serve path only needs the ids, and reading N full docs per pinned turn would claw back the
    retrieval time pinning exists to save."""
    known = {doc_id_for(fn) for fn in storage.list_doc_filenames(corpus_id)}
    missing = [d for d in doc_ids if d not in known]
    if missing:
        raise KeyError(f"unknown doc_ids for this corpus: {missing}")


def context_for(corpus_id: str, doc_ids: list[str]) -> str:
    """Concatenated text for already-retrieved doc_ids — the compare-stream RAG side reuses
    the cart side's retrieval instead of re-running it (retrieval is one GPU round-trip per
    call on the fused backend). doc_ids come from the client, so unknown ids raise KeyError
    rather than silently serving a smaller context than the cart side saw."""
    by_id = {doc_id_for(fn): fn for fn in storage.list_doc_filenames(corpus_id)}
    missing = [d for d in doc_ids if d not in by_id]
    if missing:
        raise KeyError(f"unknown doc_ids for this corpus: {missing}")
    return "\n\n".join(storage.read_text(corpus_id, by_id[d]) for d in doc_ids)


def _doc_text(corpus_id: str, doc_id: str) -> str:
    """Text of one doc by its cart id (reverse of doc_id_for over the stored filenames)."""
    for fn in storage.list_doc_filenames(corpus_id):
        if doc_id_for(fn) == doc_id:
            return storage.read_text(corpus_id, fn)
    return ""


def doc_sources(corpus_id: str, doc_ids: list[str]) -> list[dict]:
    """[{id, title}] for the UI's Sources block — title = the document's first non-empty line
    (our stored format puts the title there)."""
    out = []
    for d in doc_ids:
        text = _doc_text(corpus_id, d)
        title = next((ln.strip() for ln in text.splitlines() if ln.strip()), d)
        out.append({"id": d, "title": title[:140]})
    return out
