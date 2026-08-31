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

# Per-tenant namespace separator for cart ids. `__` is inside the cart-store's allowed charset
# (validate_cart_id: [A-Za-z0-9][A-Za-z0-9._-]{0,127}, no '/', no '..'), so a namespaced id is a
# legal store key AND path-traversal-safe. Two underscores can't appear inside a slug — doc_id_for
# collapses each run of non-alnum to a SINGLE '_' — so `<tenant>__<slug>` stays unambiguous.
_NS_SEP = "__"
# Mirror of cartridges.cart_store.validate_cart_id's rule (kept in sync by test_namespacing); we
# re-check here rather than import the IP module so the control plane stays torch-free.
_CART_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def doc_id_for(rel_path: str) -> str:
    """Filename -> per-doc SLUG (the intra-tenant dedup key), matching the id used at onboarding
    (one cart per doc): 'handbook/ch1/intro.md' -> 'handbook_ch1_intro'. NOT the cart id on its own
    anymore — cart_id_for() namespaces this by tenant so two tenants uploading the same filename get
    DIFFERENT carts. Same tenant + same slug still collapses to one cart (cross-corpus dedup kept)."""
    name = Path(rel_path).name
    stem = rel_path[: -(len(name) - name.rfind("."))] if "." in name else rel_path
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_") or "doc"


def cart_id_for(tenant_id: str, rel_path: str) -> str:
    """Filename + owning tenant -> the TENANT-NAMESPACED cartridge id: '<tenant_id>__<slug>'. This is
    the single source of truth for cart-id derivation (jobs/corpora/retrieval all route through it).

    Why namespaced: cart ids address a KV blob in a store shared across ALL tenants, so a bare filename
    slug let two tenants uploading a same-named file collide onto ONE blob — a data-isolation bug. The
    tenant_id (a uuid4 hex, [0-9a-f]{32}) prefix guarantees no two tenants ever address the same cart,
    while a tenant reusing a doc across its OWN corpora still resolves to one cart (dedup preserved).

    The result is validated against the cart-store charset (validate_cart_id) so it's a legal store key
    and can't escape a storage root: tenant_id is hex, the slug is [A-Za-z0-9_], the '__' separator is
    allowed, and 32 + 2 + slug stays under the 128-char cap for any realistic filename."""
    cid = f"{tenant_id}{_NS_SEP}{doc_id_for(rel_path)}"
    if not _CART_ID_RE.fullmatch(cid) or ".." in cid:
        # Defensive: a non-uuid tenant_id (or a pathological slug) could push past the charset/length
        # rule. Never emit an id the store would reject at put/get time — surface it here instead.
        raise ValueError(f"derived cart id {cid!r} is not a valid cart_id")
    return cid


def _tenant_for_corpus(corpus_id: str) -> str:
    """Resolve a corpus's owning tenant_id for namespacing. retrieve()/context helpers only receive a
    corpus_id (chat.py's `retrieve(corpus.id, ...)` call site is fixed), so we look the tenant up in a
    FRESH short-lived session — read-only, no caller session to entangle. A vanished corpus (deleted
    between resolve and read) yields no tenant; callers already treat that as an empty corpus."""
    from .db import SessionLocal
    from .models import Corpus
    db = SessionLocal()
    try:
        c = db.get(Corpus, corpus_id)
        return c.tenant_id if c is not None else ""
    finally:
        db.close()


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


def _corpus_docs(corpus_id: str, tenant_id: str) -> tuple[list[tuple[str, list[str]]], dict[str, str]]:
    """(bm25 input, {cart_id: text}) over a corpus's non-empty documents — the shared basis for both
    retrieve() and retrieve_context(). Keys are the TENANT-NAMESPACED cart ids so what retrieve()
    returns is exactly the id the Inference Service serves (no bare-slug leaks past this layer)."""
    docs: list[tuple[str, list[str]]] = []
    texts: dict[str, str] = {}
    for fn in storage.list_doc_filenames(corpus_id):
        text = storage.read_text(corpus_id, fn)
        if text.strip():
            did = cart_id_for(tenant_id, fn)
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
    # Resolve the owning tenant from the corpus_id the call site already passes — so chat.py's
    # `retrieve(corpus.id, ...)` stays unchanged yet the returned ids are tenant-namespaced.
    docs, _ = _corpus_docs(corpus_id, _tenant_for_corpus(corpus_id))
    return bm25_rank(question, docs, k)


def retrieve_context(corpus_id: str, question: str, k: int | None = None) -> tuple[list[str], str]:
    """Top-k doc_ids AND their concatenated text. The doc_ids feed the resident-KV cart serve path; the
    SAME text is the RAG baseline's re-prefilled context — so the head-to-head compare sees identical
    evidence on both sides (honest apples-to-apples)."""
    k = k or config.INFERENCE_TOPK
    if config.RETRIEVAL_BACKEND == "fused":
        ids = ml_client.retrieve_fused(_corpus_dir_str(corpus_id), question, k)
        return ids, "\n\n".join(_doc_text(corpus_id, d) for d in ids)
    docs, texts = _corpus_docs(corpus_id, _tenant_for_corpus(corpus_id))
    ids = bm25_rank(question, docs, k)
    return ids, "\n\n".join(texts.get(d, "") for d in ids)


def validate_doc_ids(corpus_id: str, doc_ids: list[str]) -> None:
    """Membership check for client-echoed doc_ids (session pinning): unknown ids raise KeyError.
    Same contract as context_for's validation but WITHOUT reading any document text — the vLLM
    serve path only needs the ids, and reading N full docs per pinned turn would claw back the
    retrieval time pinning exists to save.

    The known set is this corpus's TENANT-NAMESPACED cart ids, so a client echoing ANOTHER tenant's
    id (different namespace prefix) is rejected here — the pin path can't address a foreign cart."""
    tenant_id = _tenant_for_corpus(corpus_id)
    known = {cart_id_for(tenant_id, fn) for fn in storage.list_doc_filenames(corpus_id)}
    missing = [d for d in doc_ids if d not in known]
    if missing:
        raise KeyError(f"unknown doc_ids for this corpus: {missing}")


def context_for(corpus_id: str, doc_ids: list[str]) -> str:
    """Concatenated text for already-retrieved doc_ids — the compare-stream RAG side reuses
    the cart side's retrieval instead of re-running it (retrieval is one GPU round-trip per
    call on the fused backend). doc_ids come from the client, so unknown ids raise KeyError
    rather than silently serving a smaller context than the cart side saw. Keys are namespaced,
    so a foreign tenant's id never resolves to this corpus's text."""
    tenant_id = _tenant_for_corpus(corpus_id)
    by_id = {cart_id_for(tenant_id, fn): fn for fn in storage.list_doc_filenames(corpus_id)}
    missing = [d for d in doc_ids if d not in by_id]
    if missing:
        raise KeyError(f"unknown doc_ids for this corpus: {missing}")
    return "\n\n".join(storage.read_text(corpus_id, by_id[d]) for d in doc_ids)


def _doc_text(corpus_id: str, doc_id: str) -> str:
    """Text of one doc by its NAMESPACED cart id (reverse of cart_id_for over the stored filenames)."""
    tenant_id = _tenant_for_corpus(corpus_id)
    for fn in storage.list_doc_filenames(corpus_id):
        if cart_id_for(tenant_id, fn) == doc_id:
            return storage.read_text(corpus_id, fn)
    return ""


def doc_sources(corpus_id: str, doc_ids: list[str]) -> list[dict]:
    """[{id, title}] for the UI's Sources block — title = the document's first non-empty line
    (our stored format puts the title there). Build the namespaced {cart_id: filename} map ONCE
    (one tenant resolve, one dir listing) instead of re-scanning per id."""
    tenant_id = _tenant_for_corpus(corpus_id)
    by_id = {cart_id_for(tenant_id, fn): fn for fn in storage.list_doc_filenames(corpus_id)}
    out = []
    for d in doc_ids:
        fn = by_id.get(d)
        text = storage.read_text(corpus_id, fn) if fn else ""
        title = next((ln.strip() for ln in text.splitlines() if ln.strip()), d)
        out.append({"id": d, "title": title[:140]})
    return out
