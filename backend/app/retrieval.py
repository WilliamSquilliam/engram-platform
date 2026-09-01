"""Control-plane retrieval: pick which cartridge(s) (doc_ids) answer a query, so the lean vLLM
Inference Service can be handed doc_ids instead of a whole corpus (per the C2 split: retrieval is the
control plane's job; serving is the GPU's).

DEFAULT backend `hybrid`: the industry-standard lexical+dense hybrid, run IN-PROCESS and torch-free.
  * Lexical: `bm25s` (fast, maintained BM25 — replaces the old bespoke pure-python scorer).
  * Dense:   `fastembed` (ONNX, no torch; small bge model). Lazily-built module singleton. GRACEFUL
             DEGRADE — if fastembed import/model-load fails OR RETRIEVAL_DENSE=off, hybrid runs
             lexical-only with ONE warning, never an exception to the caller.
  * Fusion:  Reciprocal Rank Fusion (k=60) over the two ranked lists, deterministic doc_id tie-break.
The per-corpus index is built lazily and cached in-process, invalidated by a cheap signal (the corpus's
filenames + their descriptions), so adding/removing a doc or re-onboarding rebuilds it.

Other backends: `fused` (the GPU-box BM25+dense+rerank pipeline — the at-scale path, untouched);
`bm25` (legacy alias -> hybrid); `pgvector` (a documented seam). All swap behind retrieve() — the rest
of the app only depends on retrieve()/retrieve_context().
"""
from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

from . import config, ml_client
from .storage import storage

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[A-Za-z0-9]+")

# Per-tenant namespace separator for cart ids. `__` is inside the cart-store's allowed charset
# (validate_cart_id: [A-Za-z0-9][A-Za-z0-9._-]{0,127}, no '/', no '..'), so a namespaced id is a
# legal store key AND path-traversal-safe. Two underscores can't appear inside a slug — doc_id_for
# collapses each run of non-alnum to a SINGLE '_' — so `<tenant>__<slug>` stays unambiguous.
_NS_SEP = "__"
# Mirror of cartridges.cart_store.validate_cart_id's rule (kept in sync by test_namespacing); we
# re-check here rather than import the IP module so the control plane stays torch-free.
_CART_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# Reciprocal Rank Fusion constant (the standard k=60 from Cormack et al.); dampens how much any single
# ranked list's top positions dominate the fused score.
_RRF_K = 60


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


def _descriptions_for_corpus(corpus_id: str, tenant_id: str) -> dict[str, str]:
    """{cart_id: description} for the corpus's documents that have an LLM description (Feature 1).
    Read from the Document rows in a fresh session; the description is folded into the doc's INDEX
    text (retrieval metadata only — never served). Empty when the feature is off / nothing filled."""
    from .db import SessionLocal
    from .models import Document
    out: dict[str, str] = {}
    db = SessionLocal()
    try:
        for d in db.query(Document).filter(Document.corpus_id == corpus_id):
            if d.description:
                out[cart_id_for(tenant_id, d.filename)] = d.description
    finally:
        db.close()
    return out


def _tokens(s: str) -> list[str]:
    return _WORD.findall(s.lower())


# --------------------------------------------------------------------------- reciprocal rank fusion
def rrf_fuse(ranked_lists: list[list[str]], k: int, rrf_k: int = _RRF_K) -> list[str]:
    """Reciprocal Rank Fusion over several ranked doc_id lists -> the fused top-k doc_ids. Each list
    contributes 1/(rrf_k + rank) to a doc's score (rank is 0-based). Ties break DETERMINISTICALLY on
    doc_id so tests are stable. Lists may differ in length / membership; a doc absent from a list just
    earns nothing from it."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    # Highest score first; doc_id ascending as the deterministic tie-break.
    fused = sorted(scores, key=lambda d: (-scores[d], d))
    return fused[:k]


# --------------------------------------------------------------------------- dense embedder singleton
_embedder = None                 # the fastembed TextEmbedding instance (built once)
_embedder_lock = threading.Lock()
_embedder_failed = False         # set True after a failed import/load so we don't retry (and re-log) it


def _dense_embedder():
    """The fastembed embedder as a lazily-built module singleton, or None when the dense stage is
    disabled/unavailable. RETRIEVAL_DENSE=off returns None immediately (no download — the tests' path).
    An import or model-load failure logs ONE warning, latches `_embedder_failed`, and returns None so
    hybrid degrades to lexical-only forever after (never an exception to the caller, never a retry)."""
    global _embedder, _embedder_failed
    if config.RETRIEVAL_DENSE == "off" or _embedder_failed:
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
            config.FASTEMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            _embedder = TextEmbedding(model_name=config.RETRIEVAL_DENSE_MODEL,
                                      cache_dir=str(config.FASTEMBED_CACHE_DIR))
            return _embedder
        except Exception as exc:  # noqa: BLE001 — degrade to lexical-only, never crash retrieval
            _embedder_failed = True
            logger.warning("dense retrieval disabled (fastembed unavailable): %s — running "
                           "hybrid lexical-only", exc)
            return None


def _dense_ranking(query: str, doc_ids: list[str], texts: list[str]) -> list[str] | None:
    """doc_ids ranked by dense cosine similarity to `query`, or None when the dense stage is
    off/unavailable (caller then fuses lexical-only). Cosine via normalized dot product; deterministic
    doc_id tie-break. Any runtime embedding error degrades to None (best-effort, never raises)."""
    embedder = _dense_embedder()
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
    except Exception as exc:  # noqa: BLE001 — a bad embed run degrades to lexical-only for this query
        logger.warning("dense ranking failed this query, using lexical-only: %s", exc)
        return None


def _lexical_ranking(query: str, doc_ids: list[str], texts: list[str]) -> list[str]:
    """doc_ids ranked by bm25s over their index texts. Returns EVERY doc_id in score order (so the
    fusion sees a full ranked list); bm25s breaks its own ties, and we append any doc it dropped in
    doc_id order for determinism."""
    import bm25s
    corpus_tokens = bm25s.tokenize(texts, stopwords="en", show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    query_tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)
    idx, _scores = retriever.retrieve(query_tokens, k=len(doc_ids), show_progress=False)
    ranked = [doc_ids[i] for i in idx[0]]
    # bm25s can drop zero-length-token docs; append them (doc_id order) so the list is complete.
    seen = set(ranked)
    ranked += sorted(d for d in doc_ids if d not in seen)
    return ranked


# --------------------------------------------------------------------------- per-corpus index cache
class _CorpusIndex:
    """A built hybrid index for one corpus: the ordered doc_ids, their index texts (filename +
    description + extracted text), the per-id served text, and the invalidation signature it was
    built under. Rebuilt when the signature changes (see _corpus_index)."""

    __slots__ = ("doc_ids", "index_texts", "texts", "signature")

    def __init__(self, doc_ids, index_texts, texts, signature):
        self.doc_ids = doc_ids
        self.index_texts = index_texts
        self.texts = texts
        self.signature = signature


_index_cache: dict[str, _CorpusIndex] = {}
_index_lock = threading.Lock()


def _corpus_signature(corpus_id: str, tenant_id: str,
                      descriptions: dict[str, str]) -> tuple:
    """Cheap staleness signal for the cached index: the corpus's filenames plus each doc's description.
    Adding/removing a doc changes the filename set; a fresh onboard that (re)writes descriptions
    changes the description tuple — either invalidates the cache and forces a rebuild. Backend-agnostic
    (no mtimes), so it works identically on the local and S3 stores."""
    filenames = tuple(sorted(storage.list_doc_filenames(corpus_id)))
    descs = tuple(sorted(descriptions.items()))
    return (filenames, descs)


def _build_index_texts(corpus_id: str, tenant_id: str,
                       descriptions: dict[str, str]) -> tuple[list[str], list[str], dict[str, str]]:
    """(doc_ids, index_texts, {cart_id: served_text}) over the corpus's non-empty documents. The INDEX
    text prepends `filename tokens + description (when present)` to the extracted text, so BOTH signals
    are visible to lexical and dense ranking. The description is retrieval METADATA only — it never
    enters `texts` (the served/context text), because the cart already holds the full document."""
    doc_ids: list[str] = []
    index_texts: list[str] = []
    texts: dict[str, str] = {}
    for fn in storage.list_doc_filenames(corpus_id):
        text = storage.read_text(corpus_id, fn)
        if not text.strip():
            continue
        cart_id = cart_id_for(tenant_id, fn)
        # Filename tokens help match queries that name the file; the description is a compact summary
        # the LLM wrote. Both are prepended so lexical + dense see them alongside the body.
        filename_terms = " ".join(_tokens(fn))
        desc = descriptions.get(cart_id, "")
        index_texts.append("\n".join(p for p in (filename_terms, desc, text) if p))
        doc_ids.append(cart_id)
        texts[cart_id] = text
    return doc_ids, index_texts, texts


def _corpus_index(corpus_id: str, tenant_id: str) -> _CorpusIndex:
    """The cached-or-freshly-built index for a corpus, keyed by corpus_id and invalidated by the cheap
    signature (filenames + descriptions). Locked so two concurrent queries don't both rebuild."""
    descriptions = _descriptions_for_corpus(corpus_id, tenant_id)
    signature = _corpus_signature(corpus_id, tenant_id, descriptions)
    cached = _index_cache.get(corpus_id)
    if cached is not None and cached.signature == signature:
        return cached
    with _index_lock:
        cached = _index_cache.get(corpus_id)
        if cached is not None and cached.signature == signature:
            return cached
        doc_ids, index_texts, texts = _build_index_texts(corpus_id, tenant_id, descriptions)
        idx = _CorpusIndex(doc_ids, index_texts, texts, signature)
        _index_cache[corpus_id] = idx
        return idx


def _hybrid_rank(corpus_id: str, tenant_id: str, question: str, k: int) -> tuple[list[str], dict[str, str]]:
    """(top-k doc_ids, {cart_id: served_text}) for the hybrid backend. Runs bm25s (always) and the
    dense stage (when on) over the cached index, fuses their ranked lists with RRF, and returns the
    top-k. On an empty corpus returns ([], {}). The served-text map is returned so retrieve_context
    reuses this exact index (no second read)."""
    idx = _corpus_index(corpus_id, tenant_id)
    if not idx.doc_ids:
        return [], {}
    lexical = _lexical_ranking(question, idx.doc_ids, idx.index_texts)
    dense = _dense_ranking(question, idx.doc_ids, idx.index_texts)
    ranked_lists = [lexical] if dense is None else [lexical, dense]
    return rrf_fuse(ranked_lists, k), idx.texts


def invalidate_index(corpus_id: str) -> None:
    """Drop a corpus's cached index (called on corpus delete). Not strictly required for CORRECTNESS —
    the signature check rebuilds a stale index on the next query, and a deleted corpus is never queried
    again — but it frees the entry promptly. Idempotent."""
    _index_cache.pop(corpus_id, None)


def _corpus_dir_str(corpus_id: str) -> str:
    return str(storage.corpus_dir(corpus_id))


def retrieve(corpus_id: str, question: str, k: int | None = None) -> list[str]:
    """Top-k cartridge doc_ids for `question` over `corpus_id`'s documents. The doc_ids match the
    cart ids the onboarding worker stored, so the Inference Service can serve them directly.

    Backends: `hybrid` (DEFAULT — in-process bm25s + fastembed fused by RRF, torch-free) or `fused`
    (the GPU box's BM25+dense+rerank pipeline). `bm25` maps to hybrid (legacy alias)."""
    k = k or config.INFERENCE_TOPK
    if config.RETRIEVAL_BACKEND == "fused":
        return ml_client.retrieve_fused(_corpus_dir_str(corpus_id), question, k)
    if config.RETRIEVAL_BACKEND == "pgvector":  # seam: vector index (RDS) — not wired here yet
        raise NotImplementedError("RETRIEVAL_BACKEND=pgvector is a documented seam; use 'hybrid' or 'fused'")
    # Resolve the owning tenant from the corpus_id the call site already passes — so chat.py's
    # `retrieve(corpus.id, ...)` stays unchanged yet the returned ids are tenant-namespaced.
    ids, _ = _hybrid_rank(corpus_id, _tenant_for_corpus(corpus_id), question, k)
    return ids


def retrieve_context(corpus_id: str, question: str, k: int | None = None) -> tuple[list[str], str]:
    """Top-k doc_ids AND their concatenated text. The doc_ids feed the resident-KV cart serve path; the
    SAME text is the RAG baseline's re-prefilled context — so the head-to-head compare sees identical
    evidence on both sides (honest apples-to-apples). The served text is the EXTRACTED document text
    only (never the retrieval-metadata description)."""
    k = k or config.INFERENCE_TOPK
    if config.RETRIEVAL_BACKEND == "fused":
        ids = ml_client.retrieve_fused(_corpus_dir_str(corpus_id), question, k)
        return ids, "\n\n".join(_doc_text(corpus_id, d) for d in ids)
    ids, texts = _hybrid_rank(corpus_id, _tenant_for_corpus(corpus_id), question, k)
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
