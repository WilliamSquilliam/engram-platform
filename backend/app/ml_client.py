"""Thin HTTP client to the ML service (train + query). Keeps torch out of the
control-plane process. On AWS this same call goes to the Inference Service /
Training Worker behind an internal ALB; only ML_SERVICE_URL changes."""
import os

import httpx

from . import config
from .config import INFERENCE_SERVICE_URL, ML_SERVICE_URL

# Training can run for many hours at paper scale; make the client timeout
# configurable (seconds) rather than capping at 1 hour. Pair with TRAIN_JOB_TIMEOUT
# on the RQ worker (see jobqueue.py) so neither layer kills a long run.
TRAIN_TIMEOUT = float(os.environ.get("ML_TRAIN_TIMEOUT", "3600"))


def _ml_headers(extra: dict | None = None) -> dict:
    """Headers for an ML-plane request: the shared `Authorization: Bearer` token when
    config.ML_AUTH_TOKEN is set (else nothing — today's unauthenticated behavior), merged with any
    per-call `extra`. Read via the config module (not a captured value) so a test that sets the token
    after import is honored. Both ML services (:8001 train/onboard/retrieve, :8002 vLLM serve) enforce
    the same token, so every call path attaches it."""
    headers = dict(extra or {})
    token = config.ML_AUTH_TOKEN
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def train(
    corpus_dir: str,
    docs: list[dict],
    *,
    job_id: str | None = None,
    progress_url: str | None = None,
    progress_token: str | None = None,
    **knobs,
) -> dict:
    payload: dict = {"corpus_dir": corpus_dir, "docs": docs, **knobs}
    # Tell the worker where to post progress heartbeats (omitted = no callbacks).
    if job_id:
        payload["job_id"] = job_id
    if progress_url:
        payload["progress_url"] = progress_url
    if progress_token:
        payload["progress_token"] = progress_token
    resp = httpx.post(f"{ML_SERVICE_URL}/train", json=payload, timeout=TRAIN_TIMEOUT,
                      headers=_ml_headers())
    resp.raise_for_status()
    return resp.json()


def onboard_cag(
    corpus_dir: str,
    docs: list[dict],
    *,
    build_index: bool = False,
    job_id: str | None = None,
    progress_url: str | None = None,
    progress_token: str | None = None,
) -> dict:
    """Onboarding for the vLLM serve path: build a CAG cart per doc (one forward pass, no training) into
    the cartridge store. Same call shape/return as train(), so jobs.py handles both identically.
    `build_index` additionally builds the fused retrieval index (set when RETRIEVAL_BACKEND=fused)."""
    payload: dict = {"corpus_dir": corpus_dir, "docs": docs, "build_index": build_index}
    if job_id:
        payload["job_id"] = job_id
    if progress_url:
        payload["progress_url"] = progress_url
    if progress_token:
        payload["progress_token"] = progress_token
    resp = httpx.post(f"{ML_SERVICE_URL}/onboard_cag", json=payload, timeout=TRAIN_TIMEOUT,
                      headers=_ml_headers())
    resp.raise_for_status()
    return resp.json()


def retrieve_fused(corpus_dir: str, question: str, k: int = 3) -> list[str]:
    """Fused retrieval on the GPU box (RETRIEVAL_BACKEND=fused): the benchmark-winning
    BM25+dense+rerank pipeline. Returns ranked doc_ids; raises on HTTP errors (no silent
    fallback — a demo silently degrading to weaker retrieval would misrepresent the numbers)."""
    resp = httpx.post(
        f"{ML_SERVICE_URL}/retrieve",
        json={"corpus_dir": corpus_dir, "question": question, "k": k},
        timeout=120.0, headers=_ml_headers(),
    )
    resp.raise_for_status()
    return resp.json()["doc_ids"]


def query(corpus_dir: str, question: str, k: int = 3) -> dict:
    resp = httpx.post(
        f"{ML_SERVICE_URL}/query",
        json={"corpus_dir": corpus_dir, "question": question, "k": k},
        timeout=300.0, headers=_ml_headers(),
    )
    resp.raise_for_status()
    return resp.json()


def inference_query(doc_ids: list[str], question: str, max_tokens: int = 96,
                    history: list[dict] | None = None) -> dict:
    """Call the vLLM Inference Service (the resident-KV serving path): the control plane has already
    retrieved `doc_ids`; this service serves those carts and returns {answer, doc_ids}. Separate URL
    from ML_SERVICE_URL because it's a distinct GPU process (vLLM env)."""
    resp = httpx.post(
        f"{INFERENCE_SERVICE_URL}/query",
        json={"doc_ids": doc_ids, "question": question, "max_tokens": max_tokens,
              "history": history or []},
        timeout=300.0, headers=_ml_headers(),
    )
    resp.raise_for_status()
    return resp.json()


def inference_describe(doc_ids: list[str], max_tokens: int = 60) -> dict:
    """Ask the vLLM Inference Service for a one-sentence description of each doc_id, served from its
    resident CAG cart (POST /describe). Returns {descriptions: {doc_id: text-or-null}} — the service
    maps a per-doc failure to null and never fails the batch. Short timeout (120s): the caller's
    describe pass is best-effort and locally there's no inference service, so the call must fail FAST
    and be swallowed rather than hang onboarding."""
    resp = httpx.post(
        f"{INFERENCE_SERVICE_URL}/describe",
        json={"doc_ids": doc_ids, "max_tokens": max_tokens},
        timeout=120.0, headers=_ml_headers(),
    )
    resp.raise_for_status()
    return resp.json()


def inference_rag(context: str, question: str, max_tokens: int = 96,
                  history: list[dict] | None = None) -> dict:
    """Live RAG baseline on the SAME vLLM engine (measured head-to-head): prefill `context` + question,
    no cart. Returns {answer, metrics}. Used by the compare view for honest, apples-to-apples numbers."""
    resp = httpx.post(
        f"{INFERENCE_SERVICE_URL}/rag_query",
        json={"context": context, "question": question, "max_tokens": max_tokens,
              "history": history or []},
        timeout=300.0, headers=_ml_headers(),
    )
    resp.raise_for_status()
    return resp.json()


def compare(corpus_dir: str, question: str, k: int = 3) -> dict:
    """Answer one question several ways (cartridge alone / cart_rag floor / adaptive / rag)
    with measured latency + token counts. Longer timeout: the cart_rag + adaptive passes
    each generate over the retrieved documents."""
    resp = httpx.post(
        f"{ML_SERVICE_URL}/compare",
        json={"corpus_dir": corpus_dir, "question": question, "k": k},
        timeout=600.0, headers=_ml_headers(),
    )
    resp.raise_for_status()
    return resp.json()


# --- cartridge lifecycle: durable-blob deletion + serving-cache invalidation --------------------
# The two halves of "deleting a memory removes the document from serving": offboard() drops the
# blob at rest (ML service /offboard -> cartridge store); inference_invalidate() purges the warm KV
# on the serving engine (/invalidate). Callers run both (delete_corpus, the GC sweep). 60s timeout:
# these are metadata/cache ops, not GPU work — no need for the multi-hour TRAIN_TIMEOUT.


def offboard(doc_ids: list[str]) -> dict:
    """Delete carts by id from the durable cartridge store. Returns {deleted, missing}."""
    resp = httpx.post(f"{ML_SERVICE_URL}/offboard", json={"doc_ids": doc_ids},
                      timeout=60.0, headers=_ml_headers())
    resp.raise_for_status()
    return resp.json()


def inference_invalidate(cart_ids: list[str]) -> dict:
    """Purge serving-side (resident-KV) caches for these carts on the vLLM Inference Service, so a
    deleted — or force-rebuilt — cart can never be served from a stale warm cache. Returns
    {invalidated, backend}."""
    resp = httpx.post(f"{INFERENCE_SERVICE_URL}/invalidate", json={"cart_ids": cart_ids},
                      timeout=60.0, headers=_ml_headers())
    resp.raise_for_status()
    return resp.json()


def list_carts() -> dict:
    """Every cart id in the durable store — the GC sweep diffs this against still-referenced slugs.
    Returns {cart_ids}."""
    resp = httpx.get(f"{ML_SERVICE_URL}/carts", timeout=60.0, headers=_ml_headers())
    resp.raise_for_status()
    return resp.json()
