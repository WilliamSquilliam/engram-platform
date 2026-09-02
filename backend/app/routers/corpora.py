"""Corpus + document management. Every query is scoped to the caller's tenant."""
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from .. import limits, ml_client, usage
from ..audit import record_event
from ..config import MAX_REQUEST_MB, MAX_UPLOAD_MB
from ..deps import get_current_user, get_db
from ..models import Corpus, Document, Tenant, User
from ..parsing import SUPPORTED_EXTS, extract_text
from ..retrieval import cart_id_for, invalidate_index
from ..schemas import CorpusCreateReq, CorpusResp, DocumentResp
from ..storage import safe_rel, storage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/corpora", tags=["corpora"])


def get_owned_corpus(db: Session, user: User, corpus_id: str) -> Corpus:
    corpus = db.get(Corpus, corpus_id)
    if corpus is None or corpus.tenant_id != user.tenant_id:
        raise HTTPException(404, "Document base not found")
    return corpus


def _to_resp(db: Session, c: Corpus) -> CorpusResp:
    n = db.query(Document).filter(Document.corpus_id == c.id).count()
    return CorpusResp(id=c.id, name=c.name, source_type=c.source_type, status=c.status,
                      n_documents=n, onboarding_step=c.onboarding_step, model_tier=c.model_tier,
                      mcp_token=c.mcp_token,
                      n_cartridges=c.n_cartridges, train_seconds=c.train_seconds,
                      corpus_tokens=c.corpus_tokens, created_at=c.created_at)


def _doc_resp(d: Document) -> DocumentResp:
    return DocumentResp(id=d.id, filename=d.filename, size=d.size,
                        parse_status=d.parse_status, parse_error=d.parse_error,
                        onboard_status=d.onboard_status, description=d.description)


def deletable_carts(db: Session, tenant_id: str, corpus_id: str,
                    cart_ids: set[str]) -> tuple[list[str], set[str]]:
    """Given a corpus's own (tenant-namespaced) cart ids, split them into (to_delete, shared).

    Cart ids are now namespaced by tenant (retrieval.cart_id_for), so cross-TENANT sharing is gone —
    two tenants uploading the same filename get DIFFERENT carts and never collide. What can still be
    shared is INTRA-tenant: the SAME tenant reusing a document across its own corpora resolves to one
    cart id, so deleting one corpus must not offboard a cart another corpus of the same tenant still
    serves. So the 'shared' check scopes to the SAME tenant's OTHER corpora only (was: every corpus of
    every tenant). Exclusion is computed in Python (cart_id_for isn't SQL-expressible; fine at this
    scale). Single source of truth so the mid-training compensation in jobs.py applies the EXACT same
    exclusion rule as the delete path."""
    sibling_docs = (
        db.query(Document.filename)
        .join(Corpus, Document.corpus_id == Corpus.id)
        .filter(Corpus.tenant_id == tenant_id, Document.corpus_id != corpus_id)
    )
    shared = {cart_id_for(tenant_id, fn) for (fn,) in sibling_docs} & cart_ids
    return sorted(cart_ids - shared), shared


def _ml_plane_cleanup(db: Session, corpus_id: str, to_delete: list[str]) -> tuple[dict, dict]:
    """Invalidate-then-offboard the given slugs, best-effort. Returns (cart_result, ml_errors).
    ORDER MATTERS: tombstones (published by inference_invalidate) are the durable, fan-out purge signal
    — they evict warm serving caches AND mirror copies on every box. Publishing them BEFORE deleting
    the durable blob means a failed offboard degrades to 'stale blob until GC' (safe: GC diffs the
    store and removes it later). The old order (offboard first) had the opposite failure: a failed
    invalidate left deleted-document KV warm in serving caches with NO automatic recovery — GC computes
    orphans as (store ids − referenced ids), and the blob is already gone, so it can never be reclassed
    an orphan and re-invalidated. Recovery from a failed invalidate is an operator re-running
    POST /invalidate with the ids from the audit receipt (see PARTNER_OPERATIONS.md, engram-dynamics-landing repo); GC does NOT
    re-invalidate. Each call is wrapped separately so an invalidate failure still lets offboard run.

    Errors are SANITIZED: the audit detail is tenant-visible (GET /audit), and raw httpx errors embed
    internal ML-plane URLs — record only the exception CLASS name + a fixed phrase, and log the full
    exception server-side. On invalidate failure the detail carries invalidate_failed + the affected ids
    so an operator can replay the purge."""
    ml_errors: dict = {}
    cart_result: dict = {}
    try:
        ml_client.inference_invalidate(to_delete)
    except Exception as exc:  # noqa: BLE001 — best-effort; operator replays /invalidate from receipt ids
        logger.exception("invalidate failed for corpus %s", corpus_id)
        ml_errors["invalidate"] = f"invalidate call failed ({type(exc).__name__})"
        ml_errors["invalidate_failed"] = True
        ml_errors["invalidate_failed_ids"] = list(to_delete)
    try:
        cart_result = ml_client.offboard(to_delete)
    except Exception as exc:  # noqa: BLE001 — best-effort; GC reconciles the stale blob later
        logger.exception("offboard failed for corpus %s", corpus_id)
        ml_errors["offboard"] = f"offboard call failed ({type(exc).__name__})"
    return cart_result, ml_errors


@router.post("", response_model=CorpusResp)
def create_corpus(req: CorpusCreateReq, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    corpus = Corpus(tenant_id=user.tenant_id, name=req.name, source_type=req.source_type)
    db.add(corpus)
    db.commit()
    return _to_resp(db, corpus)


@router.get("", response_model=list[CorpusResp])
def list_corpora(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(Corpus).filter(Corpus.tenant_id == user.tenant_id).order_by(Corpus.created_at.desc()).all()
    return [_to_resp(db, c) for c in rows]


@router.get("/{corpus_id}", response_model=CorpusResp)
def get_corpus(corpus_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return _to_resp(db, get_owned_corpus(db, user, corpus_id))


@router.delete("/{corpus_id}", status_code=204)
def delete_corpus(corpus_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Delete a corpus and everything under it — the FULL data-deletion path that makes 'deleting a
    memory removes the document from serving' true end-to-end. The DB cascade removes its documents +
    jobs; storage.delete_corpus removes the on-disk docs tree; then the ML plane drops the durable cart
    blobs (ml_client.offboard) and purges the serving engine's warm KV (inference_invalidate).
    A training run still in flight is harmless — its worker no-ops when it finds the corpus gone, and
    if it deleted MID-onboard it runs a compensating cleanup (see jobs._run_training)."""
    corpus = get_owned_corpus(db, user, corpus_id)

    # n_documents counts Document ROWS of this corpus (not distinct cart ids): two files that collide
    # to the same cart id are still two documents, and the receipt reports what the user deleted.
    docs = db.query(Document).filter(Document.corpus_id == corpus.id).all()
    n_documents = len(docs)
    cart_ids = {cart_id_for(corpus.tenant_id, d.filename) for d in docs}
    to_delete, shared = deletable_carts(db, corpus.tenant_id, corpus.id, cart_ids)

    # DB + storage deletion first (existing behavior, unchanged order): once these commit the data is
    # gone from the control plane's own stores regardless of what the ML plane does next.
    db.delete(corpus)
    db.commit()
    storage.delete_corpus(corpus_id)
    invalidate_index(corpus_id)  # free the in-process hybrid index for this corpus

    # ML-plane cleanup is BEST-EFFORT and must NOT fail the 204: the authoritative deletion (DB +
    # storage) already happened, so raising here would report failure for a delete that did occur. The
    # helper invalidates (tombstone fan-out purge) BEFORE offboarding the durable blob so a failure
    # degrades in the safe direction; see _ml_plane_cleanup for the full ordering rationale.
    ml_errors: dict = {}
    cart_result: dict = {}
    if to_delete:
        cart_result, ml_errors = _ml_plane_cleanup(db, corpus_id, to_delete)

    # Always one receipt. cart_ids_skipped_shared makes the intra-tenant exclusion auditable — a
    # reviewer sees those carts were RETAINED on purpose (another corpus OF THE SAME TENANT still
    # references the cart), not missed. Cross-tenant sharing can't happen now (ids are tenant-namespaced),
    # so this only ever lists same-tenant reuse. Corner case: if two corpora sharing a cart are deleted
    # concurrently, each may see the other still live and classify it shared (both skip; never
    # over-delete). The blob then backs no live document and becomes an orphan the GC sweep
    # (/internal/gc/carts) removes — so the receipt is HONEST about what this delete retained.
    record_event(
        db, tenant_id=user.tenant_id, user_id=user.id, event="corpus.delete", corpus_id=corpus_id,
        n_documents=n_documents,
        cart_ids_deleted=cart_result.get("deleted", []),
        cart_ids_missing=cart_result.get("missing", []),
        cart_ids_skipped_shared=sorted(shared),
        ml_errors=ml_errors,
    )
    return None


@router.post("/{corpus_id}/documents", response_model=list[DocumentResp])
async def upload_documents(
    corpus_id: str,
    files: list[UploadFile] = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    corpus = get_owned_corpus(db, user, corpus_id)
    # Beta document limit (invisible until hit): reject BEFORE saving anything if this workspace's
    # document count + the incoming files would cross the cap, so a request that would exceed it is
    # rejected whole (never a partial upload). Counted per-workspace (across all the tenant's corpora);
    # incoming is len(files) — an upper bound (empties are skipped below), so we never let a request past.
    tenant = db.get(Tenant, user.tenant_id)
    if tenant is not None:
        limits.check_document_limit(
            tenant, usage.tenant_document_count(db, user.tenant_id), incoming=len(files)
        )
    created = []
    max_file = MAX_UPLOAD_MB * 1024 * 1024
    max_total = MAX_REQUEST_MB * 1024 * 1024
    total = 0
    for f in files:
        data = await f.read()
        if not data:
            continue  # skip empties (e.g. folder placeholders / .DS_Store stripped client-side)
        if len(data) > max_file:
            raise HTTPException(413, f"{f.filename} exceeds the {MAX_UPLOAD_MB} MB per-file limit")
        total += len(data)
        if total > max_total:
            raise HTTPException(413, f"Upload exceeds the {MAX_REQUEST_MB} MB total limit")
        # Normalize once and use the SAME key for storage and the DB row, so the
        # dedup check below matches what's actually on disk (and a missing
        # filename — allowed by multipart — can't crash the upload).
        fname = safe_rel(f.filename or "document.txt")
        ext = ("." + fname.rsplit(".", 1)[-1].lower()) if "." in fname else ""
        if ext not in SUPPORTED_EXTS:
            # Reject an unsupported type at the edge (before storing) so we never keep raw
            # bytes we can't onboard. The message lists what we DO accept.
            raise HTTPException(
                400,
                f"{fname}: unsupported file type. Accepted: "
                + " ".join(sorted(SUPPORTED_EXTS)),
            )
        # Keep the raw file (for download / re-parse), then extract text. The extracted
        # text is persisted as a sidecar so the onboard + retrieval paths read WORDS, not
        # raw PDF/DOCX bytes (storage.read_text prefers the sidecar). On FAILURE we write an
        # EMPTY sidecar: read_text then returns "" (dropped by the onboard path's non-empty
        # filter) instead of falling back to a lossy decode of the raw binary — a failed-parse
        # document must never be onboarded as garbage bytes.
        key, size = storage.save_document(corpus.id, fname, data)
        text, ok, parse_error = extract_text(fname, data)
        storage.save_text(corpus.id, fname, text if ok else "")
        parse_status = "parsed" if ok else "failed"
        # Idempotent on (corpus, path): re-dropping a folder updates docs in place
        # instead of creating duplicate rows.
        doc = (
            db.query(Document)
            .filter(Document.corpus_id == corpus.id, Document.filename == fname)
            .first()
        )
        if doc is None:
            doc = Document(corpus_id=corpus.id, filename=fname, storage_key=key, size=size,
                           parse_status=parse_status, parse_error=parse_error or None)
            db.add(doc)
        else:
            doc.storage_key, doc.size = key, size
            doc.parse_status, doc.parse_error = parse_status, parse_error or None
        created.append(doc)
    # uploading new docs invalidates a previous "ready" state until retrained. Also drop the wizard
    # cursor back to the "documents" step so the two stay coherent — a re-upload after a finished
    # onboard shouldn't leave the user parked on the terminal "ready" screen with unonboarded docs.
    if corpus.status == "ready":
        corpus.status = "new"
        corpus.onboarding_step = "documents"
    db.commit()
    return [_doc_resp(d) for d in created]


@router.get("/{corpus_id}/documents", response_model=list[DocumentResp])
def list_documents(corpus_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    corpus = get_owned_corpus(db, user, corpus_id)
    docs = db.query(Document).filter(Document.corpus_id == corpus.id).all()
    return [_doc_resp(d) for d in docs]
