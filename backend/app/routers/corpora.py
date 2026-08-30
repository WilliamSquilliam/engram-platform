"""Corpus + document management. Every query is scoped to the caller's tenant."""
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from .. import ml_client
from ..audit import record_event
from ..config import MAX_REQUEST_MB, MAX_UPLOAD_MB
from ..deps import get_current_user, get_db
from ..models import Corpus, Document, User
from ..retrieval import doc_id_for
from ..schemas import CorpusCreateReq, CorpusResp, DocumentResp
from ..storage import safe_rel, storage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/corpora", tags=["corpora"])


def get_owned_corpus(db: Session, user: User, corpus_id: str) -> Corpus:
    corpus = db.get(Corpus, corpus_id)
    if corpus is None or corpus.tenant_id != user.tenant_id:
        raise HTTPException(404, "Corpus not found")
    return corpus


def _to_resp(db: Session, c: Corpus) -> CorpusResp:
    n = db.query(Document).filter(Document.corpus_id == c.id).count()
    return CorpusResp(id=c.id, name=c.name, source_type=c.source_type, status=c.status,
                      n_documents=n, mcp_token=c.mcp_token,
                      n_cartridges=c.n_cartridges, train_seconds=c.train_seconds,
                      corpus_tokens=c.corpus_tokens, created_at=c.created_at)


def deletable_slugs(db: Session, corpus_id: str, slugs: set[str]) -> tuple[list[str], set[str]]:
    """Given a corpus's own cart slugs, split them into (to_delete, shared) against the SHARED store.
    Cart ids are FILENAME SLUGS in a store shared across corpora/tenants (retrieval.doc_id_for), so a
    slug this corpus produced may still back a live document in ANOTHER corpus — deleting it would
    silently break that corpus's serving. Exclusion is computed in Python (doc_id_for isn't
    SQL-expressible; fine at this scale). Single source of truth so the mid-training compensation in
    jobs.py applies the EXACT same exclusion rule as the delete path."""
    shared = {doc_id_for(d.filename) for d in
              db.query(Document).filter(Document.corpus_id != corpus_id)} & slugs
    return sorted(slugs - shared), shared


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

    # n_documents counts Document ROWS of this corpus (not distinct slugs): two files that collide to
    # the same slug are still two documents, and the receipt reports what the user deleted.
    docs = db.query(Document).filter(Document.corpus_id == corpus.id).all()
    n_documents = len(docs)
    slugs = {doc_id_for(d.filename) for d in docs}
    to_delete, shared = deletable_slugs(db, corpus.id, slugs)

    # DB + storage deletion first (existing behavior, unchanged order): once these commit the data is
    # gone from the control plane's own stores regardless of what the ML plane does next.
    db.delete(corpus)
    db.commit()
    storage.delete_corpus(corpus_id)

    # ML-plane cleanup is BEST-EFFORT and must NOT fail the 204: the authoritative deletion (DB +
    # storage) already happened, so raising here would report failure for a delete that did occur. The
    # helper invalidates (tombstone fan-out purge) BEFORE offboarding the durable blob so a failure
    # degrades in the safe direction; see _ml_plane_cleanup for the full ordering rationale.
    ml_errors: dict = {}
    cart_result: dict = {}
    if to_delete:
        cart_result, ml_errors = _ml_plane_cleanup(db, corpus_id, to_delete)

    # Always one receipt. cart_ids_skipped_shared makes the collision exclusion auditable — a reviewer
    # sees those slugs were RETAINED on purpose (another live corpus still references the slug), not
    # missed. Corner case: if two slug-sharing corpora are deleted concurrently, each may see the other
    # still live and classify the slug as shared (both skip; never over-delete). The blob then backs no
    # live document and becomes an orphan the GC sweep (/internal/gc/carts) removes — so the receipt is
    # HONEST about what this delete retained even though GC ultimately reaps it.
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
        key, size = storage.save_document(corpus.id, fname, data)
        # Idempotent on (corpus, path): re-dropping a folder updates docs in place
        # instead of creating duplicate rows.
        doc = (
            db.query(Document)
            .filter(Document.corpus_id == corpus.id, Document.filename == fname)
            .first()
        )
        if doc is None:
            doc = Document(corpus_id=corpus.id, filename=fname, storage_key=key, size=size)
            db.add(doc)
        else:
            doc.storage_key, doc.size = key, size
        created.append(doc)
    # uploading new docs invalidates a previous "ready" state until retrained
    if corpus.status == "ready":
        corpus.status = "new"
    db.commit()
    return [DocumentResp(id=d.id, filename=d.filename, size=d.size) for d in created]


@router.get("/{corpus_id}/documents", response_model=list[DocumentResp])
def list_documents(corpus_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    corpus = get_owned_corpus(db, user, corpus_id)
    docs = db.query(Document).filter(Document.corpus_id == corpus.id).all()
    return [DocumentResp(id=d.id, filename=d.filename, size=d.size) for d in docs]
