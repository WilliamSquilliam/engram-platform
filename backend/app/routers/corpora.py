"""Corpus + document management. Every query is scoped to the caller's tenant."""
import datetime
import logging

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from .. import config, limits, ml_client, usage
from ..audit import record_event
from ..config import MAX_REQUEST_MB, MAX_UPLOAD_MB
from ..connectors import providers
from ..db import SessionLocal
from ..deps import get_current_user, get_db
from ..models import ConnectorConnection, Corpus, Document, ImportRun, Tenant, User
from ..parsing import SUPPORTED_EXTS, extract_text
from ..retrieval import cart_id_for, invalidate_index
from ..schemas import (
    CorpusCreateReq,
    CorpusResp,
    DocumentResp,
    ImportReq,
    ImportStatusResp,
)
from ..storage import safe_rel, storage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/corpora", tags=["corpora"])


def _now() -> datetime.datetime:
    # Naive UTC to match the DateTime columns (see models._now).
    return datetime.datetime.now(datetime.UTC).replace(tzinfo=None)


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


# --- import from a connected source (Google Drive / SharePoint) --------------------------------
# Import walks a connected folder and feeds each supported file through the SAME save path uploads use
# (storage.save_document + a Document row + text extraction), so imported and uploaded documents are
# indistinguishable downstream. It runs in a background thread (the codebase's parse/train background
# pattern) and reports progress via an ImportRun row (GET .../import-status).

def _save_imported_file(db: Session, corpus: Corpus, rel_path: str, data: bytes) -> None:
    """Persist one imported file EXACTLY like an upload: raw bytes to storage, extracted text sidecar,
    and an idempotent Document row keyed on (corpus, path). The doc-limit check happens in the worker
    BEFORE this is called, so a save here is already within the cap."""
    fname = safe_rel(rel_path)
    key, size = storage.save_document(corpus.id, fname, data)
    text, ok, parse_error = extract_text(fname, data)
    storage.save_text(corpus.id, fname, text if ok else "")
    parse_status = "parsed" if ok else "failed"
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


def _walk_for(provider: str, token: str, folder_id: str, site_id: str | None, max_bytes: int):
    """The provider-specific recursive file walk (Drive vs Graph), yielding (rel_path, downloader)."""
    if provider == "google_drive":
        return providers.drive_walk(token, folder_id, max_bytes)
    return providers.graph_walk(token, folder_id, site_id, max_bytes)


def _run_import(run_id: str, connection_id: str, folder_id: str, site_id: str | None) -> None:
    """Background worker: walk the connected folder and import every supported file, updating the
    ImportRun counters as it goes. Contract:
      - Enforce the beta document limit BEFORE each save; on 429 stop gracefully, state='limited'
        (what imported so far is KEPT).
      - Oversized (> MAX_UPLOAD_MB) or unsupported files -> skipped++ (never a crash).
      - A per-file error -> failed++, continue to the next file.
      - A 401 on a download -> refresh the token once and retry that file.
    Runs in its own Session (background thread), like the training worker."""
    db = SessionLocal()
    try:
        run = db.get(ImportRun, run_id)
        if run is None:
            return
        corpus = db.get(Corpus, run.corpus_id)
        conn = db.get(ConnectorConnection, connection_id)
        if corpus is None or conn is None:
            run.state = "failed"
            run.error = "The document base or connection no longer exists."
            run.finished_at = _now()
            db.commit()
            return
        tenant = db.get(Tenant, corpus.tenant_id)
        # Read the cap off config (not the import-time constant) so it honors an env/test override.
        max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
        try:
            token = providers.access_token(db, conn)
            for rel_path, download in _walk_for(conn.provider, token, folder_id, site_id, max_bytes):
                # Beta document cap: check with the LIVE count before each save so a run stops at the
                # cap with everything before it kept. incoming=1 (this file).
                try:
                    if tenant is not None:
                        limits.check_document_limit(
                            tenant, usage.tenant_document_count(db, corpus.tenant_id), incoming=1
                        )
                except HTTPException as exc:
                    if exc.status_code == 429:
                        run.state = "limited"
                        run.error = "Reached the workspace document limit; imported files were kept."
                        break
                    raise
                # Download (size-capped stream). ImportSizeExceeded -> skipped. A 401 -> refresh once.
                try:
                    data = download(token)
                except providers.ImportSizeExceeded:
                    run.skipped += 1
                    db.commit()
                    continue
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 401:
                        token = providers.access_token(db, conn, force_refresh=True)
                        try:
                            data = download(token)
                        except providers.ImportSizeExceeded:
                            run.skipped += 1
                            db.commit()
                            continue
                    else:
                        run.failed += 1
                        db.commit()
                        continue
                try:
                    _save_imported_file(db, corpus, rel_path, data)
                    run.imported += 1
                except Exception:  # noqa: BLE001 — one bad file must not abort the whole import
                    logger.exception("import: failed to save %s", rel_path)
                    run.failed += 1
                db.commit()
            else:
                # Loop finished without hitting the limit-break.
                run.state = "done"
            # A "documents" re-import invalidates a prior "ready" state (new docs to onboard), mirroring
            # the upload path so the wizard doesn't sit on a stale terminal screen.
            if run.imported and corpus.status == "ready":
                corpus.status = "new"
                corpus.onboarding_step = "documents"
        except HTTPException as exc:
            # A clean provider/config error (503 needs-reconfig, 401 expired) surfaced as a failed run.
            run.state = "failed"
            run.error = str(exc.detail)[:400]
        except Exception as exc:  # noqa: BLE001 — any unexpected error fails the run cleanly, no 500 path
            logger.exception("import run %s failed", run_id)
            run.state = "failed"
            run.error = f"Import failed ({type(exc).__name__})."
        if run.state == "running":  # safety net if we broke out without setting a terminal state
            run.state = "done"
        run.finished_at = _now()
        db.commit()
    finally:
        db.close()


@router.post("/{corpus_id}/import", response_model=ImportStatusResp)
def start_import(
    corpus_id: str,
    req: ImportReq,
    background: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Import a connected source's folder into this document base. Requires ownership of the base AND
    that the connection belong to the same tenant (a cross-tenant connection id 404s). 409 if an import
    for this base is already running. The walk + downloads happen in the background; poll import-status."""
    get_owned_corpus(db, user, corpus_id)  # 404s if not this tenant's base
    conn = db.get(ConnectorConnection, req.connection_id)
    if conn is None or conn.tenant_id != user.tenant_id:
        raise HTTPException(404, "Connection not found")
    running = (
        db.query(ImportRun)
        .filter(ImportRun.corpus_id == corpus_id, ImportRun.state == "running")
        .first()
    )
    if running is not None:
        raise HTTPException(409, "An import for this document base is already running")
    run = ImportRun(
        corpus_id=corpus_id, connection_id=conn.id, folder_id=req.folder_id or "",
        folder_name=req.folder_name or "", state="running",
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    background.add_task(_run_import, run.id, conn.id, req.folder_id or "", req.site_id)
    return _import_status_resp(run)


@router.get("/{corpus_id}/import-status", response_model=ImportStatusResp)
def import_status(corpus_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Latest import run for this document base, or {"state": "none"} if nothing has been imported."""
    get_owned_corpus(db, user, corpus_id)
    run = (
        db.query(ImportRun)
        .filter(ImportRun.corpus_id == corpus_id)
        .order_by(ImportRun.created_at.desc())
        .first()
    )
    if run is None:
        return ImportStatusResp(state="none")
    return _import_status_resp(run)


def _import_status_resp(run: ImportRun) -> ImportStatusResp:
    return ImportStatusResp(
        state=run.state, id=run.id, folder_name=run.folder_name,
        imported=run.imported, skipped=run.skipped, failed=run.failed, error=run.error,
        created_at=run.created_at, finished_at=run.finished_at,
    )
