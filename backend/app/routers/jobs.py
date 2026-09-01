"""Training jobs. Locally a job runs in a FastAPI BackgroundTask that calls the ML
service; on AWS this becomes an SQS message + Temporal workflow (C2/C3) — the API
contract (POST train, GET status) is identical, so the frontend doesn't change."""
import logging
import secrets

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from .. import config, jobqueue, ml_client
from ..audit import record_event
from ..db import SessionLocal
from ..deps import get_current_user, get_db
from ..models import Corpus, Document, Job, User
from ..ratelimit import limiter
from ..retrieval import cart_id_for
from ..schemas import GcCartsReq, JobResp, ProgressReq
from ..storage import storage
from .corpora import _ml_plane_cleanup, deletable_carts

logger = logging.getLogger(__name__)
router = APIRouter(tags=["jobs"])


def _job_resp(j: Job) -> JobResp:
    return JobResp(id=j.id, corpus_id=j.corpus_id, kind=j.kind, status=j.status,
                   detail=j.detail, progress=j.progress, eta_seconds=j.eta_seconds,
                   created_at=j.created_at, updated_at=j.updated_at)


def _compensate_if_deleted(corpus_id: str, tenant_id: str, run_cart_ids: list[str]) -> bool:
    """Race guard for the vLLM onboard path: a corpus DELETE can land WHILE an onboard is in flight, so
    the delete's ML-plane cleanup runs before this onboard has written its blobs — and then the onboard
    resurrects offboarded blobs that no live corpus references. Re-check (FRESH session) whether the
    corpus still exists; if it is GONE, run the same invalidate-then-offboard cleanup the delete path
    would have, over this run's cart ids MINUS ids still referenced by the SAME tenant's OTHER live
    corpora (identical exclusion rule via deletable_carts). The tenant_id was captured before onboard
    (the corpus row is cascaded away on delete), so the intra-tenant scoping still resolves.
    Returns True if it compensated (corpus gone), else False.

    Fully contained — a compensation failure must NOT crash the worker: it best-effort records whatever
    it can to a 'corpus.delete_compensation' receipt and swallows the rest."""
    db = SessionLocal()
    try:
        if db.get(Corpus, corpus_id) is not None:
            return False  # corpus still live — normal success path
        to_delete, shared = deletable_carts(db, tenant_id, corpus_id, set(run_cart_ids))
        cart_result: dict = {}
        ml_errors: dict = {}
        if to_delete:
            cart_result, ml_errors = _ml_plane_cleanup(db, corpus_id, to_delete)
        record_event(
            db, tenant_id=tenant_id, event="corpus.delete_compensation", corpus_id=corpus_id,
            cart_ids_deleted=cart_result.get("deleted", []),
            cart_ids_missing=cart_result.get("missing", []),
            cart_ids_skipped_shared=sorted(shared),
            ml_errors=ml_errors,
        )
        return True
    except Exception:  # noqa: BLE001 — compensation is best-effort; never crash the worker
        logger.exception("delete-compensation failed for corpus %s", corpus_id)
        return True  # corpus was gone; caller must still skip the (cascaded-away) row updates
    finally:
        db.close()


def _run_training(corpus_id: str, job_id: str) -> None:
    """Background worker: read docs, call the ML service, update job + corpus."""
    db = SessionLocal()
    try:
        corpus = db.get(Corpus, corpus_id)
        job = db.get(Job, job_id)
        if corpus is None or job is None:
            return  # corpus was deleted before/while training started — nothing to do
        # Capture the tenant BEFORE onboarding: if the corpus is deleted mid-run the row (and its
        # tenant_id) is cascaded away, but the compensation receipt still needs it.
        tenant_id = corpus.tenant_id
        try:
            filenames = storage.list_doc_filenames(corpus_id)
            docs = [{"doc_id": cart_id_for(tenant_id, fn), "text": storage.read_text(corpus_id, fn)}
                    for fn in filenames]
            docs = [d for d in docs if d["text"].strip()]
            if not docs:
                raise RuntimeError("no readable text documents in document base")
            progress_url = f"{config.BACKEND_INTERNAL_URL}/internal/jobs/{job_id}/progress"
            if config.INFERENCE_BACKEND == "vllm":
                # Resident-KV serving: onboard = build a CAG cart per doc (one forward pass, no
                # training) into the cartridge store the vLLM Inference Service serves from.
                result = ml_client.onboard_cag(
                    str(storage.corpus_dir(corpus_id)), docs,
                    build_index=config.RETRIEVAL_BACKEND == "fused",
                    job_id=job_id, progress_url=progress_url,
                    progress_token=config.INTERNAL_API_TOKEN or None,
                )
            else:  # HF path: per-doc training (or the amortized encoder)
                result = ml_client.train(
                    str(storage.corpus_dir(corpus_id)), docs,
                    job_id=job_id, progress_url=progress_url,
                    progress_token=config.INTERNAL_API_TOKEN or None,
                    cart_tokens=config.TRAIN_CART_TOKENS, steps=config.TRAIN_STEPS,
                    grad_accum=config.TRAIN_GRAD_ACCUM, gen_qs=config.TRAIN_GEN_QS,
                    method=config.ONBOARD_METHOD, encoder_ckpt=config.ENCODER_CKPT or None,
                )
            if result.get("canceled"):
                # Worker aborted cooperatively after the user requested cancel.
                # Corpus returns to "new" so it can be retrained from scratch.
                corpus.status = "new"
                # Onboarding wizard: a canceled onboard drops the cursor back to "review" so the
                # user re-confirms and restarts (documents/tier selections are preserved).
                if corpus.onboarding_step == "onboarding":
                    corpus.onboarding_step = "review"
                # Reset the per-doc ONBOARD state back to "pending" — a cancel means nothing onboarded,
                # so docs must not stay stuck at "onboarding" forever. parse_status (upload-time) is
                # left alone.
                for _d in db.query(Document).filter(Document.corpus_id == corpus_id):
                    _d.onboard_status = "pending"
                job.status = "canceled"
                job.detail = "Training canceled"
                job.eta_seconds = None
            elif config.INFERENCE_BACKEND == "vllm" and _compensate_if_deleted(
                    corpus_id, tenant_id, [d["doc_id"] for d in docs]):
                # A DELETE landed WHILE this onboard was in flight: the delete's ML-plane cleanup ran
                # against a store that did not yet hold these carts (or held the old ones), and then
                # our onboard resurrected them as durable blobs the deleted corpus no longer references.
                # _compensate_if_deleted re-checked with a fresh session, found the corpus GONE, and
                # invalidated+offboarded this run's now-orphaned slugs. The corpus/job rows were
                # cascaded away by the delete, so there is nothing more to update — skip the writes
                # below and return without touching (stale) ORM objects.
                return
            else:
                corpus.status = "ready"
                # Onboarding wizard: advance the cursor to the terminal "ready" step so a user who
                # left the wizard reopens on the finished screen. Only advance an onboard that was
                # actually in flight — a plain re-train (status path, no wizard) leaves it alone.
                if corpus.onboarding_step == "onboarding":
                    corpus.onboarding_step = "ready"
                # Per-file ONBOARD outcome (parse_status is an upload-time fact — never touched here).
                # A doc that extracted text is "ready"; a doc whose sidecar text was EMPTY was an
                # upload-time parse failure the onboard path already excluded (same non-empty filter as
                # the docs list above) — mark it "failed" so the wizard shows exactly which files didn't
                # make it, instead of falsely reporting the whole corpus onboarded.
                for _d in db.query(Document).filter(Document.corpus_id == corpus_id):
                    has_text = bool(storage.read_text(corpus_id, _d.filename).strip())
                    _d.onboard_status = "ready" if has_text else "failed"
                if not corpus.mcp_token:
                    corpus.mcp_token = secrets.token_urlsafe(24)
                # Persist timing/size from the run for the cost + break-even view.
                corpus.n_cartridges = result.get("n_cartridges")
                corpus.corpus_tokens = result.get("corpus_tokens")
                # train_seconds must reflect the ACTUAL cart-build GPU time (read-once cost), not the
                # wall-clock of a re-run where every cart was idempotently reused and only the retrieval
                # index was rebuilt (~100s) — that would understate onboarding ~50x. Use the measured
                # cart-build time when this run built carts; on a pure reuse-run (n_built==0) preserve
                # the value from the real build run instead of clobbering it.
                n_built = result.get("n_built")
                cart_s = result.get("cart_seconds")
                if n_built and cart_s:
                    corpus.train_seconds = cart_s
                elif corpus.train_seconds is None:
                    corpus.train_seconds = result.get("train_seconds")
                job.status = "succeeded"
                _how = ("via amortized encoder" if result.get("method") == "encoder"
                        else "via per-doc training")
                job.detail = (f"Onboarded {result.get('n_cartridges')} cartridges {_how} "
                              f"in {result.get('train_seconds')}s")
                job.progress = 1.0
                job.eta_seconds = 0
                # vLLM serve path: onboarding just (re)wrote a cart blob under each doc's slug. On a
                # FORCE re-onboard the new blob shares the SAME slug as the old one, so a serving
                # engine still holding the previous cart's warm KV would keep serving the OLD document
                # until eviction. Invalidate those ids so the fresh blob is what gets served. Best-effort
                # only — a failure here must never fail a job whose carts already built successfully.
                if config.INFERENCE_BACKEND == "vllm":
                    try:
                        ml_client.inference_invalidate([d["doc_id"] for d in docs])
                    except Exception as exc:  # noqa: BLE001 — cache purge is best-effort
                        logger.error("post-train invalidate failed for corpus %s: %s", corpus_id, exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Training failed for corpus %s (job %s)", corpus_id, job_id)
            corpus.status = "failed"
            # Onboarding wizard: a failed onboard returns the cursor to "review" so the user can
            # inspect and retry (the "onboarding" step is transient, only valid while a run is live).
            if corpus.onboarding_step == "onboarding":
                corpus.onboarding_step = "review"
            # The onboard failed, so every doc's ONBOARD state is "failed" (parse_status, an upload-time
            # fact, is left alone). Unconditional — a re-train that fails must also mark its docs failed,
            # not just a wizard-driven onboard.
            for _d in db.query(Document).filter(Document.corpus_id == corpus_id):
                _d.onboard_status = "failed"
            job.status = "failed"
            job.detail = str(exc)[:500]
            job.eta_seconds = None
        db.commit()
    finally:
        db.close()


def dispatch_training(db: Session, background: BackgroundTasks, corpus: Corpus) -> Job:
    """Flip the corpus to 'training', create the Job row, and enqueue the run via the configured job
    backend. Single source of truth for job dispatch so the onboarding flow (routers/onboarding.py)
    reuses the EXACT progress/cancel machinery instead of re-implementing it. Callers own the
    tenant-scoping + document/precondition checks; this only starts the run.

    Raises 409 if a run is already in flight (mirrors start_training's guard)."""
    if corpus.status == "training":
        raise HTTPException(409, "Training already in progress")
    corpus.status = "training"
    job = Job(corpus_id=corpus.id, kind="train", status="running", detail="Queued")
    db.add(job)
    db.commit()
    db.refresh(job)
    jobqueue.enqueue_training(background, corpus.id, job.id)
    return job


@router.post("/corpora/{corpus_id}/train", response_model=JobResp)
@limiter.limit("5/minute")  # onboarding occupies the GPU for minutes+; don't let jobs stack
def start_training(
    request: Request,
    corpus_id: str,
    background: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    corpus = db.get(Corpus, corpus_id)
    if corpus is None or corpus.tenant_id != user.tenant_id:
        raise HTTPException(404, "Document base not found")
    if not storage.list_doc_filenames(corpus_id):
        raise HTTPException(400, "Upload documents before training")
    return _job_resp(dispatch_training(db, background, corpus))


@router.post("/corpora/{corpus_id}/cancel", response_model=JobResp)
def cancel_training(corpus_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Request cancellation of the in-flight training run. We don't kill the GPU
    worker directly; we set a flag the worker reads from its progress-heartbeat
    response and then aborts cooperatively (maps to Temporal cancel on AWS)."""
    corpus = db.get(Corpus, corpus_id)
    if corpus is None or corpus.tenant_id != user.tenant_id:
        raise HTTPException(404, "Document base not found")
    job = (
        db.query(Job)
        .filter(Job.corpus_id == corpus_id, Job.status == "running")
        .order_by(Job.created_at.desc())
        .first()
    )
    if job is None:
        raise HTTPException(409, "No training run in progress")
    job.cancel_requested = True
    job.detail = "Canceling…"
    db.commit()
    db.refresh(job)
    return _job_resp(job)


@router.get("/jobs/{job_id}", response_model=JobResp)
def get_job(job_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    corpus = db.get(Corpus, job.corpus_id)
    if corpus is None or corpus.tenant_id != user.tenant_id:
        raise HTTPException(404, "Job not found")
    return _job_resp(job)


@router.get("/corpora/{corpus_id}/jobs", response_model=list[JobResp])
def list_jobs(corpus_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    corpus = db.get(Corpus, corpus_id)
    if corpus is None or corpus.tenant_id != user.tenant_id:
        raise HTTPException(404, "Document base not found")
    jobs = db.query(Job).filter(Job.corpus_id == corpus_id).order_by(Job.created_at.desc()).all()
    return [_job_resp(j) for j in jobs]


@router.post("/internal/jobs/{job_id}/progress")
def report_progress(
    job_id: str,
    body: ProgressReq,
    x_internal_token: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    """Worker -> control-plane heartbeat (NOT user-facing; guarded by a shared
    token, not the user JWT). The ML worker posts here every few steps so the UI
    can show a live progress bar + ETA. On AWS the Temporal worker emits the same.

    Fail CLOSED when INTERNAL_API_TOKEN is unset (mirrors gc_carts): an unauthenticated progress
    endpoint lets any caller drive a tenant's job progress/ETA, so the safe failure is to refuse to
    run at all (503) rather than skip the check."""
    if not config.INTERNAL_API_TOKEN:
        raise HTTPException(503, "progress reporting requires INTERNAL_API_TOKEN")
    if not secrets.compare_digest(config.INTERNAL_API_TOKEN, x_internal_token or ""):
        raise HTTPException(401, "invalid internal token")
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    job.progress = max(0.0, min(1.0, body.progress))
    job.eta_seconds = body.eta_seconds
    if body.detail:
        job.detail = body.detail[:500]
    db.commit()
    # The heartbeat response is the worker's only inbound channel: echo whether a
    # cancel was requested so the training loop can abort at the next checkpoint.
    return {"ok": True, "cancel": bool(job.cancel_requested)}


@router.post("/internal/gc/carts")
def gc_carts(
    body: GcCartsReq,
    x_internal_token: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    """OPERATOR-INVOKED cart garbage collection — reconcile the durable cartridge store against the
    documents this DB still references, deleting carts nothing references (orphans). NEVER scheduled:
    the store can legitimately hold carts onboarded OUTSIDE this DB (a separate onboarding run, or a
    fresh control-plane DB that has never seen them), so an automatic sweep would delete live carts this DB just
    doesn't know about. That is exactly why the DRY RUN is the default and `confirm=true` is required
    to delete — an operator confirms the orphan list is really garbage before anything is removed.

    Auth is the shared internal token (X-Internal-Token, constant-time compared), NOT a user JWT —
    this is a cross-tenant, store-wide operation. If INTERNAL_API_TOKEN is unset we 503 rather than
    expose an UNAUTHENTICATED store-wide delete: a public endpoint that can wipe every tenant's carts
    must never exist, so the safe failure is to refuse to run at all."""
    if not config.INTERNAL_API_TOKEN:
        raise HTTPException(503, "GC requires INTERNAL_API_TOKEN")
    if not secrets.compare_digest(config.INTERNAL_API_TOKEN, x_internal_token or ""):
        raise HTTPException(401, "invalid internal token")

    # Referenced = the TENANT-NAMESPACED cart id of EVERY document across ALL corpora/tenants (the
    # store is shared, so GC must consider the whole DB, not one tenant). Namespacing keeps GC
    # tenant-safe: an orphan is computed as (store ids − referenced ids), and because each id carries
    # its owning tenant's prefix, one tenant's live cart can never look like another tenant's orphan.
    # Join to each document's corpus for its tenant_id (cart_id_for isn't SQL-expressible).
    referenced = {cart_id_for(tenant_id, fn) for (tenant_id, fn) in
                  db.query(Corpus.tenant_id, Document.filename)
                  .join(Document, Document.corpus_id == Corpus.id).all()}
    store_ids = set(ml_client.list_carts().get("cart_ids", []))
    orphans = sorted(store_ids - referenced)

    if not body.confirm:
        # Dry run (the default): show what WOULD be deleted, delete nothing. Cap the returned list so a
        # huge store can't return an unbounded body; n_orphans carries the true total.
        return {"orphans": orphans[:500], "n_orphans": len(orphans), "deleted": False}

    # Confirmed: delete the durable blobs and purge any serving caches for them. Both best-effort in the
    # same spirit as delete_corpus — but here we surface counts so the operator sees the outcome.
    ml_errors: dict[str, str] = {}
    cart_result: dict = {}
    if orphans:
        try:
            cart_result = ml_client.offboard(orphans)
        except Exception as exc:  # noqa: BLE001 — record and continue; a later sweep retries
            logger.error("GC offboard failed: %s", exc)
            ml_errors["offboard"] = str(exc)
        try:
            ml_client.inference_invalidate(orphans)
        except Exception as exc:  # noqa: BLE001
            logger.error("GC invalidate failed: %s", exc)
            ml_errors["invalidate"] = str(exc)

    # "_system" tenant sentinel: the sweep is store-wide, owned by no tenant (see AuditEvent docstring).
    record_event(
        db, tenant_id="_system", event="carts.gc",
        n_orphans=len(orphans),
        cart_ids_deleted=cart_result.get("deleted", []),
        cart_ids_missing=cart_result.get("missing", []),
        ml_errors=ml_errors,
    )
    return {"n_orphans": len(orphans), "deleted": True,
            "cart_ids_deleted": cart_result.get("deleted", []),
            "cart_ids_missing": cart_result.get("missing", []),
            "ml_errors": ml_errors}
