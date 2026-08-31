"""Resumable per-corpus onboarding wizard (the 5-step flow: name -> documents -> model -> review ->
onboard). State is persisted on the Corpus/Document rows so a user can exit and reopen at the exact
step, and onboarding runs SERVER-SIDE via the existing job path (routers/jobs.dispatch_training).

The wizard cursor (corpus.onboarding_step) is distinct from the corpus lifecycle (corpus.status):
the cursor is where the USER is; the status is where the WORK is. The two only touch at the start
(onboard flips status to 'training') and end (the worker sets both to their terminal values).

Placeholder-tier reality: no serving engine is enabled yet (serving.py ships all tiers disabled), so
step 5 is GATED — /onboard returns {"status": "no_serving_engine"} (HTTP 409) and dispatches nothing
until a tier is available. Steps 1-4 are fully functional and testable with no GPU.
"""
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from .. import metrics, serving
from ..deps import get_current_user, get_db
from ..models import Corpus, Document, User
from ..schemas import OnboardingPatchReq, OnboardingStateResp
from ..storage import storage
from .corpora import _doc_resp, get_owned_corpus
from .jobs import dispatch_training

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/corpora", tags=["onboarding"])


def _state(db: Session, corpus: Corpus) -> OnboardingStateResp:
    """The resumable-onboarding snapshot: wizard cursor + tier + per-document status."""
    docs = db.query(Document).filter(Document.corpus_id == corpus.id).all()
    return OnboardingStateResp(
        corpus_id=corpus.id,
        onboarding_step=corpus.onboarding_step,
        status=corpus.status,
        model_tier=corpus.model_tier,
        model_ref=corpus.model_ref,
        n_documents=len(docs),
        documents=[_doc_resp(d) for d in docs],
    )


@router.get("/{corpus_id}/onboarding", response_model=OnboardingStateResp)
def get_onboarding(corpus_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Read the wizard state so the frontend reopens at the right step with per-file progress."""
    return _state(db, get_owned_corpus(db, user, corpus_id))


@router.patch("/{corpus_id}/onboarding", response_model=OnboardingStateResp)
def patch_onboarding(
    corpus_id: str,
    req: OnboardingPatchReq,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Persist the wizard cursor and/or the chosen model tier as the user moves through the steps.
    `model_tier` is validated against the serving registry (unknown ids rejected) — a placeholder
    (disabled) tier is still a VALID selection here so the review step can show it as 'coming soon';
    the availability gate is enforced at /onboard, not at selection time."""
    corpus = get_owned_corpus(db, user, corpus_id)
    if req.model_tier is not None:
        if serving.tier(req.model_tier) is None:
            raise HTTPException(400, f"Unknown model tier '{req.model_tier}'")
        corpus.model_tier = req.model_tier
    if req.onboarding_step is not None:
        corpus.onboarding_step = req.onboarding_step
    db.commit()
    return _state(db, corpus)


@router.get("/{corpus_id}/estimate")
def estimate(corpus_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Review step (4): a pre-run sizing summary — doc count, detected file types (from filename
    extensions), total bytes, and a coarse estimated onboarding time + cost. The estimate constants
    live in ONE place (metrics.onboard_estimate); the real figures land on the corpus after the run
    (see /corpora/{id}/economics)."""
    corpus = get_owned_corpus(db, user, corpus_id)
    docs = db.query(Document).filter(Document.corpus_id == corpus.id).all()
    total_bytes = sum(d.size for d in docs)
    # Detected file types = lowercased extension of each filename, "none" when there is no extension.
    file_types: dict[str, int] = {}
    for d in docs:
        ext = d.filename.rsplit(".", 1)[-1].lower() if "." in d.filename else "none"
        file_types[ext] = file_types.get(ext, 0) + 1
    return {
        "n_documents": len(docs),
        "total_bytes": total_bytes,
        "file_types": file_types,
        "model_tier": corpus.model_tier,
        **metrics.onboard_estimate(len(docs)),
    }


@router.post("/{corpus_id}/onboard", response_model=OnboardingStateResp)
def onboard(
    corpus_id: str,
    background: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Step 5: start onboarding server-side. Resolve the chosen tier to concrete weights and dispatch
    via the EXISTING job path (jobs.dispatch_training -> ml_client onboard/train, progress + cancel
    reused). The worker sets onboarding_step='ready' on success.

    GATE (current placeholder reality): if the chosen tier is not `available` (no enabled serving
    engine yet), dispatch NOTHING — return {"status": "no_serving_engine"} at HTTP 409 and leave the
    cursor at 'review' so the UI shows 'onboarding starts once a model is enabled'. This keeps the
    whole flow buildable/testable through step 4 with no GPU."""
    corpus = get_owned_corpus(db, user, corpus_id)

    if not corpus.model_tier:
        raise HTTPException(400, "Select a model tier before onboarding")
    tier = serving.tier(corpus.model_tier)
    if tier is None:
        raise HTTPException(400, f"Unknown model tier '{corpus.model_tier}'")
    if not storage.list_doc_filenames(corpus_id):
        raise HTTPException(400, "Upload documents before onboarding")

    # Availability gate: a placeholder / disabled tier has no live engine to serve carts, so onboarding
    # cannot start. Return a STRUCTURED 409 body (not an error string) so the UI shows "onboarding
    # starts once a model is enabled"; the cursor stays at "review" (nothing dispatched).
    if not tier.available:
        corpus.onboarding_step = "review"
        db.commit()
        return JSONResponse(status_code=409, content={"status": "no_serving_engine", "tier": tier.id})

    # Tier is live: pin the resolved weights the carts are stamped to (model-binding), advance the
    # wizard cursor, mark docs as onboarding, then dispatch through the shared job machinery.
    corpus.model_ref = serving.model_ref_for_tier(corpus.model_tier)
    corpus.onboarding_step = "onboarding"
    for d in db.query(Document).filter(Document.corpus_id == corpus.id):
        d.parse_status = "parsing"
        d.onboard_status = "onboarding"
    dispatch_training(db, background, corpus)  # flips status->training, enqueues, commits
    db.refresh(corpus)
    return _state(db, corpus)
