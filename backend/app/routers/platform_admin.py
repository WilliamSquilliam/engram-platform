"""Platform-admin (founder / cross-tenant superuser) endpoints (E1).

Gated by require_platform_admin — a normal tenant admin is forbidden (403). These
drive the invite-only-beta approval flow: review pending access requests, then
approve (provision a tenant + its first admin user, hand back an accept-invite link)
or deny.
"""
import datetime
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from sqlalchemy import func

from .. import pricing, usage
from ..config import FRONTEND_URL, INVITE_EXPIRE_HOURS
from ..deps import get_db, require_platform_admin
from ..email import send_email
from ..models import AccessRequest, Corpus, Invite, Tenant, User
from ..schemas import (
    AccessRequestResp,
    LinkResp,
    PlatformTenantResp,
    PlatformTenantUsageResp,
    PlatformUsageResp,
    PlatformUsageTotalsResp,
)
from ..security import generate_token, hash_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/platform-admin", tags=["platform-admin"])


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC).replace(tzinfo=None)


def _invite_link(token: str) -> str:
    return f"{FRONTEND_URL}/accept-invite#token={token}"


@router.get("/access-requests", response_model=list[AccessRequestResp])
def list_access_requests(
    _: User = Depends(require_platform_admin), db: Session = Depends(get_db)
):
    """All still-pending access requests, newest first."""
    rows = (
        db.query(AccessRequest)
        .filter(AccessRequest.status == "pending")
        .order_by(AccessRequest.created_at.desc())
        .all()
    )
    return [
        AccessRequestResp(
            id=r.id, email=r.email, name=r.name, tenant_name=r.tenant_name,
            reason=r.reason, status=r.status, created_at=r.created_at,
        )
        for r in rows
    ]


@router.post("/access-requests/{request_id}/approve", response_model=LinkResp)
def approve_access_request(
    request_id: str,
    _: User = Depends(require_platform_admin),
    db: Session = Depends(get_db),
):
    """Approve a waitlist request: create the tenant, mint an admin-role invite for the
    requester (they set their own password via accept-invite), and return the link
    (when EMAIL_BACKEND=none) or email it. The user row itself is created on
    accept-invite, so no password is ever set on the requester's behalf."""
    req = db.get(AccessRequest, request_id)
    if req is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Access request not found")
    if req.status != "pending":
        raise HTTPException(status.HTTP_409_CONFLICT, f"Request already {req.status}")
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status.HTTP_409_CONFLICT, "A user with this email already exists")

    tenant = Tenant(name=req.tenant_name)
    db.add(tenant)
    db.flush()
    token = generate_token()
    db.add(Invite(
        tenant_id=tenant.id,
        email=req.email,
        role="admin",  # first user of the new tenant is its workspace admin
        token_hash=hash_token(token),
        expires_at=_now() + datetime.timedelta(hours=INVITE_EXPIRE_HOURS),
    ))
    req.status = "approved"
    db.commit()

    link = _invite_link(token)
    sent = send_email(req.email, "Your access is approved", f"Set up your account: {link}")
    logger.info("Approved access request %s for %s", request_id, req.email)
    return LinkResp(status="approved", invite_link=None if sent else link)


@router.post("/access-requests/{request_id}/deny", response_model=LinkResp)
def deny_access_request(
    request_id: str,
    _: User = Depends(require_platform_admin),
    db: Session = Depends(get_db),
):
    """Deny a waitlist request (idempotent for an already-denied row)."""
    req = db.get(AccessRequest, request_id)
    if req is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Access request not found")
    if req.status == "approved":
        raise HTTPException(status.HTTP_409_CONFLICT, "Request already approved")
    req.status = "denied"
    db.commit()
    logger.info("Denied access request %s for %s", request_id, req.email)
    return LinkResp(status="denied")


# --- E11: platform-admin console (CROSS-tenant) --------------------------
# The one place that INTENTIONALLY spans tenants. Both routes are gated by
# require_platform_admin (a normal tenant admin 403s) — the hard gate is the only thing
# standing between one tenant and every tenant's data, so it stays on every route here.


@router.get("/tenants", response_model=list[PlatformTenantResp])
def list_tenants(
    _: User = Depends(require_platform_admin), db: Session = Depends(get_db)
):
    """Every tenant on the platform with headline counts (users, corpora) + plan/status. Counts
    are computed with grouped queries (not N+1) and merged onto the tenant list."""
    tenants = db.query(Tenant).order_by(Tenant.created_at.desc()).all()

    user_counts = dict(
        db.query(User.tenant_id, func.count(User.id)).group_by(User.tenant_id).all()
    )
    corpus_counts = dict(
        db.query(Corpus.tenant_id, func.count(Corpus.id)).group_by(Corpus.tenant_id).all()
    )
    return [
        PlatformTenantResp(
            id=t.id,
            name=t.name,
            created_at=t.created_at,
            n_users=int(user_counts.get(t.id, 0)),
            n_corpora=int(corpus_counts.get(t.id, 0)),
            plan=t.plan,
            status=t.status,
        )
        for t in tenants
    ]


@router.get("/usage", response_model=PlatformUsageResp)
def platform_usage(
    _: User = Depends(require_platform_admin), db: Session = Depends(get_db)
):
    """Fleet usage: per-tenant documents/storage/gpu-seconds + estimated cost, and the fleet totals
    (which are exactly the sum of the per-tenant rows). Cost per tenant uses the SAME
    pricing.estimate_cost_usd the tenant billing shell uses, so a tenant's bill and its line in this
    view tie out. Queries are the deployment-level served-query count (Measurement is global), so
    the same total is reported against the fleet rather than split per tenant."""
    tenants = db.query(Tenant).order_by(Tenant.created_at.desc()).all()

    fleet_queries = usage.total_query_count(db)  # deployment-global served-query count

    rows: list[PlatformTenantUsageResp] = []
    sum_storage = 0.0
    sum_gpu = 0.0
    sum_cost = 0.0
    for t in tenants:
        documents = usage.tenant_document_count(db, t.id)
        storage_gb = usage.tenant_storage_gb(db, t.id)
        gpu_seconds = usage.tenant_gpu_seconds(db, t.id)
        # Per-tenant est cost from the shared rate card. Queries aren't attributable per tenant, so
        # only the tenant-scoped facts (storage + documents) drive the per-tenant line-item cost.
        est_cost = pricing.estimate_cost_usd(
            queries=0, storage_gb=storage_gb, documents=documents
        )
        rows.append(PlatformTenantUsageResp(
            tenant_id=t.id,
            name=t.name,
            queries=0,
            documents=documents,
            storage_gb=storage_gb,
            gpu_seconds=gpu_seconds,
            est_cost_usd=est_cost,
        ))
        sum_storage += storage_gb
        sum_gpu += gpu_seconds
        sum_cost += est_cost

    totals = PlatformUsageTotalsResp(
        # queries is the deployment-level served-query count (Measurement is global) — reported as a
        # fleet signal, NOT a sum of the per-tenant rows (those are 0, since queries aren't attributable
        # per tenant). storage/gpu/cost totals ARE exactly the sum of the per-tenant rows, so the fleet
        # cost ties out to the line items a reviewer can add up.
        queries=fleet_queries,
        storage_gb=round(sum_storage, 4),
        gpu_seconds=round(sum_gpu, 1),
        est_cost_usd=round(sum_cost, 2),
        n_tenants=len(tenants),
    )
    return PlatformUsageResp(tenants=rows, totals=totals)
