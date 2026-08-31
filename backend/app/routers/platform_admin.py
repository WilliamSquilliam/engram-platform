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

from ..config import FRONTEND_URL, INVITE_EXPIRE_HOURS
from ..deps import get_db, require_platform_admin
from ..email import send_email
from ..models import AccessRequest, Invite, Tenant, User
from ..schemas import AccessRequestResp, LinkResp
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
