"""Tenant-admin workspace management (E1): members + teammate invites.

Every route is gated by require_tenant_admin (a tenant `member` gets 403) and scoped
to the caller's own tenant — an admin can only ever see/modify users and invites of
their own workspace. Invites are gated the same way email is: when EMAIL_BACKEND=none
the invite_link is returned in the response; otherwise it's emailed and not leaked.
"""
import datetime
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..config import FRONTEND_URL, INVITE_EXPIRE_HOURS
from ..deps import get_db, require_tenant_admin
from ..email import send_email
from ..models import Invite, User
from ..schemas import (
    InviteCreateReq,
    LinkResp,
    MemberResp,
    MembersResp,
    PendingInviteResp,
    RoleUpdateReq,
)
from ..security import generate_token, hash_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC).replace(tzinfo=None)


def _invite_link(token: str) -> str:
    return f"{FRONTEND_URL}/accept-invite#token={token}"


@router.get("/members", response_model=MembersResp)
def list_members(
    admin: User = Depends(require_tenant_admin), db: Session = Depends(get_db)
):
    """Members of the caller's tenant + invites that haven't been accepted yet."""
    members = db.query(User).filter(User.tenant_id == admin.tenant_id).all()
    pending = (
        db.query(Invite)
        .filter(Invite.tenant_id == admin.tenant_id, Invite.accepted_at.is_(None))
        .all()
    )
    return MembersResp(
        members=[
            MemberResp(id=u.id, email=u.email, role=u.role, is_active=u.is_active)
            for u in members
        ],
        invites=[
            PendingInviteResp(id=i.id, email=i.email, role=i.role, created_at=i.created_at)
            for i in pending
        ],
    )


@router.post("/invites", response_model=LinkResp)
def create_invite(
    req: InviteCreateReq,
    admin: User = Depends(require_tenant_admin),
    db: Session = Depends(get_db),
):
    """Invite a teammate into the caller's workspace. Returns the accept-invite link
    when EMAIL_BACKEND=none; otherwise emails it and omits it from the response."""
    # Already a member of THIS tenant? Nothing to do.
    existing = (
        db.query(User)
        .filter(User.email == req.email, User.tenant_id == admin.tenant_id)
        .first()
    )
    if existing is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "User is already a member")

    token = generate_token()
    invite = Invite(
        tenant_id=admin.tenant_id,
        email=req.email,
        role=req.role,
        token_hash=hash_token(token),
        expires_at=_now() + datetime.timedelta(hours=INVITE_EXPIRE_HOURS),
        created_by=admin.id,
    )
    db.add(invite)
    db.commit()

    link = _invite_link(token)
    sent = send_email(req.email, "You're invited", f"Accept your invitation: {link}")
    return LinkResp(status="sent", invite_link=None if sent else link)


@router.delete("/invites/{invite_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_invite(
    invite_id: str,
    admin: User = Depends(require_tenant_admin),
    db: Session = Depends(get_db),
):
    """Revoke a pending invite. Tenant-scoped: an admin can only revoke their own
    tenant's invites (a cross-tenant id 404s, not 403, to avoid leaking existence)."""
    invite = db.get(Invite, invite_id)
    if invite is None or invite.tenant_id != admin.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Invite not found")
    db.delete(invite)
    db.commit()


@router.patch("/members/{user_id}", response_model=MemberResp)
def update_member_role(
    user_id: str,
    req: RoleUpdateReq,
    admin: User = Depends(require_tenant_admin),
    db: Session = Depends(get_db),
):
    """Change a member's workspace role (admin<->member), scoped to the tenant. An
    admin can't demote themselves if they're the last admin (avoids locking the
    workspace out of its own /admin/*)."""
    target = db.get(User, user_id)
    if target is None or target.tenant_id != admin.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Member not found")
    if req.role != "admin" and target.id == admin.id:
        n_admins = (
            db.query(User)
            .filter(User.tenant_id == admin.tenant_id, User.role == "admin")
            .count()
        )
        if n_admins <= 1:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "Cannot demote the last admin of the workspace"
            )
    target.role = req.role
    db.commit()
    return MemberResp(id=target.id, email=target.email, role=target.role, is_active=target.is_active)


@router.delete("/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_member(
    user_id: str,
    admin: User = Depends(require_tenant_admin),
    db: Session = Depends(get_db),
):
    """Remove a member from the workspace (tenant-scoped). An admin can't remove
    themselves via this endpoint, and can't remove the last admin."""
    target = db.get(User, user_id)
    if target is None or target.tenant_id != admin.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Member not found")
    if target.id == admin.id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Cannot remove yourself")
    if target.role == "admin":
        n_admins = (
            db.query(User)
            .filter(User.tenant_id == admin.tenant_id, User.role == "admin")
            .count()
        )
        if n_admins <= 1:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "Cannot remove the last admin of the workspace"
            )
    db.delete(target)
    db.commit()
