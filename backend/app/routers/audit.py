"""Audit-trail read API. Exposes the caller's own tenant's lifecycle receipts (see app/audit.py) so
an operator — or a security reviewer — can inspect what was deleted, when, and by whom. Tenant-scoped
like every other user-facing route: the JWT's tenant is the only tenant whose rows are ever returned."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..deps import get_current_user, get_db
from ..models import AuditEvent, User
from ..schemas import AuditEventResp

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("", response_model=list[AuditEventResp])
def list_audit_events(
    limit: int = Query(100, ge=1, le=1000),  # bounded so a client can't page the whole table at once
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """The caller's tenant's audit events, newest first. `limit` defaults to 100 (max 1000). Filtered
    strictly on the JWT's tenant_id — a tenant never sees another tenant's (or the _system GC) receipts."""
    rows = (
        db.query(AuditEvent)
        .filter(AuditEvent.tenant_id == user.tenant_id)
        .order_by(AuditEvent.created_at.desc(), AuditEvent.id.desc())
        .limit(limit)
        .all()
    )
    return [AuditEventResp(id=r.id, event=r.event, corpus_id=r.corpus_id,
                           detail=r.detail, created_at=r.created_at) for r in rows]
