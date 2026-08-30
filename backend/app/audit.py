"""Audit trail for data-lifecycle actions. One row per action, append-only.

Lifecycle actions (a corpus delete, the GC cart sweep, an offboard that failed and got deferred to
GC) must be PROVABLE after the fact: this is the deletion receipt a security reviewer asks for when
the privacy pitch ('deleting a memory removes the document from serving') is put to the test. The
row is self-contained — `detail` carries the compact JSON facts (which cart ids were deleted, which
were retained because another live corpus still references them, which ML-plane calls errored) — so
it stays meaningful even though the delete already removed the live rows those facts describe."""
import json
import logging

from sqlalchemy.orm import Session

from .models import AuditEvent

logger = logging.getLogger(__name__)


def record_event(db: Session, *, tenant_id: str, event: str,
                 user_id: str | None = None, corpus_id: str | None = None, **detail) -> AuditEvent:
    """Write one audit row and commit it. `**detail` is JSON-dumped into the `detail` column (sort_keys
    for a stable, diffable receipt; default=str so a stray non-serializable value degrades to its repr
    rather than raising and losing the receipt). Returns the persisted row."""
    row = AuditEvent(
        tenant_id=tenant_id, user_id=user_id, event=event, corpus_id=corpus_id,
        detail=json.dumps(detail, sort_keys=True, default=str),
    )
    db.add(row)
    db.commit()
    return row
