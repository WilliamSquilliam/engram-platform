"""audit_events: append-only lifecycle receipts (corpus.delete, carts.gc, offboard failures)

Revision ID: 0004_audit_events
Revises: 0003_measurements
Create Date: 2026-07-31
"""
import sqlalchemy as sa
from alembic import op

revision = "0004_audit_events"
down_revision = "0003_measurements"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "audit_events",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("tenant_id", sa.String(), nullable=False),  # "_system" for operator/GC events
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("event", sa.String(), nullable=False),
        sa.Column("corpus_id", sa.String(), nullable=True),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_audit_events_tenant_id", "audit_events", ["tenant_id"])
    op.create_index("ix_audit_events_event", "audit_events", ["event"])
    op.create_index("ix_audit_events_created_at", "audit_events", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_audit_events_created_at", table_name="audit_events")
    op.drop_index("ix_audit_events_event", table_name="audit_events")
    op.drop_index("ix_audit_events_tenant_id", table_name="audit_events")
    op.drop_table("audit_events")
