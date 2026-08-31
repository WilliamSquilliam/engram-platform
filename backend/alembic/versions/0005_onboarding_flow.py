"""onboarding flow: resumable per-corpus wizard state + per-document onboard status

Adds the wizard cursor (corpora.onboarding_step), the chosen tier + pinned weights
(corpora.model_tier / model_ref), and per-document progress (documents.parse_status /
onboard_status). All columns carry server defaults so existing rows migrate cleanly.

Revision ID: 0005_onboarding_flow
Revises: 0004_audit_events
Create Date: 2026-08-31
"""
import sqlalchemy as sa
from alembic import op

revision = "0005_onboarding_flow"
down_revision = "0004_audit_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Wizard cursor: existing corpora predate the wizard, so default them to the entry step "name"
    # (a corpus with a name/docs already will simply skip ahead when the user opens it).
    op.add_column("corpora", sa.Column(
        "onboarding_step", sa.String(), nullable=False, server_default="name"))
    op.add_column("corpora", sa.Column("model_tier", sa.String(), nullable=True))
    op.add_column("corpora", sa.Column("model_ref", sa.String(), nullable=True))
    # Per-document onboard progress; existing docs start "pending" (they onboard on the next run).
    op.add_column("documents", sa.Column(
        "parse_status", sa.String(), nullable=False, server_default="pending"))
    op.add_column("documents", sa.Column(
        "onboard_status", sa.String(), nullable=False, server_default="pending"))


def downgrade() -> None:
    op.drop_column("documents", "onboard_status")
    op.drop_column("documents", "parse_status")
    op.drop_column("corpora", "model_ref")
    op.drop_column("corpora", "model_tier")
    op.drop_column("corpora", "onboarding_step")
