"""parse_error: per-document text-extraction failure reason

Adds documents.parse_error (nullable) so a document that fails text extraction
(unsupported type, encrypted, corrupt) carries a short human-readable reason the
onboarding wizard can show. Nullable with no server default — existing rows and
successfully-parsed docs simply have NULL.

Revision ID: 0006_parse_error
Revises: 0005_onboarding_flow
Create Date: 2026-08-31
"""
import sqlalchemy as sa
from alembic import op

revision = "0006_parse_error"
down_revision = "0005_onboarding_flow"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("parse_error", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "parse_error")
