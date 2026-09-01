"""Add documents.description — the one-sentence LLM description written at onboarding (Feature 1).

Nullable: null until the (flag-gated, best-effort) describe pass fills it after a wizard onboard, and
it stays null when descriptions are off or the pass fails — onboarding succeeds regardless.

Revision ID: 0011_document_description
Revises: 0010_user_name
"""
import sqlalchemy as sa
from alembic import op

revision = "0011_document_description"
down_revision = "0010_user_name"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("description", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("documents") as batch:
        batch.drop_column("description")
