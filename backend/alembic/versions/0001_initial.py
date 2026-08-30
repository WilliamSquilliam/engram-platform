"""initial schema: tenants, users, corpora, documents, jobs

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-14
"""
import sqlalchemy as sa
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tenants",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_table(
        "users",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("tenant_id", sa.String(), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False, server_default="admin"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_table(
        "corpora",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("tenant_id", sa.String(), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("source_type", sa.String(), nullable=False, server_default="upload"),
        sa.Column("status", sa.String(), nullable=False, server_default="new"),
        sa.Column("mcp_token", sa.String(), nullable=True),
        sa.Column("n_cartridges", sa.Integer(), nullable=True),
        sa.Column("train_seconds", sa.Float(), nullable=True),
        sa.Column("corpus_tokens", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_corpora_tenant_id", "corpora", ["tenant_id"])
    op.create_table(
        "documents",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("corpus_id", sa.String(), sa.ForeignKey("corpora.id"), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("storage_key", sa.String(), nullable=False),
        sa.Column("size", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_documents_corpus_id", "documents", ["corpus_id"])
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("corpus_id", sa.String(), sa.ForeignKey("corpora.id"), nullable=False),
        sa.Column("kind", sa.String(), nullable=False, server_default="train"),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("detail", sa.Text(), nullable=False, server_default=""),
        sa.Column("progress", sa.Float(), nullable=False, server_default="0"),
        sa.Column("eta_seconds", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_jobs_corpus_id", "jobs", ["corpus_id"])


def downgrade() -> None:
    op.drop_table("jobs")
    op.drop_table("documents")
    op.drop_table("corpora")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
    op.drop_table("tenants")
