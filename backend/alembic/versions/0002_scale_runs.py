"""scale_runs: saved fleet scale-test runs (per corpus)

Revision ID: 0002_scale_runs
Revises: 0001_initial
Create Date: 2026-07-23
"""
import sqlalchemy as sa
from alembic import op

revision = "0002_scale_runs"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "scale_runs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("corpus_id", sa.String(), sa.ForeignKey("corpora.id"), nullable=False),
        sa.Column("max_concurrency", sa.Integer(), nullable=False),
        sa.Column("n_queries", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("points", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_scale_runs_corpus_id", "scale_runs", ["corpus_id"])


def downgrade() -> None:
    op.drop_index("ix_scale_runs_corpus_id", table_name="scale_runs")
    op.drop_table("scale_runs")
