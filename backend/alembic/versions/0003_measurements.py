"""measurements: durable per-query measured metrics (cart vs rag head-to-heads)

Revision ID: 0003_measurements
Revises: 0002_scale_runs
Create Date: 2026-07-27
"""
import sqlalchemy as sa
from alembic import op

revision = "0003_measurements"
down_revision = "0002_scale_runs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "measurements",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("side", sa.String(), nullable=False),  # "cart" | "rag"
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("ttft_ms", sa.Float(), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("resident_kv_tokens", sa.Integer(), nullable=True),
        sa.Column("gen_tokens", sa.Integer(), nullable=True),
        sa.Column("decode_tps", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("cost_per_query", sa.Float(), nullable=True),
        sa.Column("model_label", sa.String(), nullable=True),
        sa.Column("instance_label", sa.String(), nullable=True),
    )
    op.create_index("ix_measurements_created_at", "measurements", ["created_at"])
    op.create_index("ix_measurements_side", "measurements", ["side"])


def downgrade() -> None:
    op.drop_index("ix_measurements_side", table_name="measurements")
    op.drop_index("ix_measurements_created_at", table_name="measurements")
    op.drop_table("measurements")
