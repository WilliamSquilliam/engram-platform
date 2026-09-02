"""tenant billing (Stripe dark-launch) + beta-limit overrides

Adds the columns the dark-launched Stripe billing + the invisible beta limits need:
  tenants.stripe_customer_id       -> Stripe customer id, created lazily on first portal open (null)
  tenants.max_docs_override        -> per-tenant document cap override (null = use config default)
  tenants.max_queries_override     -> per-tenant monthly-query cap override (null = use config default)
  tenants.billing_reported_queries -> high-water mark of queries already reported to Stripe's meter,
                                      so each report pushes only the delta (idempotent metering)

The three overrides/mark are nullable / server-defaulted so existing tenant rows backfill on upgrade
with no billing state. Nothing here charges anyone — billing stays DISABLED via config.BILLING_ENABLED.

Revision ID: 0012_tenant_billing
Revises: 0011_document_description
Create Date: 2026-09-02
"""
import sqlalchemy as sa
from alembic import op

revision = "0012_tenant_billing"
down_revision = "0011_document_description"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tenants", sa.Column("stripe_customer_id", sa.String(), nullable=True))
    op.add_column("tenants", sa.Column("max_docs_override", sa.Integer(), nullable=True))
    op.add_column("tenants", sa.Column("max_queries_override", sa.Integer(), nullable=True))
    # server_default="0" so pre-existing tenants backfill; nullable=False matches the app invariant
    # (the delta calc reads it as an int, never None).
    op.add_column(
        "tenants",
        sa.Column("billing_reported_queries", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    with op.batch_alter_table("tenants") as batch:
        batch.drop_column("billing_reported_queries")
        batch.drop_column("max_queries_override")
        batch.drop_column("max_docs_override")
        batch.drop_column("stripe_customer_id")
