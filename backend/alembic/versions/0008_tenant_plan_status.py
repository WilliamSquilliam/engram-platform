"""tenant plan + status (E10/E11 dashboards)

Adds the two tenant billing/lifecycle columns the dashboards read:
  tenants.plan   -> billing plan ("beta" default; paid tiers in pricing.PLAN_LIMITS)
  tenants.status -> lifecycle status ("active" default; a platform_admin can suspend)

Both carry server defaults so existing tenant rows backfill on upgrade.

Revision ID: 0008_tenant_plan_status
Revises: 0007_auth_roles_invites
Create Date: 2026-08-31
"""
import sqlalchemy as sa
from alembic import op

revision = "0008_tenant_plan_status"
down_revision = "0007_auth_roles_invites"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # server_default so pre-existing tenants backfill; nullable=False for the app invariant.
    op.add_column(
        "tenants",
        sa.Column("plan", sa.String(), nullable=False, server_default="beta"),
    )
    op.add_column(
        "tenants",
        sa.Column("status", sa.String(), nullable=False, server_default="active"),
    )


def downgrade() -> None:
    with op.batch_alter_table("tenants") as batch:
        batch.drop_column("status")
        batch.drop_column("plan")
