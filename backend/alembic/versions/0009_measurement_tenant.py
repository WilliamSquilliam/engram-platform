"""measurements.tenant_id: per-tenant query attribution (the billing fix)

Adds measurements.tenant_id (nullable, indexed) so a served-query row can be attributed to the
tenant whose corpus it ran against. The corpus-scoped serve paths (chat / mcp / compare) stamp it;
non-corpus callers and legacy rows recorded before this column stay NULL. Tenant-scoped usage/billing
filters on tenant_id == <tenant>; NULL rows are deployment-level and surface only in the platform
fleet totals. Nullable with no server default — existing rows simply carry NULL.

Revision ID: 0009_measurement_tenant
Revises: 0008_tenant_plan_status
Create Date: 2026-09-01
"""
import sqlalchemy as sa
from alembic import op

revision = "0009_measurement_tenant"
down_revision = "0008_tenant_plan_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("measurements", sa.Column("tenant_id", sa.String(), nullable=True))
    op.create_index("ix_measurements_tenant_id", "measurements", ["tenant_id"])


def downgrade() -> None:
    op.drop_index("ix_measurements_tenant_id", table_name="measurements")
    op.drop_column("measurements", "tenant_id")
