"""Add users.name — the display name collected (required) at accept-invite.

Nullable: accounts that predate the column (register / early Google sign-ins) have no
name; the accept-invite API requires it going forward.

Revision ID: 0010_user_name
Revises: 0009_measurement_tenant
"""
import sqlalchemy as sa
from alembic import op

revision = "0010_user_name"
down_revision = "0009_measurement_tenant"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("name", sa.String(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.drop_column("name")
