"""auth self-serve + roles/invites + waitlist (E1)

Adds the two-tier authz + invite-only-beta plumbing:
  users: platform_admin (founder / cross-tenant superuser), email_verified,
         is_active. `role` server-default flips admin -> member (the first user of a
         tenant is set to admin in code; teammates default to member).
  access_requests: the public waitlist a platform_admin approves.
  invites:  pending invitations (teammate invites AND approval invites). Only the
            token HASH is stored.
  password_resets: single-use reset grants; only the token HASH is stored.

Revision ID: 0007_auth_roles_invites
Revises: 0006_parse_error
Create Date: 2026-08-31
"""
import sqlalchemy as sa
from alembic import op

revision = "0007_auth_roles_invites"
down_revision = "0006_parse_error"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- users: new authz columns -----------------------------------------
    # server_default so existing rows backfill; nullable=False for the app invariant.
    op.add_column(
        "users",
        sa.Column("platform_admin", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "users",
        sa.Column("email_verified", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "users",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    # New users default to "member" at the DB level; the first user of a tenant is
    # promoted to "admin" in application code. Existing single-user tenants created
    # under the old default were already "admin" (0001 server_default="admin").
    # batch_alter_table so SQLite (no ALTER COLUMN) does this via table-copy too.
    with op.batch_alter_table("users") as batch:
        batch.alter_column("role", server_default="member")

    # --- access_requests: the invite-only waitlist ------------------------
    op.create_table(
        "access_requests",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("tenant_name", sa.String(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_access_requests_email", "access_requests", ["email"])
    op.create_index("ix_access_requests_status", "access_requests", ["status"])

    # --- invites: teammate + approval invitations (hashed token) ----------
    op.create_table(
        "invites",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("tenant_id", sa.String(), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False, server_default="member"),
        sa.Column("token_hash", sa.String(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("accepted_at", sa.DateTime(), nullable=True),
        sa.Column("created_by", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_invites_tenant_id", "invites", ["tenant_id"])
    op.create_index("ix_invites_email", "invites", ["email"])
    op.create_index("ix_invites_token_hash", "invites", ["token_hash"])

    # --- password_resets: single-use reset grants (hashed token) ----------
    op.create_table(
        "password_resets",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("token_hash", sa.String(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("used_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_password_resets_user_id", "password_resets", ["user_id"])
    op.create_index("ix_password_resets_token_hash", "password_resets", ["token_hash"])


def downgrade() -> None:
    op.drop_index("ix_password_resets_token_hash", table_name="password_resets")
    op.drop_index("ix_password_resets_user_id", table_name="password_resets")
    op.drop_table("password_resets")

    op.drop_index("ix_invites_token_hash", table_name="invites")
    op.drop_index("ix_invites_email", table_name="invites")
    op.drop_index("ix_invites_tenant_id", table_name="invites")
    op.drop_table("invites")

    op.drop_index("ix_access_requests_status", table_name="access_requests")
    op.drop_index("ix_access_requests_email", table_name="access_requests")
    op.drop_table("access_requests")

    with op.batch_alter_table("users") as batch:
        batch.alter_column("role", server_default="admin")
        batch.drop_column("is_active")
        batch.drop_column("email_verified")
        batch.drop_column("platform_admin")
