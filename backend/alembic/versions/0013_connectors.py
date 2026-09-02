"""connector connections + import runs

Adds the two tables the Google Drive / SharePoint connectors need:
  connector_connections -> a tenant's authorized link to a source; OAuth tokens stored ENCRYPTED
                           (Fernet ciphertext columns, never plaintext). One row per
                           (tenant, provider, account); reconnecting the same account upserts.
  import_runs           -> one folder-import run into a corpus (the import-status surface):
                           state + imported/skipped/failed counters, updated by the background worker.

Indexes mirror the query paths: connector_connections by tenant_id + provider (list this tenant's
connections), import_runs by corpus_id + connection_id (latest run for a corpus / the running guard).

Revision ID: 0013_connectors
Revises: 0012_tenant_billing
Create Date: 2026-09-02
"""
import sqlalchemy as sa
from alembic import op

revision = "0013_connectors"
down_revision = "0012_tenant_billing"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "connector_connections",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("tenant_id", sa.String(), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("account_label", sa.String(), nullable=False),
        # Text (not String): Fernet ciphertext of an OAuth token is well over a short-String length.
        sa.Column("enc_refresh_token", sa.Text(), nullable=False),
        sa.Column("enc_access_token", sa.Text(), nullable=True),
        sa.Column("token_expires_at", sa.DateTime(), nullable=True),
        sa.Column("created_by", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "ix_connector_connections_tenant_id", "connector_connections", ["tenant_id"]
    )
    op.create_index(
        "ix_connector_connections_provider", "connector_connections", ["provider"]
    )

    op.create_table(
        "import_runs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("corpus_id", sa.String(), sa.ForeignKey("corpora.id"), nullable=False),
        sa.Column("connection_id", sa.String(), nullable=False),
        sa.Column("folder_id", sa.String(), nullable=False),
        sa.Column("folder_name", sa.String(), nullable=False),
        sa.Column("state", sa.String(), nullable=False, server_default="running"),
        sa.Column("imported", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("skipped", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_import_runs_corpus_id", "import_runs", ["corpus_id"])
    op.create_index("ix_import_runs_connection_id", "import_runs", ["connection_id"])


def downgrade() -> None:
    op.drop_index("ix_import_runs_connection_id", table_name="import_runs")
    op.drop_index("ix_import_runs_corpus_id", table_name="import_runs")
    op.drop_table("import_runs")
    op.drop_index("ix_connector_connections_provider", table_name="connector_connections")
    op.drop_index("ix_connector_connections_tenant_id", table_name="connector_connections")
    op.drop_table("connector_connections")
