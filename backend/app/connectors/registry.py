"""Connector registry — the config-driven catalog of ingestion SOURCES the product offers.

Mirrors serving.py's placeholder-gated registry pattern: every connector ships as a
product-facing entry with an `available` flag. `filesystem` (plain upload) is always
available; `google_drive` and `sharepoint` render as "coming soon" (available=False) until
their OAuth app credentials are configured via env (config.*_ENABLED) — flipping one on is a
config change, no code change.

This is the product-catalog layer for GET /connectors. It is DISTINCT from the runtime
`Connector` ABC (base.py) + `get_connector` factory (__init__.py): the ABC/factory is the
seam that ingests bytes once a source is wired; this registry is what the UI lists as
available/coming-soon. Only `filesystem` has a live runtime implementation today; the OAuth
connectors are interface scaffolds (see google_drive.py / sharepoint.py) gated off here until
their OAuth dance is built.
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import config
from . import google_drive, sharepoint


@dataclass(frozen=True)
class ConnectorInfo:
    """A product-facing ingestion source for the 'connect a source' menu.

    `available` decides whether the UI offers it now or shows "coming soon". No field names an
    OAuth secret — availability is derived from config so a credential leak can't happen through
    this contract."""
    id: str            # stable id: "filesystem" | "google_drive" | "sharepoint"
    label: str         # UI label, e.g. "Google Drive"
    description: str    # one line for the menu
    available: bool    # selectable now (filesystem always; the rest need OAuth creds configured)


def _connectors() -> list[ConnectorInfo]:
    """Build the catalog fresh from current config so a test (or a redeploy) that sets creds sees
    the connector flip to available without a module reload."""
    return [
        ConnectorInfo(
            id="filesystem",
            label="File upload",
            description="Upload files or drop a folder (.txt .md .pdf .docx .doc .html .xlsx .xls .csv).",
            available=True,  # the always-on path; no external credentials needed
        ),
        # Availability = the connector's IMPLEMENTED flag AND its creds+enc-key gate (config.*_ENABLED,
        # which already require CONNECTOR_ENC_KEY). Creds alone must never flip a connector on. The
        # description doubles as the OPERATOR setup note (redirect URI + scopes to register with the
        # provider) — the same values app/oauth.py sends.
        ConnectorInfo(
            id="google_drive",
            label="Google Drive",
            description=(
                "Connect a Drive folder and import its documents. Operator setup: a Google Cloud OAuth "
                "client (GDRIVE_CLIENT_ID/SECRET, or reuse GOOGLE_*), scope "
                "drive.readonly, redirect URI "
                f"{config.API_BASE_URL}/connectors/google_drive/callback, and CONNECTOR_ENC_KEY set."
            ),
            available=google_drive.IMPLEMENTED and config.GDRIVE_ENABLED,
        ),
        ConnectorInfo(
            id="sharepoint",
            label="SharePoint",
            description=(
                "Connect a SharePoint site or library and import its documents. Operator setup: a "
                "multi-tenant Entra ID app (SHAREPOINT_CLIENT_ID/SECRET), delegated scopes "
                "offline_access Files.Read.All Sites.Read.All User.Read, redirect URI "
                f"{config.API_BASE_URL}/connectors/sharepoint/callback, and CONNECTOR_ENC_KEY set."
            ),
            available=sharepoint.IMPLEMENTED and config.SHAREPOINT_ENABLED,
        ),
    ]


def connectors() -> list[ConnectorInfo]:
    """All product connectors for the menu (coming-soon ones included)."""
    return _connectors()


def connector(connector_id: str) -> ConnectorInfo | None:
    return next((c for c in _connectors() if c.id == connector_id), None)


def available_connectors() -> list[ConnectorInfo]:
    """Connectors a user can actually pick right now."""
    return [c for c in _connectors() if c.available]
