"""SharePointConnector — OAuth-gated ingestion scaffold (interface only, not wired).

Follows the FilesystemConnector template (discover -> fetch -> delta) so the ingest path
consumes it identically once the Microsoft Graph app + calls are implemented. Today it is a
SCAFFOLD: constructing it without configured credentials raises ConnectorNotConfigured, and the
API methods raise NotImplementedError. The registry (registry.py) gates it off (available=False,
"coming soon") until config.SHAREPOINT_ENABLED, so this is never reached in the live product yet.

When built out: discover() lists a document library via Microsoft Graph (paged), fetch()
downloads + parses each item (binary formats through app.parsing), and delta() uses Graph's
delta query token as the incremental cursor — the same shape FilesystemConnector proves with
mtime.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator

from .base import Connector, Document, SourceRef
from .google_drive import ConnectorNotConfigured

# Live via routers/connectors.py + app/connectors/providers.py (Graph REST). The registry ANDs this
# with the creds+enc-key check (see the matching flag in google_drive.py). This class stays the
# ingestion-seam scaffold; the import path uses the providers module directly.
IMPLEMENTED = True


class SharePointConnector(Connector):
    source_type = "sharepoint"

    def __init__(self, credentials: dict | None = None, site_id: str | None = None):
        if not credentials:
            raise ConnectorNotConfigured(
                "SharePoint connector requires OAuth credentials "
                "(SHAREPOINT_CLIENT_ID/SECRET/TENANT_ID)"
            )
        self.credentials = credentials
        self.site_id = site_id

    def discover(self) -> Iterable[SourceRef]:
        raise NotImplementedError("SharePoint ingestion is not implemented yet")

    def fetch(self, refs: Iterable[SourceRef]) -> Iterator[Document]:
        raise NotImplementedError("SharePoint ingestion is not implemented yet")

    def delta(self, cursor: str | None) -> tuple[Iterable[SourceRef], str]:
        raise NotImplementedError("SharePoint ingestion is not implemented yet")
