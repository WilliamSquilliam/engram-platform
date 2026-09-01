"""GoogleDriveConnector — OAuth-gated ingestion scaffold (interface only, not wired).

Follows the FilesystemConnector template (discover -> fetch -> delta) so the ingest path
consumes it identically once the OAuth dance + Drive API calls are implemented. Today it is a
SCAFFOLD: constructing it without configured credentials raises ConnectorNotConfigured, and the
API methods raise NotImplementedError. The registry (registry.py) gates it off (available=False,
"coming soon") until config.GDRIVE_ENABLED, so this is never reached in the live product yet.

When built out: discover() lists Drive files via the Drive v3 API (paged), fetch() downloads +
parses each (Google Docs export to text; binary formats through app.parsing), and delta() uses
Drive's changes.list pageToken as the incremental cursor — the same shape FilesystemConnector
proves with mtime.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator

from .base import Connector, Document, SourceRef

# Flip to True when the OAuth dance + discover/fetch/delta are actually built. The registry
# ANDs this with the creds check — creds alone must never light the connector up (caught in
# the e2e run-up: a configured Google OAuth client made the UI offer a NotImplementedError).
IMPLEMENTED = False


class ConnectorNotConfigured(RuntimeError):
    """Raised when a connector is constructed without its OAuth credentials configured."""


class GoogleDriveConnector(Connector):
    source_type = "google_drive"

    def __init__(self, credentials: dict | None = None, folder_id: str | None = None):
        # Gate at construction: no credentials -> this connector isn't available (the registry
        # already shows it "coming soon"; this is the belt-and-suspenders runtime guard).
        if not credentials:
            raise ConnectorNotConfigured(
                "Google Drive connector requires OAuth credentials (GDRIVE_CLIENT_ID/SECRET)"
            )
        self.credentials = credentials
        self.folder_id = folder_id

    def discover(self) -> Iterable[SourceRef]:
        raise NotImplementedError("Google Drive ingestion is not implemented yet")

    def fetch(self, refs: Iterable[SourceRef]) -> Iterator[Document]:
        raise NotImplementedError("Google Drive ingestion is not implemented yet")

    def delta(self, cursor: str | None) -> tuple[Iterable[SourceRef], str]:
        raise NotImplementedError("Google Drive ingestion is not implemented yet")
