"""Source connectors: the pluggable ingestion seam (see base.Connector).

`get_connector(source_type, **kwargs)` is the factory the ingest path uses so adding a
new source (SharePoint/Confluence/Drive/S3) is a registry entry, not a code change at
the call site.
"""
from __future__ import annotations

from .base import Connector, Document, SourceRef
from .filesystem import FilesystemConnector
from .google_drive import ConnectorNotConfigured, GoogleDriveConnector
from .registry import (
    ConnectorInfo, available_connectors, connector, connectors,
)
from .sharepoint import SharePointConnector

#: source_type -> Connector class. New connectors register here. The OAuth connectors are
#: registered so the factory can construct them, but they gate themselves off at __init__
#: (ConnectorNotConfigured) until creds are set — the product-facing availability is decided
#: by the registry above (registry.connectors()), which GET /connectors serves.
REGISTRY: dict[str, type[Connector]] = {
    FilesystemConnector.source_type: FilesystemConnector,
    GoogleDriveConnector.source_type: GoogleDriveConnector,
    SharePointConnector.source_type: SharePointConnector,
}


def get_connector(source_type: str, **kwargs) -> Connector:
    try:
        cls = REGISTRY[source_type]
    except KeyError as e:
        raise ValueError(
            f"unknown source_type {source_type!r}; known: {sorted(REGISTRY)}"
        ) from e
    return cls(**kwargs)


__all__ = [
    "Connector", "Document", "SourceRef", "FilesystemConnector",
    "GoogleDriveConnector", "SharePointConnector", "ConnectorNotConfigured",
    "ConnectorInfo", "connectors", "connector", "available_connectors",
    "REGISTRY", "get_connector",
]
