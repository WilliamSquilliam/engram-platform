"""Source connectors: the pluggable ingestion seam (see base.Connector).

`get_connector(source_type, **kwargs)` is the factory the ingest path uses so adding a
new source (SharePoint/Confluence/Drive/S3) is a registry entry, not a code change at
the call site.
"""
from __future__ import annotations

from .base import Connector, Document, SourceRef
from .filesystem import FilesystemConnector

#: source_type -> Connector class. New connectors register here.
REGISTRY: dict[str, type[Connector]] = {
    FilesystemConnector.source_type: FilesystemConnector,
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
    "REGISTRY", "get_connector",
]
