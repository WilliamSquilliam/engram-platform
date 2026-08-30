"""Source-connector interface — the seam for ingesting a corpus from ANY backend.

The platform must onboard corpora from many sources (plain file upload today;
SharePoint, Confluence, Google Drive, a tenant S3 bucket, generic DB tomorrow) without
the training pipeline caring where the bytes came from. Every source implements the
same three-method contract and the downstream stages (parse -> shard -> self-study ->
train) consume a uniform `Document` stream.

This is the abstraction docs/PRE_PRODUCTION.md calls for ("Ingest by *connecting a source*,
not uploading"):
  - discover()  -> cheap listing of what exists (metadata only)
  - fetch(refs) -> stream full, parsed Documents for those refs (the bulk-bytes path)
  - delta(cur)  -> only what changed since a saved cursor (incremental sync)

Pure stdlib so it can be imported and unit-tested without the FastAPI app or any
cloud SDK. Concrete connectors (FilesystemConnector here; SharePoint/Confluence/Drive/
S3 later) live alongside this module and depend on their own SDKs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field


@dataclass
class SourceRef:
    """A lightweight pointer returned by discover(); fetch() turns it into a Document."""
    doc_id: str             # stable, filesystem/S3-safe id (slug of path/title)
    source_uri: str         # where it lives (path, Graph item id, Drive file id, ...)
    title: str = ""         # human-readable title
    version: str = ""       # etag / changeToken / mtime — drives incremental delta()


@dataclass
class Document:
    """A parsed document ready for the pipeline (parse -> shard -> self-study -> train)."""
    doc_id: str
    text: str
    title: str = ""
    source_uri: str = ""
    metadata: dict = field(default_factory=dict)


class Connector(ABC):
    """Read-only ingestion contract shared by every source.

    Implementations must be safe to call repeatedly (idempotent) and must stream rather
    than materialize everything in memory — a tenant corpus can be 500k+ documents.
    """

    #: short identifier persisted on Corpus.source_type (e.g. "filesystem", "sharepoint")
    source_type: str = "abstract"

    @abstractmethod
    def discover(self) -> Iterable[SourceRef]:
        """List every available document as cheap metadata refs (no bodies)."""

    @abstractmethod
    def fetch(self, refs: Iterable[SourceRef]) -> Iterator[Document]:
        """Stream full, parsed Documents for the given refs."""

    def delta(self, cursor: str | None) -> tuple[Iterable[SourceRef], str]:
        """Refs changed since `cursor`; returns (changed_refs, new_cursor).

        Default implementation has no incremental support: it re-discovers everything
        and returns an empty cursor. Connectors backed by a changeToken/etag override
        this so a sync pulls only changes (the PRE_PRODUCTION incremental-sync goal).
        """
        return list(self.discover()), ""
