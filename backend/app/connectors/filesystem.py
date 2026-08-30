"""FilesystemConnector — the reference Connector implementation.

Walks a directory tree of text-like files (.txt/.md/.json text) and yields them as
Documents. It is the simplest concrete proof of the Connector seam (used by tests and
by the plain "upload a folder" ingest path), and the template every richer connector
(SharePoint/Confluence/Drive/S3) follows: discover() lists, fetch() reads + parses,
delta() uses mtime as the change cursor.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from .base import Connector, Document, SourceRef

_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".text"}


def _slug(rel_path: str) -> str:
    """Stable, filesystem/S3-safe doc_id from a relative path (same id every run)."""
    base = re.sub(r"[^a-z0-9]+", "_", rel_path.lower()).strip("_")[:64]
    h = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:8]
    return f"{base}_{h}" if base else h


class FilesystemConnector(Connector):
    source_type = "filesystem"

    def __init__(self, root: str | Path, suffixes: Iterable[str] | None = None):
        self.root = Path(root)
        self.suffixes = {s.lower() for s in (suffixes or _TEXT_SUFFIXES)}

    def _files(self) -> list[Path]:
        return sorted(
            p for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in self.suffixes
        )

    def discover(self) -> Iterable[SourceRef]:
        for p in self._files():
            rel = p.relative_to(self.root).as_posix()
            yield SourceRef(
                doc_id=_slug(rel), source_uri=str(p), title=rel,
                version=str(p.stat().st_mtime_ns),
            )

    def fetch(self, refs: Iterable[SourceRef]) -> Iterator[Document]:
        for ref in refs:
            text = Path(ref.source_uri).read_text(encoding="utf-8", errors="replace")
            yield Document(
                doc_id=ref.doc_id, text=text, title=ref.title, source_uri=ref.source_uri,
            )

    def delta(self, cursor: str | None) -> tuple[Iterable[SourceRef], str]:
        """Return refs modified after `cursor` (a max mtime-ns watermark)."""
        watermark = int(cursor) if cursor else -1
        refs = [r for r in self.discover() if int(r.version) > watermark]
        new_cursor = str(max((int(r.version) for r in self.discover()), default=watermark))
        return refs, new_cursor
