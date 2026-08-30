"""FilesystemConnector + the connector factory — the ingestion seam (no cloud SDKs)."""
import pytest
from app.connectors import FilesystemConnector, get_connector


def _seed(root):
    (root / "a.txt").write_text("alpha doc", encoding="utf-8")
    sub = root / "sub"
    sub.mkdir()
    (sub / "b.md").write_text("beta doc", encoding="utf-8")
    (root / "ignore.bin").write_text("binary-ish", encoding="utf-8")  # wrong suffix


def test_discover_lists_only_text_files_with_stable_ids(tmp_path):
    _seed(tmp_path)
    c = FilesystemConnector(tmp_path)
    refs = list(c.discover())
    titles = sorted(r.title for r in refs)
    assert titles == ["a.txt", "sub/b.md"]            # .bin excluded, posix relpaths
    assert all(r.doc_id and r.version for r in refs)
    assert [r.doc_id for r in c.discover()] == [r.doc_id for r in refs]  # deterministic


def test_fetch_returns_parsed_documents(tmp_path):
    _seed(tmp_path)
    c = FilesystemConnector(tmp_path)
    docs = {d.title: d for d in c.fetch(c.discover())}
    assert docs["a.txt"].text == "alpha doc"
    assert docs["sub/b.md"].text == "beta doc"
    assert docs["a.txt"].doc_id == next(r.doc_id for r in c.discover() if r.title == "a.txt")


def test_delta_uses_mtime_watermark(tmp_path):
    _seed(tmp_path)
    c = FilesystemConnector(tmp_path)
    changed, cursor = c.delta(None)
    assert len(list(changed)) == 2 and cursor
    # Nothing changed since the cursor -> empty delta.
    changed2, cursor2 = c.delta(cursor)
    assert list(changed2) == [] and cursor2 == cursor


def test_factory_resolves_and_rejects(tmp_path):
    assert isinstance(get_connector("filesystem", root=tmp_path), FilesystemConnector)
    with pytest.raises(ValueError):
        get_connector("does-not-exist")
