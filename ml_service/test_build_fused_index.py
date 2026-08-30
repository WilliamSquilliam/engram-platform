"""Unit tests for build_fused_index GPU-memory hygiene on failure — CPU-only, encoders
injected (no sentence-transformers / GPU needed). The load-bearing test proves a FAILED build
releases the partial index even while a caller keeps the exception alive (the job runner stores
it as the failure reason), so a retry does not start with leaked GPU memory — the compounding
leak that made three attempts accumulate ~10.5 GB during the 2026-07-22 deploy. Run from the
repo root in a torch env:

    python -m pytest platform/ml_service/test_build_fused_index.py -q
"""
import gc
import sys
import weakref
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import app as mlapp  # noqa: E402
import retrieval_fused  # noqa: E402
from retrieval_fused import _toks  # noqa: E402


def _fake_embed(texts):
    """Deterministic bag-of-words vectors — same fakes style as test_retrieval_fused."""
    out = torch.zeros(len(texts), 16)
    for i, t in enumerate(texts):
        for w in _toks(t):
            out[i, hash(w) % 16] += 1.0
    return torch.nn.functional.normalize(out, dim=-1)


def _fake_rerank(pairs):
    return [sum(1.0 for w in _toks(q) if w in _toks(d)) for q, d in pairs]


DOCS = [{"doc_id": "a", "text": "alpha beta gamma reactor plasma " * 6},
        {"doc_id": "b", "text": "delta epsilon zeta ocean hydrophone " * 6}]


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    # Never touch real encoders or S3; keep the module's fused-index cache clean per test.
    monkeypatch.setattr(mlapp, "_fused_encoders", lambda: (_fake_embed, _fake_rerank, ""))
    monkeypatch.setenv("CARTRIDGE_STORE_BACKEND", "local")  # _index_sync_up becomes a no-op
    mlapp._FUSED_CACHE.clear()
    yield
    mlapp._FUSED_CACHE.clear()


def test_success_builds_caches_and_persists(tmp_path):
    """Happy path unchanged: index built, cached, written to disk, and usable."""
    mlapp.build_fused_index(str(tmp_path), DOCS)
    assert (tmp_path / "fused_index.pt").exists()
    assert str(tmp_path) in mlapp._FUSED_CACHE
    assert mlapp._FUSED_CACHE[str(tmp_path)].retrieve("reactor plasma", k=1) == ["a"]


def test_failure_releases_partial_index_despite_held_traceback(tmp_path, monkeypatch):
    """The fix: an OOM (here simulated) during embed() must release the partial FusedIndex even
    though a caller keeps the exception + traceback — otherwise the traceback pins the dense
    matrix on the GPU and the next build OOMs with less headroom. A live weakref after the failed
    build == the leak; a dead one == released."""
    created = []

    class BoomIndex(retrieval_fused.FusedIndex):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(weakref.ref(self))

        def embed(self):
            raise RuntimeError("simulated CUDA OOM")

    monkeypatch.setattr(retrieval_fused, "FusedIndex", BoomIndex)

    held_tb = None
    try:
        mlapp.build_fused_index(str(tmp_path), DOCS)
    except RuntimeError as e:
        held_tb = e.__traceback__      # a caller that stores the failure reason + its traceback
    assert held_tb is not None, "build_fused_index should have raised the embed error"

    gc.collect()
    assert created, "BoomIndex was never constructed — test wired wrong"
    assert created[0]() is None, "partial FusedIndex still pinned after failed build (GPU leak)"
    # A failed build must not leave a half-built index cached or a stray file on disk.
    assert str(tmp_path) not in mlapp._FUSED_CACHE
    assert not (tmp_path / "fused_index.pt").exists()


def test_failure_reraises_original_error_unchanged(tmp_path, monkeypatch):
    """Cleanup must not swallow or replace the real error — the job runner needs the true cause."""
    class BoomIndex(retrieval_fused.FusedIndex):
        def embed(self):
            raise RuntimeError("simulated CUDA OOM: tried to allocate 236 MiB")

    monkeypatch.setattr(retrieval_fused, "FusedIndex", BoomIndex)
    with pytest.raises(RuntimeError, match="tried to allocate 236 MiB"):
        mlapp.build_fused_index(str(tmp_path), DOCS)


def test_durable_sync_failure_neither_leaks_nor_publishes_nondurable_index(tmp_path, monkeypatch):
    """The likeliest NON-OOM failure: the index builds fine but the S3 durable sync throws
    (throttle / network / creds). save() copies the matrix to CPU but leaves idx.mat on the GPU,
    so if the built index were published to _FUSED_CACHE before the sync, the module-level ref
    would keep that matrix resident even after the except nulls the local ref — a leak the earlier
    OOM tests can't catch (they fail before caching). This asserts the sync runs before publish:
    on sync failure the index is released AND never reaches the serve cache."""
    created = []

    class TrackedIndex(retrieval_fused.FusedIndex):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(weakref.ref(self))   # embed() runs normally; only the sync fails

    monkeypatch.setattr(retrieval_fused, "FusedIndex", TrackedIndex)

    def _boom_sync(_corpus_dir):
        raise RuntimeError("simulated S3 upload failure")

    monkeypatch.setattr(mlapp, "_index_sync_up", _boom_sync)

    with pytest.raises(RuntimeError, match="S3 upload failure"):
        mlapp.build_fused_index(str(tmp_path), DOCS)

    gc.collect()
    assert created, "index was never built"
    assert created[0]() is None, "built index still pinned after sync failure (GPU leak via _FUSED_CACHE)"
    assert str(tmp_path) not in mlapp._FUSED_CACHE, "a non-durable index was published to the serve cache"
