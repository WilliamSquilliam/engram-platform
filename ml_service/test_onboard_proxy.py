"""app.py /onboard_cag PROXY path (ONBOARD_VIA_ENGINE) — the control-plane-facing seam.

app.py imports torch + the engram-cartridge wheel at module scope (it IS the transformers
onboarding worker), so this runs where those are installed (dev/GPU box), not the torch-free
backend suite — mirrors the other ml_service tests. What it pins:
  * flag OFF (default) -> today's transformers path (onboard_cag_corpus) runs, no forward to :8002,
  * flag ON -> the VERBATIM request body + bearer auth are forwarded to
    INFERENCE_SERVICE_URL/onboard_cag and the engine's response/status are relayed unchanged,
  * corpus_dir confinement (_safe_corpus_dir) is enforced on the proxy path too (rejected BEFORE
    any forward).

The engine endpoint is mocked by monkeypatching httpx.AsyncClient (no network, no engine).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("cartridges")   # app.py imports the wheel at module scope

sys.path.insert(0, str(Path(__file__).resolve().parent))

import app as app_mod  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


class _FakeResp:
    def __init__(self, status_code, content, content_type="application/json"):
        self.status_code = status_code
        self.content = content
        self.headers = {"content-type": content_type}


class _FakeAsyncClient:
    """Records the single forwarded POST and returns a canned response. Async-context-managed like
    the real httpx.AsyncClient so `async with httpx.AsyncClient(...)` works unchanged."""
    captured: dict = {}

    def __init__(self, *a, **k):
        self.init_kwargs = k

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, content=None, headers=None):
        _FakeAsyncClient.captured = {"url": url, "content": content, "headers": headers or {}}
        return _FakeResp(200, b'{"n_cartridges": 3, "method": "cag_engine"}')


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    _FakeAsyncClient.captured = {}
    # A corpus root the request's corpus_dir sits under, so _safe_corpus_dir accepts it.
    monkeypatch.setenv("ML_ALLOWED_CORPUS_ROOTS", str(Path.cwd()))
    monkeypatch.setenv("ML_AUTH_TOKEN", "")
    yield


def _client():
    return TestClient(app_mod.app)


def test_flag_off_uses_transformers_path(monkeypatch):
    """Default (flag off) never forwards; it calls the in-process onboard_cag_corpus."""
    monkeypatch.setattr(app_mod, "ONBOARD_VIA_ENGINE", False)
    called = {}

    def _fake_onboard(corpus_dir, docs, report=None, build_index=False, should_cancel=None):
        called["corpus_dir"] = corpus_dir
        return {"n_cartridges": len(docs), "method": "cag"}

    monkeypatch.setattr(app_mod, "onboard_cag_corpus", _fake_onboard)
    monkeypatch.setattr(app_mod.httpx, "AsyncClient", _FakeAsyncClient)

    corpus = str(Path.cwd() / "corpusX")
    r = _client().post("/onboard_cag", json={"corpus_dir": corpus,
                                             "docs": [{"doc_id": "d1", "text": "hello"}]})
    assert r.status_code == 200
    assert r.json()["method"] == "cag"          # transformers path result relayed
    assert called["corpus_dir"] == corpus
    assert _FakeAsyncClient.captured == {}       # nothing forwarded


def test_flag_on_forwards_verbatim_and_relays(monkeypatch):
    """Flag on: the raw body + bearer auth are forwarded to :8002 and the response is relayed."""
    monkeypatch.setattr(app_mod, "ONBOARD_VIA_ENGINE", True)
    monkeypatch.setattr(app_mod, "INFERENCE_SERVICE_URL", "http://127.0.0.1:8002")
    monkeypatch.setattr(app_mod.httpx, "AsyncClient", _FakeAsyncClient)
    # If the transformers path were hit it would blow up — proves the proxy short-circuits it.
    monkeypatch.setattr(app_mod, "onboard_cag_corpus",
                        lambda *a, **k: pytest.fail("transformers path ran under the proxy flag"))

    corpus = str(Path.cwd() / "corpusY")
    body = {"corpus_dir": corpus, "docs": [{"doc_id": "d1", "text": "hello"}],
            "progress_url": "http://cp/internal/jobs/J/progress", "progress_token": "tok",
            "build_index": True}
    r = _client().post("/onboard_cag", json=body,
                       headers={"Authorization": "Bearer secret-xyz"})

    assert r.status_code == 200
    assert r.json() == {"n_cartridges": 3, "method": "cag_engine"}   # engine response relayed verbatim
    cap = _FakeAsyncClient.captured
    assert cap["url"] == "http://127.0.0.1:8002/onboard_cag"
    # Body forwarded VERBATIM (byte-for-byte the request the control plane sent).
    import json
    assert json.loads(cap["content"]) == body
    assert cap["headers"].get("Authorization") == "Bearer secret-xyz"


def test_proxy_rejects_corpus_dir_outside_roots(monkeypatch):
    """corpus_dir confinement fires on the proxy path BEFORE any forward."""
    monkeypatch.setattr(app_mod, "ONBOARD_VIA_ENGINE", True)
    monkeypatch.setattr(app_mod.httpx, "AsyncClient", _FakeAsyncClient)

    r = _client().post("/onboard_cag",
                       json={"corpus_dir": "/etc/evil", "docs": [{"doc_id": "d", "text": "x"}]},
                       headers={"Authorization": "Bearer secret-xyz"})
    assert r.status_code == 400
    assert _FakeAsyncClient.captured == {}        # nothing forwarded — failed fast


def test_proxy_empty_docs_400_before_forward(monkeypatch):
    monkeypatch.setattr(app_mod, "ONBOARD_VIA_ENGINE", True)
    monkeypatch.setattr(app_mod.httpx, "AsyncClient", _FakeAsyncClient)
    corpus = str(Path.cwd() / "corpusZ")
    r = _client().post("/onboard_cag", json={"corpus_dir": corpus, "docs": []})
    assert r.status_code == 400
    assert _FakeAsyncClient.captured == {}
