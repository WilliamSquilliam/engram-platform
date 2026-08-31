"""Streaming chat endpoint (/corpora/{id}/chat/stream) — the SSE the conversational chat UI (E3)
consumes. Mirrors the compare-stream tests: the GPU inference service is faked at the httpx.stream
seam so these run with no torch/network. Covers the head->deltas->done contract, tenant/auth
scoping, doc_id pinning on follow-up turns, and the adaptive escalation to the RAG backup."""
import json
from contextlib import contextmanager

import httpx
import pytest
from app import config, retrieval


def _ready_corpus(client, headers, make_corpus, upload_doc) -> str:
    cid = make_corpus(headers)
    upload_doc(headers, cid, "Alpha Paper Title\nalpha beta gamma")
    client.post(f"/corpora/{cid}/train", headers=headers)  # mock_ml -> ready
    return cid


def _to_vllm(monkeypatch):
    monkeypatch.setattr(config, "INFERENCE_BACKEND", "vllm")


@pytest.fixture()
def fake_engine(monkeypatch):
    """Fake the GPU service's SSE stream; captures the proxied URL + payload."""
    captured: dict = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield 'data: {"delta": "hi"}'
            yield ('data: ' + json.dumps({"done": True, "metrics": {
                "latency_ms": 12.0, "ttft_ms": 5.0, "decode_tps": 40.0,
                "prompt_tokens": 10, "gen_tokens": 3, "confidence": -0.1}}))

    @contextmanager
    def _stream(method, url, **kw):
        captured["url"] = url
        captured["payload"] = kw.get("json")
        yield _Resp()

    monkeypatch.setattr(httpx, "stream", _stream)
    return captured


def _last_frame(text: str) -> dict:
    return json.loads(text.strip().split("\n\n")[-1].removeprefix("data: "))


def test_stream_head_deltas_done(client, auth, make_corpus, upload_doc, mock_ml, fake_engine, monkeypatch):
    """First turn: retrieves doc_ids, hits /query_stream, and emits head -> delta -> done."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)

    r = client.post(f"/corpora/{cid}/chat/stream", json={"question": "what is alpha?"}, headers=headers)
    assert r.status_code == 200
    assert fake_engine["url"].endswith("/query_stream")
    assert fake_engine["payload"]["doc_ids"] == ["a"]
    head = json.loads(r.text.split("\n\n")[0].removeprefix("data: "))
    assert head["head"] is True
    assert head["used_docs"] == ["a"] and head["sources"][0]["title"] == "Alpha Paper Title"
    assert '"delta"' in r.text
    done = _last_frame(r.text)
    assert done["done"] is True and done["metrics"]["tier"] == "cartridge"


def test_stream_pins_doc_ids_skips_retrieval(client, auth, make_corpus, upload_doc,
                                             mock_ml, fake_engine, monkeypatch):
    """Follow-up turn echoes the first turn's doc_ids -> retrieval is skipped, ids reused verbatim."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    monkeypatch.setattr(retrieval, "retrieve",
                        lambda *a, **k: pytest.fail("pinned turn must not retrieve"))

    r = client.post(f"/corpora/{cid}/chat/stream",
                    json={"question": "follow up?", "doc_ids": ["a"]}, headers=headers)
    assert r.status_code == 200
    assert fake_engine["payload"]["doc_ids"] == ["a"]


def test_stream_rejects_foreign_doc_ids(client, auth, make_corpus, upload_doc,
                                        mock_ml, fake_engine, monkeypatch):
    """Client-echoed ids not in the corpus -> 400, never a silent wrong-evidence answer."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    r = client.post(f"/corpora/{cid}/chat/stream",
                    json={"question": "q?", "doc_ids": ["../../etc/passwd", "a"]}, headers=headers)
    assert r.status_code == 400


def test_stream_requires_auth(client, make_corpus, upload_doc, auth, mock_ml, monkeypatch):
    """No JWT -> 401 (tenant/auth scoping, same as /chat)."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    r = client.post(f"/corpora/{cid}/chat/stream", json={"question": "q?"})
    assert r.status_code == 401


def test_stream_other_tenant_404(client, auth, make_corpus, upload_doc, mock_ml, monkeypatch):
    """A corpus owned by another tenant is not found for this user."""
    import uuid
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    # Register a second, unrelated tenant and try to reach the first tenant's corpus.
    r2 = client.post("/auth/register", json={
        "email": f"other-{uuid.uuid4().hex[:8]}@test.local", "password": "pw123456",
        "tenant_name": "Other"})
    other = {"Authorization": f"Bearer {r2.json()['access_token']}"}
    r = client.post(f"/corpora/{cid}/chat/stream", json={"question": "q?"}, headers=other)
    assert r.status_code == 404


def _adaptive_engine(monkeypatch, cart_conf):
    """Fake the GPU SSE for the adaptive router: /query_stream returns the cart-alone answer with
    the given confidence; /rag_query_stream returns the backup. Records every proxied url."""
    calls: list[str] = []

    class _Resp:
        def __init__(self, url):
            self._url = url

        def raise_for_status(self):
            pass

        def iter_lines(self):
            if self._url.endswith("/query_stream"):
                yield 'data: {"delta": "cart-alone-answer"}'
                yield ("data: " + json.dumps({"done": True, "metrics": {
                    "latency_ms": 10.0, "prompt_tokens": 8, "gen_tokens": 3, "confidence": cart_conf}}))
            else:
                yield 'data: {"delta": "backup-answer"}'
                yield ("data: " + json.dumps({"done": True, "metrics": {
                    "latency_ms": 30.0, "prompt_tokens": 200, "gen_tokens": 5, "confidence": None}}))

    @contextmanager
    def _stream(method, url, **kw):
        calls.append(url)
        yield _Resp(url)

    monkeypatch.setattr(httpx, "stream", _stream)
    return calls


def test_stream_adaptive_escalates(client, auth, make_corpus, upload_doc, mock_ml, monkeypatch):
    """Under-confident cart answer -> serve the RAG backup, flag it in-band, final tier rag-backup."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    monkeypatch.setattr(config, "ADAPTIVE_THETA", "-0.5")
    calls = _adaptive_engine(monkeypatch, cart_conf=-1.2)

    r = client.post(f"/corpora/{cid}/chat/stream", json={"question": "what is alpha?"}, headers=headers)
    assert r.status_code == 200
    assert any(u.endswith("/rag_query_stream") for u in calls)   # backup fired
    assert '"escalate"' in r.text and "backup-answer" in r.text
    assert _last_frame(r.text)["metrics"]["tier"] == "rag-backup"


def test_stream_adaptive_disabled_stays_cartridge(client, auth, make_corpus, upload_doc,
                                                  mock_ml, monkeypatch):
    """ADAPTIVE_THETA unset -> pure single-cart CAG, no backup even at low confidence."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    monkeypatch.setattr(config, "ADAPTIVE_THETA", "")
    calls = _adaptive_engine(monkeypatch, cart_conf=-9.9)

    r = client.post(f"/corpora/{cid}/chat/stream", json={"question": "what is alpha?"}, headers=headers)
    assert r.status_code == 200
    assert not any(u.endswith("/rag_query_stream") for u in calls)
    assert _last_frame(r.text)["metrics"]["tier"] == "cartridge"
