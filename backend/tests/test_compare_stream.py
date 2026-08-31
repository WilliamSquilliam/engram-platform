"""Streaming compare on the vLLM backend — specifically the one-retrieval-per-question
contract: the cart side retrieves; the rag side reuses the cart side's doc_ids (passed by
the UI) and must NOT hit retrieval again. The GPU inference service is faked at the
httpx.stream seam, so these run with no torch/network."""
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
    """Flip the request-time backend check to the production serve path. Must happen AFTER
    _ready_corpus: onboarding itself runs on the mocked hf path (mock_ml), not vllm."""
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
                "prompt_tokens": 10, "gen_tokens": 3}}))

    @contextmanager
    def _stream(method, url, **kw):
        captured["url"] = url
        captured["payload"] = kw.get("json")
        yield _Resp()

    monkeypatch.setattr(httpx, "stream", _stream)
    return captured


def test_context_for_known_and_unknown_ids(client, auth, make_corpus, upload_doc, mock_ml, cart_id):
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    a = cart_id(cid)  # the tenant-namespaced id for a.txt
    assert "alpha beta gamma" in retrieval.context_for(cid, [a])
    with pytest.raises(KeyError):
        retrieval.context_for(cid, [a, "nope"])


def test_stream_cart_side_retrieves_ids_only(client, auth, make_corpus, upload_doc,
                                             mock_ml, fake_engine, monkeypatch, cart_id):
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    a = cart_id(cid)  # tenant-namespaced id retrieve() now returns
    _to_vllm(monkeypatch)
    calls = {"retrieve": 0, "retrieve_context": 0}
    real_retrieve = retrieval.retrieve
    monkeypatch.setattr(retrieval, "retrieve",
                        lambda *a, **k: calls.__setitem__("retrieve", calls["retrieve"] + 1)
                        or real_retrieve(*a, **k))
    monkeypatch.setattr(retrieval, "retrieve_context",
                        lambda *a, **k: pytest.fail("cart side must not build RAG context"))

    r = client.post(f"/corpora/{cid}/compare/stream?side=cart",
                    json={"question": "what is alpha?"}, headers=headers)
    assert r.status_code == 200
    assert calls["retrieve"] == 1
    assert fake_engine["url"].endswith("/query_stream")
    assert fake_engine["payload"]["doc_ids"] == [a]
    head = json.loads(r.text.split("\n\n")[0].removeprefix("data: "))
    assert head["used_docs"] == [a] and head["sources"][0]["title"] == "Alpha Paper Title"


def test_stream_rag_with_doc_ids_skips_retrieval(client, auth, make_corpus, upload_doc,
                                                 mock_ml, fake_engine, monkeypatch, cart_id):
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    monkeypatch.setattr(retrieval, "retrieve",
                        lambda *a, **k: pytest.fail("rag side with doc_ids must not retrieve"))
    monkeypatch.setattr(retrieval, "retrieve_context",
                        lambda *a, **k: pytest.fail("rag side with doc_ids must not retrieve"))

    r = client.post(f"/corpora/{cid}/compare/stream?side=rag",
                    json={"question": "what is alpha?", "doc_ids": [cart_id(cid)]}, headers=headers)
    assert r.status_code == 200
    assert fake_engine["url"].endswith("/rag_query_stream")
    # context_for assembled the SAME evidence the cart side saw, from the passed ids
    assert "alpha beta gamma" in fake_engine["payload"]["context"]
    assert '"delta"' in r.text and '"summary"' in r.text  # engine deltas + cost proxied through


def test_stream_rag_standalone_still_retrieves(client, auth, make_corpus, upload_doc,
                                               mock_ml, fake_engine, monkeypatch):
    """No doc_ids passed (direct API use) -> the rag side retrieves for itself, as before."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    r = client.post(f"/corpora/{cid}/compare/stream?side=rag",
                    json={"question": "what is alpha?"}, headers=headers)
    assert r.status_code == 200
    assert "alpha beta gamma" in fake_engine["payload"]["context"]


def test_stream_rag_rejects_foreign_doc_ids(client, auth, make_corpus, upload_doc,
                                            mock_ml, fake_engine, monkeypatch):
    """Client-echoed ids that don't belong to the corpus -> 400, never a silent empty context."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    r = client.post(f"/corpora/{cid}/compare/stream?side=rag",
                    json={"question": "q?", "doc_ids": ["../../etc/passwd", "a"]}, headers=headers)
    assert r.status_code == 400
    assert "unknown doc_ids" in r.text


def _adaptive_engine(monkeypatch, cart_conf):
    """Fake the GPU SSE for the adaptive router: /query_stream returns the cart-alone answer with
    the given confidence; /rag_query_stream returns the backup. Records every proxied url so a test
    can assert whether the backup fired."""
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
                    "latency_ms": 10.0, "ttft_ms": 4.0, "decode_tps": 40.0,
                    "prompt_tokens": 8, "gen_tokens": 3, "confidence": cart_conf}}))
            else:  # /rag_query_stream backup
                yield 'data: {"delta": "backup-answer"}'
                yield ("data: " + json.dumps({"done": True, "metrics": {
                    "latency_ms": 30.0, "ttft_ms": 12.0, "decode_tps": 40.0,
                    "prompt_tokens": 200, "gen_tokens": 5, "confidence": None}}))

    @contextmanager
    def _stream(method, url, **kw):
        calls.append(url)
        yield _Resp(url)

    monkeypatch.setattr(httpx, "stream", _stream)
    return calls


def _last_frame(text: str) -> dict:
    return json.loads(text.strip().split("\n\n")[-1].removeprefix("data: "))


def test_adaptive_escalates_when_underconfident(client, auth, make_corpus, upload_doc,
                                                mock_ml, monkeypatch):
    """Cart-alone confidence below ADAPTIVE_THETA -> the router serves the RAG backup too, flags
    the escalation in-band, and the final tier is rag-backup."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    monkeypatch.setattr(config, "ADAPTIVE_THETA", "-0.5")
    calls = _adaptive_engine(monkeypatch, cart_conf=-1.2)      # below theta -> escalate

    r = client.post(f"/corpora/{cid}/compare/stream?side=cart",
                    json={"question": "what is alpha?"}, headers=headers)
    assert r.status_code == 200
    assert any(u.endswith("/query_stream") for u in calls)
    assert any(u.endswith("/rag_query_stream") for u in calls)   # the backup fired
    assert '"escalate"' in r.text and "backup-answer" in r.text  # flagged + backup streamed through
    summary = _last_frame(r.text)
    assert summary["tier"] == "rag-backup" and summary["escalated"] is True


def test_adaptive_stays_cartridge_when_confident(client, auth, make_corpus, upload_doc,
                                                 mock_ml, monkeypatch):
    """Cart-alone confidence at/above ADAPTIVE_THETA -> no backup, tier stays cartridge, 0 escalation."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    monkeypatch.setattr(config, "ADAPTIVE_THETA", "-0.5")
    calls = _adaptive_engine(monkeypatch, cart_conf=-0.1)       # above theta -> no escalate

    r = client.post(f"/corpora/{cid}/compare/stream?side=cart",
                    json={"question": "what is alpha?"}, headers=headers)
    assert r.status_code == 200
    assert not any(u.endswith("/rag_query_stream") for u in calls)   # backup NOT fired
    assert '"escalate"' not in r.text
    summary = _last_frame(r.text)
    assert summary["tier"] == "cartridge" and summary["escalated"] is False


def test_adaptive_disabled_never_escalates(client, auth, make_corpus, upload_doc,
                                           mock_ml, monkeypatch):
    """ADAPTIVE_THETA unset (default) -> pure single-cart CAG even at low confidence (opt-in feature)."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    monkeypatch.setattr(config, "ADAPTIVE_THETA", "")
    calls = _adaptive_engine(monkeypatch, cart_conf=-9.9)       # very low, but router is off

    r = client.post(f"/corpora/{cid}/compare/stream?side=cart",
                    json={"question": "what is alpha?"}, headers=headers)
    assert r.status_code == 200
    assert not any(u.endswith("/rag_query_stream") for u in calls)
    assert _last_frame(r.text)["tier"] == "cartridge"
