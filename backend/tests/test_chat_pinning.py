"""Session-pinned doc_ids on the vLLM chat path: a follow-up turn echoes the doc_ids the first
turn resolved, so the control-plane retrieval (221.7ms/query GPU) is skipped and the pinned carts
are reused verbatim. Ownership of the client-echoed ids is still validated (unknown ids -> 400).
The vLLM inference service is faked at the ml_client seam (no torch/CUDA in tests)."""
import pytest
from app import config, ml_client, retrieval


def _ready_corpus(client, headers, make_corpus, upload_doc) -> str:
    cid = make_corpus(headers)
    upload_doc(headers, cid, "Alpha Paper Title\nalpha beta gamma")
    client.post(f"/corpora/{cid}/train", headers=headers)  # mock_ml -> ready
    return cid


def _to_vllm(monkeypatch):
    """Flip the request-time backend to the production serve path. Must run AFTER _ready_corpus:
    onboarding itself runs on the mocked hf path (mock_ml), not vllm.

    Also pins CHAT_PIN_REFRESH=off so these tests exercise the trust-the-pin contract they were
    written for (reuse the pinned ids verbatim, skip the refresh retrieval). The topic-shift REFRESH
    default (on) is covered separately in test_chat_pin_refresh.py."""
    monkeypatch.setattr(config, "INFERENCE_BACKEND", "vllm")
    monkeypatch.setattr(config, "CHAT_PIN_REFRESH", "off")


@pytest.fixture()
def fake_inference(monkeypatch):
    """Stand-in for the vLLM Inference Service /query. Deterministic answer + minimal metrics.
    Accepts the QRC `context` kwarg plus the resident span kwargs (doc_spans/doc_texts/doc_titles,
    all None on this single-doc pinning path) via **_ so the stub matches ml_client.inference_query's
    signature regardless of serve mode."""
    def _iq(doc_ids, question, max_tokens=96, history=None, context="", **_):
        return {"answer": "pinned answer", "doc_ids": doc_ids,
                "metrics": {"latency_ms": 8.0, "prompt_tokens": 10, "gen_tokens": 3}}

    monkeypatch.setattr(ml_client, "inference_query", _iq)


def test_chat_pinned_doc_ids_skips_retrieve(client, auth, make_corpus, upload_doc,
                                            mock_ml, fake_inference, monkeypatch, cart_id):
    """Pinned doc_ids -> retrieval.retrieve is NOT called; the answer still comes back with used_docs
    echoing the pinned ids so the next turn can re-pin."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    a = cart_id(cid)  # the tenant-namespaced id a first turn would have resolved
    _to_vllm(monkeypatch)
    monkeypatch.setattr(retrieval, "retrieve",
                        lambda *a, **k: pytest.fail("pinned chat must not re-retrieve"))

    r = client.post(f"/corpora/{cid}/chat",
                    json={"question": "follow-up?", "doc_ids": [a]}, headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == "pinned answer"
    assert body["used_docs"] == [a]                # client echoes these back on the next turn
    assert body["sources"][0]["title"] == "Alpha Paper Title"


def test_chat_without_doc_ids_still_retrieves(client, auth, make_corpus, upload_doc,
                                              mock_ml, fake_inference, monkeypatch, cart_id):
    """No doc_ids (first turn) -> retrieval runs as before; the resolved ids come back as used_docs
    (now the tenant-namespaced id, which the next turn re-pins)."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    calls = {"n": 0}
    real = retrieval.retrieve
    monkeypatch.setattr(retrieval, "retrieve",
                        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or real(*a, **k))

    r = client.post(f"/corpora/{cid}/chat", json={"question": "what is alpha?"}, headers=headers)
    assert r.status_code == 200, r.text
    assert calls["n"] == 1
    assert r.json()["used_docs"] == [cart_id(cid)]


def test_chat_pinned_unknown_ids_400(client, auth, make_corpus, upload_doc,
                                     mock_ml, fake_inference, monkeypatch):
    """Client-echoed ids that don't belong to the corpus (foreign corpus / path-y garbage) -> 400,
    never a silent serve of the wrong carts. Validation mirrors the compare paths."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    monkeypatch.setattr(ml_client, "inference_query",
                        lambda *a, **k: pytest.fail("must reject before hitting the engine"))

    r = client.post(f"/corpora/{cid}/chat",
                    json={"question": "q?", "doc_ids": ["../../etc/passwd", "a"]}, headers=headers)
    assert r.status_code == 400
    assert "unknown doc_ids" in r.text


def test_mcp_query_pins_too(client, auth, make_corpus, upload_doc,
                            mock_ml, fake_inference, monkeypatch, cart_id):
    """/mcp/{id}/query flows through the same _answer -> pinning works for the MCP path as well."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    a = cart_id(cid)
    # mcp token lives on the corpus row; fetch it via the JSON API (created at corpus create).
    token = client.get(f"/corpora/{cid}", headers=headers).json()["mcp_token"]
    _to_vllm(monkeypatch)
    monkeypatch.setattr(retrieval, "retrieve",
                        lambda *a, **k: pytest.fail("pinned mcp query must not re-retrieve"))

    r = client.post(f"/mcp/{cid}/query",
                    json={"question": "follow-up?", "doc_ids": [a]},
                    headers={"X-MCP-Token": token})
    assert r.status_code == 200, r.text
    assert r.json()["used_docs"] == [a]


def test_compare_nonstream_pinned_doc_ids_no_reretrieve(client, auth, make_corpus, upload_doc,
                                                        mock_ml, monkeypatch, cart_id):
    """Non-stream /compare on the vLLM path with pinned doc_ids -> retrieve_context is NOT called;
    the pinned ids are validated + reused, and both cart and rag sides see the SAME evidence."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    monkeypatch.setattr(retrieval, "retrieve_context",
                        lambda *a, **k: pytest.fail("pinned compare must not re-retrieve"))
    seen = {}

    def _iq(doc_ids, question, max_tokens=96, history=None, context="", **_):
        return {"answer": "cart", "doc_ids": doc_ids,
                "metrics": {"latency_ms": 8.0, "ttft_ms": 3.0, "decode_tps": 40.0,
                            "prompt_tokens": 10, "gen_tokens": 3, "resident_kv_tokens": 5}}

    def _rag(context, question, max_tokens=96, history=None):
        seen["context"] = context
        return {"answer": "rag", "metrics": {"latency_ms": 20.0, "ttft_ms": 8.0,
                "decode_tps": 40.0, "prompt_tokens": 200, "gen_tokens": 3}}

    monkeypatch.setattr(ml_client, "inference_query", _iq)
    monkeypatch.setattr(ml_client, "inference_rag", _rag)

    r = client.post(f"/corpora/{cid}/compare",
                    json={"question": "what is alpha?", "doc_ids": [cart_id(cid)]}, headers=headers)
    assert r.status_code == 200, r.text
    keys = [s["key"] for s in r.json()["strategies"]]
    assert keys == ["everyday", "rag"]
    # context_for rebuilt the SAME evidence the cart side saw, from the pinned ids
    assert "alpha beta gamma" in seen["context"]


def test_compare_nonstream_pinned_unknown_ids_400(client, auth, make_corpus, upload_doc,
                                                  mock_ml, monkeypatch):
    """Pinned ids not in this corpus -> 400 before any generation (same guard as the stream path)."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    monkeypatch.setattr(ml_client, "inference_query",
                        lambda *a, **k: pytest.fail("must reject before hitting the engine"))

    r = client.post(f"/corpora/{cid}/compare",
                    json={"question": "q?", "doc_ids": ["nope"]}, headers=headers)
    assert r.status_code == 400
    assert "unknown doc_ids" in r.text
