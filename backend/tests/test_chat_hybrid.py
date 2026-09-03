"""QRC hybrid serving split on the /chat vLLM path. When QRC_MODE=hybrid and retrieval returns >1
doc, the control plane serves the TOP-1 doc as the resident cart and routes docs 2..k in as a small
`context` string — while still reporting ALL retrieved doc_ids as used_docs/sources (the user sees the
same evidence; only the serve mechanism changed). QRC_MODE=off is the legacy multi-cart serve.

The vLLM inference service AND the chunk router are faked (no torch/GPU/network); retrieval.retrieve is
stubbed to a fixed multi-doc result so the split is what's under test, not ranking."""
import json
from contextlib import contextmanager

import httpx
import pytest
from app import config, ml_client, retrieval


def _ready_corpus(client, headers, make_corpus, upload_doc) -> str:
    cid = make_corpus(headers)
    upload_doc(headers, cid, "Alpha Paper Title\nalpha beta gamma")
    client.post(f"/corpora/{cid}/train", headers=headers)  # mock_ml -> ready
    return cid


def _to_vllm(monkeypatch):
    monkeypatch.setattr(config, "INFERENCE_BACKEND", "vllm")


@pytest.fixture()
def capture_iq(monkeypatch):
    """Capture the exact args the serve path hands to the vLLM /query (doc_ids + context + resident
    span fields)."""
    seen: dict = {}

    def _iq(doc_ids, question, max_tokens=96, history=None, context="",
            doc_spans=None, doc_texts=None, doc_titles=None):
        seen["doc_ids"] = list(doc_ids)
        seen["context"] = context
        seen["doc_spans"] = doc_spans
        seen["doc_texts"] = doc_texts
        seen["doc_titles"] = doc_titles
        return {"answer": "served answer", "doc_ids": doc_ids,
                "metrics": {"latency_ms": 8.0, "prompt_tokens": 10, "gen_tokens": 3}}

    monkeypatch.setattr(ml_client, "inference_query", _iq)
    return seen


def test_hybrid_serves_top1_cart_plus_context_reports_all(client, auth, make_corpus, upload_doc,
                                                          mock_ml, capture_iq, monkeypatch):
    """QRC_MODE=hybrid + multi-doc retrieval -> serve doc_ids=[top1] with a non-empty routed context;
    used_docs/sources still list ALL retrieved ids."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    monkeypatch.setattr(config, "QRC_MODE", "hybrid")
    # Three retrieved docs; only the first is the resident cart.
    monkeypatch.setattr(retrieval, "retrieve", lambda *a, **k: ["top1", "doc2", "doc3"])
    # Fake the router: assert it's asked for docs 2..k, return a marker context.
    routed_calls = {}

    def _route(corpus_id, question, doc_ids):
        routed_calls["doc_ids"] = list(doc_ids)
        return "ROUTED-CONTEXT"

    monkeypatch.setattr(retrieval, "route_chunks_context", _route)
    monkeypatch.setattr(retrieval, "doc_sources",
                        lambda cid_, ids: [{"id": d, "title": d} for d in ids])

    r = client.post(f"/corpora/{cid}/chat", json={"question": "what is alpha?"}, headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    # Served: ONLY the top-1 cart, plus the routed context of docs 2..k.
    assert capture_iq["doc_ids"] == ["top1"]
    assert capture_iq["context"] == "ROUTED-CONTEXT"
    assert routed_calls["doc_ids"] == ["doc2", "doc3"]       # docs 2..k were routed
    # Reported: ALL retrieved ids (evidence the user sees is unchanged).
    assert body["used_docs"] == ["top1", "doc2", "doc3"]
    assert [s["id"] for s in body["sources"]] == ["top1", "doc2", "doc3"]


def test_off_serves_all_doc_ids_no_context(client, auth, make_corpus, upload_doc,
                                           mock_ml, capture_iq, monkeypatch):
    """QRC_MODE=off -> legacy multi-cart serve: every retrieved doc_id handed over, context empty, and
    the chunk router is never called."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    monkeypatch.setattr(config, "QRC_MODE", "off")
    monkeypatch.setattr(retrieval, "retrieve", lambda *a, **k: ["top1", "doc2", "doc3"])
    monkeypatch.setattr(retrieval, "route_chunks_context",
                        lambda *a, **k: pytest.fail("off mode must not route chunks"))
    monkeypatch.setattr(retrieval, "doc_sources",
                        lambda cid_, ids: [{"id": d, "title": d} for d in ids])

    r = client.post(f"/corpora/{cid}/chat", json={"question": "what is alpha?"}, headers=headers)
    assert r.status_code == 200, r.text
    assert capture_iq["doc_ids"] == ["top1", "doc2", "doc3"]   # all carts served
    assert capture_iq["context"] == ""                          # no routed context
    assert r.json()["used_docs"] == ["top1", "doc2", "doc3"]


def test_resident_serves_all_docs_with_spans_no_context(client, auth, make_corpus, upload_doc,
                                                        mock_ml, capture_iq, monkeypatch):
    """QRC_MODE=resident + multi-doc -> serve ALL doc_ids (top full-cart + docs 2..k as KV spans) with
    NO text context; the payload carries doc_spans/doc_texts/doc_titles for the spanned (non-top) docs,
    and used_docs/sources still list ALL retrieved ids."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    monkeypatch.setattr(config, "QRC_MODE", "resident")
    monkeypatch.setattr(retrieval, "retrieve", lambda *a, **k: ["top1", "doc2", "doc3"])
    routed = {}

    def _spans(corpus_id, question, doc_ids):
        routed["doc_ids"] = list(doc_ids)
        return {"doc2": [[0, 20]], "doc3": [[5, 30]]}   # both non-top docs resolved to spans

    monkeypatch.setattr(retrieval, "route_chunk_spans", _spans)
    monkeypatch.setattr(retrieval, "served_texts_for",
                        lambda cid_, ids: {d: f"text of {d}" for d in ids})
    monkeypatch.setattr(retrieval, "doc_titles_for",
                        lambda cid_, ids: {d: f"Title {d}" for d in ids})
    monkeypatch.setattr(retrieval, "route_chunks_context",
                        lambda *a, **k: pytest.fail("resident mode must not build text context"))
    monkeypatch.setattr(retrieval, "doc_sources",
                        lambda cid_, ids: [{"id": d, "title": d} for d in ids])

    r = client.post(f"/corpora/{cid}/chat", json={"question": "what is alpha?"}, headers=headers)
    assert r.status_code == 200, r.text
    # Served: ALL doc_ids, NO context, spans for the non-top docs only.
    assert capture_iq["doc_ids"] == ["top1", "doc2", "doc3"]
    assert capture_iq["context"] == ""
    assert routed["doc_ids"] == ["doc2", "doc3"]            # docs 2..k were span-routed
    assert capture_iq["doc_spans"] == {"doc2": [[0, 20]], "doc3": [[5, 30]]}
    assert capture_iq["doc_texts"] == {"doc2": "text of doc2", "doc3": "text of doc3"}
    assert capture_iq["doc_titles"] == {"doc2": "Title doc2", "doc3": "Title doc3"}
    # Reported: ALL retrieved ids.
    assert r.json()["used_docs"] == ["top1", "doc2", "doc3"]


def test_resident_drops_docs_with_no_resolved_spans(client, auth, make_corpus, upload_doc,
                                                    mock_ml, capture_iq, monkeypatch):
    """A non-top doc whose spans all resolve empty is absent from doc_spans -> the control plane sends
    text/titles only for the docs that DID resolve (never re-tokenize or attribute a dropped doc)."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    monkeypatch.setattr(config, "QRC_MODE", "resident")
    monkeypatch.setattr(retrieval, "retrieve", lambda *a, **k: ["top1", "doc2", "doc3"])
    # Only doc2 resolved to spans; doc3 dropped out of routing.
    monkeypatch.setattr(retrieval, "route_chunk_spans", lambda *a, **k: {"doc2": [[0, 12]]})
    monkeypatch.setattr(retrieval, "served_texts_for",
                        lambda cid_, ids: {d: f"text of {d}" for d in ids})
    monkeypatch.setattr(retrieval, "doc_titles_for",
                        lambda cid_, ids: {d: f"Title {d}" for d in ids})
    monkeypatch.setattr(retrieval, "doc_sources",
                        lambda cid_, ids: [{"id": d, "title": d} for d in ids])

    r = client.post(f"/corpora/{cid}/chat", json={"question": "what is alpha?"}, headers=headers)
    assert r.status_code == 200, r.text
    assert capture_iq["doc_ids"] == ["top1", "doc2", "doc3"]   # all still served/reported
    assert capture_iq["doc_spans"] == {"doc2": [[0, 12]]}
    # texts/titles only for the doc that resolved — doc3 is neither re-tokenized nor attributed.
    assert capture_iq["doc_texts"] == {"doc2": "text of doc2"}
    assert capture_iq["doc_titles"] == {"doc2": "Title doc2"}


def test_hybrid_single_doc_serves_as_cart_no_context(client, auth, make_corpus, upload_doc,
                                                     mock_ml, capture_iq, monkeypatch):
    """Hybrid but only ONE retrieved doc -> nothing to route: serve it as a lone cart, empty context
    (the split only kicks in with >1 doc)."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    monkeypatch.setattr(config, "QRC_MODE", "hybrid")
    monkeypatch.setattr(retrieval, "retrieve", lambda *a, **k: ["only1"])
    monkeypatch.setattr(retrieval, "route_chunks_context",
                        lambda *a, **k: pytest.fail("single-doc hybrid must not route chunks"))
    monkeypatch.setattr(retrieval, "doc_sources",
                        lambda cid_, ids: [{"id": d, "title": d} for d in ids])

    r = client.post(f"/corpora/{cid}/chat", json={"question": "what is alpha?"}, headers=headers)
    assert r.status_code == 200, r.text
    assert capture_iq["doc_ids"] == ["only1"]
    assert capture_iq["context"] == ""
    assert r.json()["used_docs"] == ["only1"]


# --- streaming path carries the same split -------------------------------------------------------

@pytest.fixture()
def capture_stream(monkeypatch):
    """Fake the GPU SSE stream and capture the proxied /query_stream payload (doc_ids + context)."""
    captured: dict = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield 'data: {"delta": "hi"}'
            yield ('data: ' + json.dumps({"done": True, "metrics": {
                "latency_ms": 12.0, "prompt_tokens": 10, "gen_tokens": 3, "confidence": -0.1}}))

    @contextmanager
    def _stream(method, url, **kw):
        captured["url"] = url
        captured["payload"] = kw.get("json")
        yield _Resp()

    monkeypatch.setattr(httpx, "stream", _stream)
    return captured


def test_stream_hybrid_serves_top1_plus_context_head_reports_all(client, auth, make_corpus, upload_doc,
                                                                mock_ml, capture_stream, monkeypatch):
    """chat/stream on hybrid: the /query_stream payload carries doc_ids=[top1] + routed context, while
    the head frame + sources still list ALL retrieved ids."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    monkeypatch.setattr(config, "QRC_MODE", "hybrid")
    monkeypatch.setattr(retrieval, "retrieve", lambda *a, **k: ["top1", "doc2", "doc3"])
    monkeypatch.setattr(retrieval, "route_chunks_context", lambda *a, **k: "ROUTED-CONTEXT")
    monkeypatch.setattr(retrieval, "doc_sources",
                        lambda cid_, ids: [{"id": d, "title": d} for d in ids])

    r = client.post(f"/corpora/{cid}/chat/stream", json={"question": "what is alpha?"}, headers=headers)
    assert r.status_code == 200
    assert capture_stream["url"].endswith("/query_stream")
    assert capture_stream["payload"]["doc_ids"] == ["top1"]        # only the resident cart is served
    assert capture_stream["payload"]["context"] == "ROUTED-CONTEXT"
    head = json.loads(r.text.split("\n\n")[0].removeprefix("data: "))
    assert head["used_docs"] == ["top1", "doc2", "doc3"]           # head reports ALL retrieved ids


def test_stream_off_serves_all_no_context(client, auth, make_corpus, upload_doc,
                                          mock_ml, capture_stream, monkeypatch):
    """chat/stream on QRC_MODE=off: every retrieved id in the payload, empty context."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    monkeypatch.setattr(config, "QRC_MODE", "off")
    monkeypatch.setattr(retrieval, "retrieve", lambda *a, **k: ["top1", "doc2", "doc3"])
    monkeypatch.setattr(retrieval, "route_chunks_context",
                        lambda *a, **k: pytest.fail("off mode must not route chunks"))
    monkeypatch.setattr(retrieval, "doc_sources",
                        lambda cid_, ids: [{"id": d, "title": d} for d in ids])

    r = client.post(f"/corpora/{cid}/chat/stream", json={"question": "what is alpha?"}, headers=headers)
    assert r.status_code == 200
    assert capture_stream["payload"]["doc_ids"] == ["top1", "doc2", "doc3"]
    assert capture_stream["payload"]["context"] == ""


def test_stream_resident_serves_spans_no_context(client, auth, make_corpus, upload_doc,
                                                 mock_ml, capture_stream, monkeypatch):
    """chat/stream on QRC_MODE=resident: the /query_stream payload carries ALL doc_ids + doc_spans/
    doc_texts/doc_titles for the spanned docs and an empty context; head/sources report ALL ids."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    monkeypatch.setattr(config, "QRC_MODE", "resident")
    monkeypatch.setattr(retrieval, "retrieve", lambda *a, **k: ["top1", "doc2", "doc3"])
    monkeypatch.setattr(retrieval, "route_chunk_spans",
                        lambda *a, **k: {"doc2": [[0, 20]], "doc3": [[5, 30]]})
    monkeypatch.setattr(retrieval, "served_texts_for",
                        lambda cid_, ids: {d: f"text of {d}" for d in ids})
    monkeypatch.setattr(retrieval, "doc_titles_for",
                        lambda cid_, ids: {d: f"Title {d}" for d in ids})
    monkeypatch.setattr(retrieval, "doc_sources",
                        lambda cid_, ids: [{"id": d, "title": d} for d in ids])

    r = client.post(f"/corpora/{cid}/chat/stream", json={"question": "what is alpha?"}, headers=headers)
    assert r.status_code == 200
    p = capture_stream["payload"]
    assert p["doc_ids"] == ["top1", "doc2", "doc3"]
    assert p["context"] == ""
    assert p["doc_spans"] == {"doc2": [[0, 20]], "doc3": [[5, 30]]}
    assert p["doc_texts"] == {"doc2": "text of doc2", "doc3": "text of doc3"}
    assert p["doc_titles"] == {"doc2": "Title doc2", "doc3": "Title doc3"}
    head = json.loads(r.text.split("\n\n")[0].removeprefix("data: "))
    assert head["used_docs"] == ["top1", "doc2", "doc3"]
