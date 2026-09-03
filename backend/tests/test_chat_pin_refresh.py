"""Topic-shift pin refresh (UPGRADE 2) + condensation wiring (UPGRADE 1) on the vLLM chat path.

Pin refresh: a follow-up echoes pinned doc_ids; the server ALSO runs a fresh retrieval on the
(condensed) question. If the best fresh doc isn't in the pinned set at all, the topic moved and the
fresh docs are served; otherwise the pinned list is kept verbatim (rank jitter within the set never
unpins). CHAT_PIN_REFRESH=off = trust-the-pin exactly (no refresh retrieval).

Condensation wiring: the rewritten (effective) query is what retrieval + _hybrid_split receive, while
the question SERVED to the model stays the original. The engine + retrieval are faked (no GPU/net)."""
import json
from contextlib import contextmanager

import httpx
import pytest
from app import condense, config, ml_client, retrieval
from app.routers import chat as chat_router


def _ready_corpus(client, headers, make_corpus, upload_doc) -> str:
    cid = make_corpus(headers)
    upload_doc(headers, cid, "Alpha Paper Title\nalpha beta gamma")
    client.post(f"/corpora/{cid}/train", headers=headers)  # mock_ml -> ready
    return cid


def _to_vllm(monkeypatch):
    monkeypatch.setattr(config, "INFERENCE_BACKEND", "vllm")


@pytest.fixture()
def fake_inference(monkeypatch):
    """Fake the vLLM /query. Records the question SERVED to the engine so a test can assert the served
    question is the ORIGINAL (never the condensed rewrite)."""
    seen: dict = {}

    def _iq(doc_ids, question, max_tokens=96, history=None, context="", **_):
        seen["served_question"] = question
        seen["doc_ids"] = list(doc_ids)
        return {"answer": "answer", "doc_ids": doc_ids,
                "metrics": {"latency_ms": 8.0, "prompt_tokens": 10, "gen_tokens": 3}}

    monkeypatch.setattr(ml_client, "inference_query", _iq)
    return seen


# --- pin refresh ------------------------------------------------------------------------------

def test_pinned_kept_when_fresh_top1_in_pinned(client, auth, make_corpus, upload_doc, mock_ml,
                                               fake_inference, monkeypatch, cart_id):
    """Fresh top-1 IS in the pinned set -> pinned list kept verbatim (no churn from rank jitter)."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    a = cart_id(cid)
    _to_vllm(monkeypatch)
    # Fresh retrieval returns a different ORDER but top-1 is still the pinned id -> kept verbatim.
    monkeypatch.setattr(retrieval, "retrieve", lambda *args, **k: [a, "some_other_ranked_id"])

    r = client.post(f"/corpora/{cid}/chat",
                    json={"question": "follow up?", "doc_ids": [a]}, headers=headers)
    assert r.status_code == 200, r.text
    assert r.json()["used_docs"] == [a]  # kept exactly as pinned


def test_topic_shift_uses_fresh_docs(client, auth, make_corpus, upload_doc, mock_ml,
                                     fake_inference, monkeypatch, cart_id):
    """Fresh top-1 is a STRANGER to the pinned set -> topic shift: serve the fresh docs and report
    them as used_docs (the client re-pins from the head frame). Fresh ids come straight from retrieve
    and bypass client validation; doc_sources tolerates an unknown id (falls back to the id as title)."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    a = cart_id(cid)  # the real, validating pinned id
    _to_vllm(monkeypatch)
    fresh = ["fresh_topic_doc"]  # top-1 is a stranger to the pinned {a}
    monkeypatch.setattr(retrieval, "retrieve", lambda *args, **k: fresh)

    r = client.post(f"/corpora/{cid}/chat",
                    json={"question": "totally new topic?", "doc_ids": [a]}, headers=headers)
    assert r.status_code == 200, r.text
    assert r.json()["used_docs"] == fresh  # followed the topic shift


def test_pin_refresh_off_keeps_pinned_and_skips_retrieve(client, auth, make_corpus, upload_doc,
                                                         mock_ml, fake_inference, monkeypatch,
                                                         cart_id):
    """CHAT_PIN_REFRESH=off -> pinned ids kept verbatim and retrieval.retrieve is NOT called for the
    refresh (today's trust-the-pin behavior exactly)."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    a = cart_id(cid)
    _to_vllm(monkeypatch)
    monkeypatch.setattr(config, "CHAT_PIN_REFRESH", "off")
    monkeypatch.setattr(retrieval, "retrieve",
                        lambda *args, **k: pytest.fail("pin-refresh off must not retrieve on a pin"))

    r = client.post(f"/corpora/{cid}/chat",
                    json={"question": "follow up?", "doc_ids": [a]}, headers=headers)
    assert r.status_code == 200, r.text
    assert r.json()["used_docs"] == [a]


# --- condensation wiring ----------------------------------------------------------------------

def test_effective_q_drives_retrieval_served_question_is_original(client, auth, make_corpus,
                                                                  upload_doc, mock_ml, fake_inference,
                                                                  monkeypatch, cart_id):
    """With history + a mocked condensation, the CONDENSED query is what retrieval + _hybrid_split
    receive, while the question SERVED to the engine stays the ORIGINAL. ChatResp.condensed_q is set."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    # Force a deterministic rewrite regardless of the (absent) engine.
    monkeypatch.setattr(condense, "standalone_question",
                        lambda history, question: "CONDENSED QUERY")
    retrieve_seen: dict = {}

    def _retrieve(corpus_id, question, k=None):
        retrieve_seen["question"] = question
        return [cart_id(cid)]

    split_seen: dict = {}
    real_split = chat_router._hybrid_split

    def _split(corpus_id, question, doc_ids):
        split_seen["question"] = question
        return real_split(corpus_id, question, doc_ids)

    monkeypatch.setattr(retrieval, "retrieve", _retrieve)
    monkeypatch.setattr(chat_router, "_hybrid_split", _split)

    r = client.post(f"/corpora/{cid}/chat",
                    json={"question": "what about its pricing?",
                          "history": [{"role": "user", "content": "tell me about Acme"},
                                      {"role": "assistant", "content": "Acme is our product."}]},
                    headers=headers)
    assert r.status_code == 200, r.text
    # Retrieval + routing saw the CONDENSED query...
    assert retrieve_seen["question"] == "CONDENSED QUERY"
    assert split_seen["question"] == "CONDENSED QUERY"
    # ...but the engine was SERVED the original question (the model has the history already).
    assert fake_inference["served_question"] == "what about its pricing?"
    # ...and the rewrite is surfaced for debug.
    assert r.json()["condensed_q"] == "CONDENSED QUERY"


def test_no_history_no_condense_no_condensed_q(client, auth, make_corpus, upload_doc, mock_ml,
                                               fake_inference, monkeypatch, cart_id):
    """Turn 1 (no history) -> condensation is a no-op, effective_q == original, condensed_q is None."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    monkeypatch.setattr(condense, "standalone_question",
                        lambda *a, **k: pytest.fail("no history -> _effective_question must not condense"))

    r = client.post(f"/corpora/{cid}/chat", json={"question": "what is alpha?"}, headers=headers)
    assert r.status_code == 200, r.text
    assert r.json()["condensed_q"] is None
    assert fake_inference["served_question"] == "what is alpha?"


# --- streaming path: condensed_q on the head frame + served question stays original ------------

@pytest.fixture()
def fake_engine_stream(monkeypatch):
    """Fake the GPU /query_stream SSE and capture the payload (so we can assert the SERVED question)."""
    captured: dict = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield 'data: {"delta": "hi"}'
            yield ('data: ' + json.dumps({"done": True, "metrics": {
                "latency_ms": 12.0, "prompt_tokens": 10, "gen_tokens": 3, "confidence": None}}))

    @contextmanager
    def _stream(method, url, **kw):
        captured["url"] = url
        captured["payload"] = kw.get("json")
        yield _Resp()

    monkeypatch.setattr(httpx, "stream", _stream)
    return captured


def test_stream_head_carries_condensed_q_and_serves_original(client, auth, make_corpus, upload_doc,
                                                            mock_ml, fake_engine_stream, monkeypatch,
                                                            cart_id):
    """chat_stream: the head frame carries condensed_q, retrieval/routing see the condensed query, and
    the /query_stream payload's question is the ORIGINAL (the model has the history as prefill)."""
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    _to_vllm(monkeypatch)
    monkeypatch.setattr(condense, "standalone_question",
                        lambda history, question: "CONDENSED QUERY")
    retrieve_seen: dict = {}

    def _retrieve(corpus_id, question, k=None):
        retrieve_seen["question"] = question
        return [cart_id(cid)]

    monkeypatch.setattr(retrieval, "retrieve", _retrieve)

    r = client.post(f"/corpora/{cid}/chat/stream",
                    json={"question": "what about its pricing?",
                          "history": [{"role": "user", "content": "tell me about Acme"},
                                      {"role": "assistant", "content": "Acme is our product."}]},
                    headers=headers)
    assert r.status_code == 200
    head = json.loads(r.text.split("\n\n")[0].removeprefix("data: "))
    assert head["condensed_q"] == "CONDENSED QUERY"
    assert retrieve_seen["question"] == "CONDENSED QUERY"       # retrieval saw the condensed query
    assert fake_engine_stream["payload"]["question"] == "what about its pricing?"  # served original
