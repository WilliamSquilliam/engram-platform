"""QRC chunk-description onboard pass (routers/jobs._write_chunk_descriptions). After a successful
vLLM onboard, and only when QRC_MODE=hybrid + QRC_CHUNK_DESC=on, one cart-resident generation per doc
yields a short line per chunk, parsed and written to the 'qrc_chunks.json' storage sidecar. Strictly
best-effort — any failure logs a warning and never fails onboarding (mirrors the doc-description pass).

The inference service is faked at the ml_client seam (no torch/GPU); storage is the real local sidecar."""
import json

from app import chunking, config
from app.storage import storage


def _stub_onboard_cag(monkeypatch):
    """Route onboarding through the vLLM branch (where the chunk-desc hook lives) with a success shape."""
    monkeypatch.setattr(config, "INFERENCE_BACKEND", "vllm")
    monkeypatch.setattr(
        "app.routers.jobs.ml_client.onboard_cag",
        lambda corpus_dir, docs, **kw: {"n_cartridges": len(docs), "canceled": False,
                                        "train_seconds": 1.0, "n_built": len(docs),
                                        "cart_seconds": 1.0, "corpus_tokens": 100},
    )
    # Doc descriptions off — isolate the chunk-desc pass under test.
    monkeypatch.setattr(config, "DOC_DESCRIPTIONS_ENABLED", False)


def _sidecar(cid: str) -> dict:
    raw = storage.read_chunk_sidecar_bytes(cid)
    return json.loads(raw) if raw else {}


def test_chunk_descs_written_when_gates_on(client, auth, make_corpus, upload_doc,
                                           mock_ml, monkeypatch, cart_id):
    """QRC_MODE=hybrid + QRC_CHUNK_DESC=on: onboarding calls the serving model per doc and writes a
    parsed descs list (length == chunk count) to the sidecar under the doc's cart id."""
    monkeypatch.setattr(config, "QRC_MODE", "hybrid")
    monkeypatch.setattr(config, "QRC_CHUNK_DESC", "on")
    _stub_onboard_cag(monkeypatch)

    # A long-enough doc that chunks into several spans, so parse_chunk_descs has multiple lines to fill.
    text = "The report covers revenue and churn and headcount across many regions. " * 12
    captured = {}

    def fake_iq(doc_ids, question, max_tokens=96, history=None, context=""):
        captured["doc_ids"] = list(doc_ids)
        captured["max_tokens"] = max_tokens
        n = len(chunking.chunk_spans(text))
        # Return a well-formed 'N. desc' reply for every chunk.
        reply = "\n".join(f"{i + 1}. Description of chunk {i + 1}." for i in range(n))
        return {"answer": reply}

    monkeypatch.setattr("app.routers.jobs.ml_client.inference_query", fake_iq)

    headers, _ = auth
    cid = make_corpus(headers)
    upload_doc(headers, cid, text)
    assert client.post(f"/corpora/{cid}/train", headers=headers).status_code == 200

    target = cart_id(cid, "a.txt")
    assert captured["doc_ids"] == [target]                 # generation rode the just-built cart
    assert captured["max_tokens"] == 384
    sidecar = _sidecar(cid)
    assert target in sidecar
    assert len(sidecar[target]) == len(chunking.chunk_spans(storage.read_text(cid, "a.txt")))
    assert sidecar[target][0].startswith("Description of chunk 1")


def test_chunk_descs_skipped_when_flag_off(client, auth, make_corpus, upload_doc,
                                           mock_ml, monkeypatch):
    """QRC_CHUNK_DESC=off: the pass never runs (no generation, no sidecar) — onboarding still succeeds."""
    monkeypatch.setattr(config, "QRC_MODE", "hybrid")
    monkeypatch.setattr(config, "QRC_CHUNK_DESC", "off")
    _stub_onboard_cag(monkeypatch)
    called = {"n": 0}

    def fake_iq(*a, **k):
        called["n"] += 1
        return {"answer": ""}

    monkeypatch.setattr("app.routers.jobs.ml_client.inference_query", fake_iq)

    headers, _ = auth
    cid = make_corpus(headers)
    upload_doc(headers, cid, "alpha beta gamma delta")
    assert client.post(f"/corpora/{cid}/train", headers=headers).status_code == 200

    assert called["n"] == 0
    assert _sidecar(cid) == {}
    assert client.get(f"/corpora/{cid}", headers=headers).json()["status"] == "ready"


def test_chunk_descs_skipped_when_mode_off(client, auth, make_corpus, upload_doc,
                                           mock_ml, monkeypatch):
    """QRC_MODE=off: the sidecar pass is gated off even with QRC_CHUNK_DESC=on."""
    monkeypatch.setattr(config, "QRC_MODE", "off")
    monkeypatch.setattr(config, "QRC_CHUNK_DESC", "on")
    _stub_onboard_cag(monkeypatch)
    monkeypatch.setattr("app.routers.jobs.ml_client.inference_query",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not generate")))

    headers, _ = auth
    cid = make_corpus(headers)
    upload_doc(headers, cid, "alpha beta gamma")
    assert client.post(f"/corpora/{cid}/train", headers=headers).status_code == 200
    assert _sidecar(cid) == {}


def test_chunk_desc_generation_failure_still_succeeds(client, auth, make_corpus, upload_doc,
                                                      mock_ml, monkeypatch):
    """A generation that RAISES must not fail onboarding: the corpus is still ready, sidecar simply
    stays absent (routing then runs desc-less)."""
    monkeypatch.setattr(config, "QRC_MODE", "hybrid")
    monkeypatch.setattr(config, "QRC_CHUNK_DESC", "on")
    _stub_onboard_cag(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("no inference service (as locally)")

    monkeypatch.setattr("app.routers.jobs.ml_client.inference_query", boom)

    headers, _ = auth
    cid = make_corpus(headers)
    upload_doc(headers, cid, "alpha beta gamma delta epsilon")
    assert client.post(f"/corpora/{cid}/train", headers=headers).status_code == 200

    assert client.get(f"/corpora/{cid}", headers=headers).json()["status"] == "ready"
    assert _sidecar(cid) == {}                              # no sidecar written on failure
