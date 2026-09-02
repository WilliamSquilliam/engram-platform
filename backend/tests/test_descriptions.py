"""LLM doc descriptions at onboarding (Feature 1), flag-gated DEFAULT ON.

The describe pass hooks into the vLLM onboard success branch (routers/jobs._run_training): on success
it calls ml_client.inference_describe for the onboarded cart ids and writes each returned description
to the matching Document row. STRICTLY best-effort — a describe failure never fails onboarding.

Covered here:
  * flag ON  -> describe is called with the onboarded ids and the result is persisted to
                Document.description (and surfaced via the documents API).
  * flag OFF -> describe is NOT called; description stays null.
  * describe RAISES -> onboarding still succeeds (corpus ready) and description stays null.
  * onboard_estimate includes the DESCRIBE_S_PER_DOC term only when the flag is on.
  * Alembic has a single head (the new 0011 migration didn't fork the graph).
"""
from app import config


def _stub_onboard_cag(monkeypatch):
    """Route the onboard through the vLLM branch (where the describe hook lives) with a success shape."""
    monkeypatch.setattr(config, "INFERENCE_BACKEND", "vllm")
    monkeypatch.setattr(
        "app.routers.jobs.ml_client.onboard_cag",
        lambda corpus_dir, docs, **kw: {"n_cartridges": len(docs), "canceled": False,
                                        "train_seconds": 1.0, "n_built": len(docs),
                                        "cart_seconds": 1.0, "corpus_tokens": 100},
    )


def _docs(client, headers, cid):
    return client.get(f"/corpora/{cid}/documents", headers=headers).json()


def test_flag_on_calls_describe_and_persists(client, auth, make_corpus, upload_doc,
                                             mock_ml, monkeypatch, cart_id):
    """Flag ON: the success branch calls inference_describe with the onboarded ids and writes the
    returned text to Document.description."""
    monkeypatch.setattr(config, "DOC_DESCRIPTIONS_ENABLED", True)
    _stub_onboard_cag(monkeypatch)
    captured = {}

    def fake_describe(doc_ids, max_tokens=60):
        captured["ids"] = list(doc_ids)
        captured["max_tokens"] = max_tokens
        return {"descriptions": {doc_ids[0]: "A short summary of the document."}}

    monkeypatch.setattr("app.routers.jobs.ml_client.inference_describe", fake_describe)

    headers, _ = auth
    cid = make_corpus(headers)
    upload_doc(headers, cid, "alpha beta gamma")
    assert client.post(f"/corpora/{cid}/train", headers=headers).status_code == 200

    assert captured["ids"] == [cart_id(cid, "a.txt")]        # onboarded id passed to describe
    assert captured["max_tokens"] == config.DESCRIBE_MAX_TOKENS
    docs = _docs(client, headers, cid)
    assert docs[0]["description"] == "A short summary of the document."


def test_flag_off_does_not_call_describe(client, auth, make_corpus, upload_doc,
                                         mock_ml, monkeypatch):
    """Flag OFF: describe is never called and the description stays null."""
    monkeypatch.setattr(config, "DOC_DESCRIPTIONS_ENABLED", False)
    _stub_onboard_cag(monkeypatch)
    called = {"n": 0}

    def fake_describe(doc_ids, max_tokens=60):
        called["n"] += 1
        return {"descriptions": {}}

    monkeypatch.setattr("app.routers.jobs.ml_client.inference_describe", fake_describe)

    headers, _ = auth
    cid = make_corpus(headers)
    upload_doc(headers, cid, "alpha beta")
    assert client.post(f"/corpora/{cid}/train", headers=headers).status_code == 200

    assert called["n"] == 0
    docs = _docs(client, headers, cid)
    assert docs[0]["description"] is None


def test_describe_failure_still_succeeds(client, auth, make_corpus, upload_doc,
                                         mock_ml, monkeypatch):
    """Describe RAISING must not fail onboarding: the corpus is still ready and description stays null."""
    monkeypatch.setattr(config, "DOC_DESCRIPTIONS_ENABLED", True)
    _stub_onboard_cag(monkeypatch)

    def boom(doc_ids, max_tokens=60):
        raise RuntimeError("no inference service (as locally)")

    monkeypatch.setattr("app.routers.jobs.ml_client.inference_describe", boom)

    headers, _ = auth
    cid = make_corpus(headers)
    upload_doc(headers, cid, "alpha beta")
    assert client.post(f"/corpora/{cid}/train", headers=headers).status_code == 200

    c = client.get(f"/corpora/{cid}", headers=headers).json()
    assert c["status"] == "ready"                    # onboarding succeeded despite the failure
    docs = _docs(client, headers, cid)
    assert docs[0]["onboard_status"] == "ready"      # doc still onboarded
    assert docs[0]["description"] is None             # description simply stayed null


# --- estimate term is gated on the flag -----------------------------------------------------------

def test_estimate_includes_describe_term_only_when_flag_on(monkeypatch):
    from app import metrics
    monkeypatch.setattr(config, "DOC_DESCRIPTIONS_ENABLED", True)
    on = metrics.onboard_estimate(10)["est_seconds"]
    monkeypatch.setattr(config, "DOC_DESCRIPTIONS_ENABLED", False)
    off = metrics.onboard_estimate(10)["est_seconds"]
    # The ONLY difference is 10 docs x DESCRIBE_S_PER_DOC.
    assert round(on - off, 3) == round(10 * metrics.DESCRIBE_S_PER_DOC, 3)
    assert on > off


# --- migration graph stays single-head ------------------------------------------------------------

def test_alembic_single_head():
    """Migrations must not fork the revision graph (a multi-head DB won't upgrade cleanly). Pinned to
    the current head so a stray parallel migration is caught; bump this when adding a new revision."""
    from pathlib import Path

    from alembic.config import Config
    from alembic.script import ScriptDirectory

    backend_dir = Path(__file__).resolve().parents[1]
    cfg = Config(str(backend_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(backend_dir / "alembic"))
    heads = ScriptDirectory.from_config(cfg).get_heads()
    assert list(heads) == ["0012_tenant_billing"], heads
