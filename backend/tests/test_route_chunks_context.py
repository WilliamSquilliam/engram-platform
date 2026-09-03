"""retrieval.route_chunks_context — the QRC hybrid serving context builder (top-1 cart aside, this
routes docs 2..k). Exercised over a REAL corpus (the conftest fixtures give real storage + tenant),
lexical-only (RETRIEVAL_DENSE=off), so it's hermetic: no GPU, no fastembed download, no network.

Covers: the composed context contains ONLY the routed docs' own text, in the given doc order; the
per-doc budget is respected; unknown ids raise KeyError (mirrors context_for); and a chunk-description
sidecar steers selection without leaking its text into the served output."""
import json

import pytest
from app import config, retrieval
from app.storage import storage


def _corpus_with_docs(client, headers, cid, docs: dict[str, str]) -> None:
    for name, text in docs.items():
        r = client.post(f"/corpora/{cid}/documents",
                        files=[("files", (name, text, "text/plain"))], headers=headers)
        assert r.status_code == 200, r.text


def test_context_contains_only_routed_docs_in_order(client, auth, make_corpus, cart_id):
    """route_chunks_context over two small docs returns their own text, joined in the ID order given —
    and nothing from a doc that wasn't routed."""
    headers, _ = auth
    cid = make_corpus(headers)
    _corpus_with_docs(client, headers, cid, {
        "vela.txt": "The Vela observatory measures distant supernovae in the southern sky.",
        "zorb.txt": "The Zorblax reactor runs on deuterium at the northern research base.",
        "misc.txt": "An unrelated recipe about onions and garlic and slow-roasted tomatoes.",
    })
    vela, zorb = cart_id(cid, "vela.txt"), cart_id(cid, "zorb.txt")
    ctx = retrieval.route_chunks_context(cid, "tell me about the reactor and the observatory",
                                         [vela, zorb])
    # Both routed docs' own words are present, vela before zorb (input order preserved).
    assert "Vela observatory" in ctx and "Zorblax reactor" in ctx
    assert ctx.index("Vela") < ctx.index("Zorblax")
    # The un-routed doc never appears.
    assert "onions" not in ctx


def test_unknown_doc_id_raises_keyerror(client, auth, make_corpus, cart_id):
    """An id not in the corpus -> KeyError (same contract as context_for), never a silent empty serve."""
    headers, _ = auth
    cid = make_corpus(headers)
    _corpus_with_docs(client, headers, cid, {"vela.txt": "The Vela observatory measures supernovae."})
    with pytest.raises(KeyError) as excinfo:
        retrieval.route_chunks_context(cid, "vela", ["definitely-not-a-real-cart-id"])
    assert "unknown doc_ids" in str(excinfo.value)


def test_budget_respected_selects_not_whole_doc(client, auth, make_corpus, cart_id, monkeypatch):
    """A doc far over budget contributes only a routed slice, not its whole body. Pin a small
    QRC_BUDGET_TOKENS so the assertion is tight and independent of the shared-core default."""
    monkeypatch.setattr(config, "QRC_BUDGET_TOKENS", 48)
    headers, _ = auth
    cid = make_corpus(headers)
    filler = "irrelevant filler about gardening and the weather and slow cooking. " * 60
    needle = "The Vela observatory was commissioned in 2203 by Doctor Sasha Pol. " * 2
    big = filler + needle + filler
    _corpus_with_docs(client, headers, cid, {"big.txt": big})
    target = cart_id(cid, "big.txt")
    ctx = retrieval.route_chunks_context(cid, "who commissioned the Vela observatory", [target])
    assert 0 < len(ctx) < len(big)          # routed a slice, not the whole doc
    assert "Vela observatory" in ctx         # and it kept the answer-bearing region


def test_sidecar_descs_steer_selection_without_leaking(client, auth, make_corpus, cart_id, monkeypatch):
    """A chunk-description sidecar folds descs into the INDEX text only: a query that matches ONLY a
    chunk's description still routes that chunk, but the description text is never in the served output.
    Writing a fresh sidecar also invalidates the cached index (folded into the corpus signature)."""
    monkeypatch.setattr(config, "QRC_BUDGET_TOKENS", 48)
    headers, _ = auth
    cid = make_corpus(headers)
    # Body shares NO terms with the query; the description will.
    region_a = "xxxx yyyy zzzz wwww vvvv uuuu tttt ssss " * 10
    region_b = "aaaa bbbb cccc dddd eeee ffff gggg hhhh " * 10
    body = region_a + region_b
    _corpus_with_docs(client, headers, cid, {"ledger.txt": body})
    target = cart_id(cid, "ledger.txt")

    # Prime the index (no sidecar yet) so we prove the sidecar write invalidates the cache.
    retrieval.route_chunks_context(cid, "quarterly revenue figures", [target])

    from app import chunking
    n_chunks = len(chunking.chunk_spans(storage.read_text(cid, "ledger.txt")))
    descs = [""] * n_chunks
    descs[-1] = "quarterly revenue figures for the finance team"   # describe the LAST chunk
    storage.save_chunk_sidecar(cid, json.dumps({target: descs}).encode("utf-8"))

    ctx = retrieval.route_chunks_context(cid, "quarterly revenue figures", [target])
    # The description terms steered selection to the last region's words...
    assert "hhhh" in ctx
    # ...but the description text itself never leaks into the served context.
    assert "revenue" not in ctx and "finance" not in ctx


def test_missing_sidecar_routes_desc_less(client, auth, make_corpus, cart_id):
    """No sidecar at all -> routing still works (descs default to None), context is non-empty."""
    headers, _ = auth
    cid = make_corpus(headers)
    _corpus_with_docs(client, headers, cid, {
        "a.txt": "The Vela observatory measures distant supernovae in the southern sky. " * 5,
    })
    ctx = retrieval.route_chunks_context(cid, "vela observatory supernovae", [cart_id(cid, "a.txt")])
    assert "Vela observatory" in ctx
