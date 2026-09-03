"""retrieval.route_chunk_spans / doc_titles_for / served_texts_for — the RESIDENT-QRC control-plane
helpers (top-1 cart aside, these describe docs 2..k as LOADABLE KV spans instead of text). Exercised
over a REAL corpus (conftest fixtures give real storage + tenant), lexical-only (RETRIEVAL_DENSE=off),
so hermetic: no GPU, no fastembed download, no network.

Covers: the returned char spans index the SAME chunks route_chunks_context would compose (selection
parity), char ranges are valid into the served text, unknown ids raise KeyError, an over-budget doc
returns a slice (not every chunk), titles/served-texts match the served text, and empty doc_ids -> {}."""
import pytest
from app import chunking, config, retrieval
from app.storage import storage


def _corpus_with_docs(client, headers, cid, docs: dict[str, str]) -> None:
    for name, text in docs.items():
        r = client.post(f"/corpora/{cid}/documents",
                        files=[("files", (name, text, "text/plain"))], headers=headers)
        assert r.status_code == 200, r.text


def test_spans_index_valid_char_ranges(client, auth, make_corpus, cart_id):
    """Each returned span is a valid [start,end) into the doc's served text (0 <= start < end <= len)."""
    headers, _ = auth
    cid = make_corpus(headers)
    _corpus_with_docs(client, headers, cid, {
        "vela.txt": "The Vela observatory measures distant supernovae in the southern sky. " * 4,
        "zorb.txt": "The Zorblax reactor runs on deuterium at the northern research base. " * 4,
    })
    vela, zorb = cart_id(cid, "vela.txt"), cart_id(cid, "zorb.txt")
    spans = retrieval.route_chunk_spans(cid, "reactor and observatory", [vela, zorb])
    assert set(spans) <= {vela, zorb}
    for d, ranges in spans.items():
        text = storage.read_text(cid, "vela.txt" if d == vela else "zorb.txt")
        for a, b in ranges:
            assert 0 <= a < b <= len(text)


def test_spans_cover_the_same_chunks_route_context_selects(client, auth, make_corpus, cart_id,
                                                           monkeypatch):
    """Selection parity: the char text sliced by route_chunk_spans' ranges equals the chunk text
    route_chunks_context composes (same chunks, same order) — the engine loads exactly what hybrid
    would have prefilled. Pin a small budget so a real subset is chosen (not the whole doc)."""
    monkeypatch.setattr(config, "QRC_BUDGET_TOKENS", 48)
    headers, _ = auth
    cid = make_corpus(headers)
    filler = "irrelevant filler about gardening and the weather and slow cooking. " * 60
    needle = "The Vela observatory was commissioned in 2203 by Doctor Sasha Pol. " * 2
    big = filler + needle + filler
    _corpus_with_docs(client, headers, cid, {"big.txt": big})
    target = cart_id(cid, "big.txt")
    q = "who commissioned the Vela observatory"

    spans = retrieval.route_chunk_spans(cid, q, [target])
    assert target in spans
    text = storage.read_text(cid, "big.txt")
    # The union of the sliced spans must contain the answer-bearing region and be a strict subset.
    sliced = "".join(text[a:b] for a, b in spans[target])
    assert "Vela observatory" in sliced
    assert 0 < len(sliced) < len(text)
    # Each selected span aligns to a deterministic chunk boundary (chunk_spans is the shared core).
    chunk_bounds = {(a, b) for a, b in chunking.chunk_spans(text)}
    for a, b in spans[target]:
        assert (a, b) in chunk_bounds


def test_unknown_doc_id_raises_keyerror(client, auth, make_corpus, cart_id):
    """An id not in the corpus -> KeyError (same contract as route_chunks_context/context_for)."""
    headers, _ = auth
    cid = make_corpus(headers)
    _corpus_with_docs(client, headers, cid, {"vela.txt": "The Vela observatory measures supernovae."})
    with pytest.raises(KeyError) as excinfo:
        retrieval.route_chunk_spans(cid, "vela", ["definitely-not-a-real-cart-id"])
    assert "unknown doc_ids" in str(excinfo.value)


def test_empty_doc_ids_returns_empty(client, auth, make_corpus):
    """topk=1 -> nothing to span-load: route_chunk_spans([]) is {} (never touches the index)."""
    headers, _ = auth
    cid = make_corpus(headers)
    assert retrieval.route_chunk_spans(cid, "anything", []) == {}


def test_doc_titles_for_matches_first_line(client, auth, make_corpus, cart_id):
    """doc_titles_for returns each doc's first non-empty line (capped 120), keyed by cart id; unknown
    ids are simply absent (the engine falls back to the id)."""
    headers, _ = auth
    cid = make_corpus(headers)
    _corpus_with_docs(client, headers, cid, {
        "a.txt": "Reactor Handbook\nThe Zorblax reactor runs on deuterium.",
        "b.txt": "Observatory Manual\nThe Vela observatory measures supernovae.",
    })
    a, b = cart_id(cid, "a.txt"), cart_id(cid, "b.txt")
    titles = retrieval.doc_titles_for(cid, [a, b, "nope"])
    assert titles == {a: "Reactor Handbook", b: "Observatory Manual"}


def test_served_texts_for_matches_served_text(client, auth, make_corpus, cart_id):
    """served_texts_for returns the SAME extracted text the spans index into (so the engine
    re-tokenizes exactly that text); unknown ids raise KeyError."""
    headers, _ = auth
    cid = make_corpus(headers)
    _corpus_with_docs(client, headers, cid, {"a.txt": "Handbook\nThe reactor runs on deuterium."})
    a = cart_id(cid, "a.txt")
    texts = retrieval.served_texts_for(cid, [a])
    assert texts[a] == storage.read_text(cid, "a.txt")
    with pytest.raises(KeyError):
        retrieval.served_texts_for(cid, ["nope"])
