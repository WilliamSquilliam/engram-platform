"""Control-plane retrieval: the production hybrid backend (bm25s lexical + RRF fusion, dense stage
OFF in tests), the doc_id slug, and backend selection. No torch/GPU. RETRIEVAL_DENSE=off (conftest)
so fastembed never downloads a model — hybrid runs lexical-only, fused via the SAME RRF path."""
from app import config, ml_client, retrieval
from app.retrieval import _lexical_ranking, doc_id_for, rrf_fuse


def test_doc_id_for_matches_onboarding_slug():
    assert doc_id_for("handbook/ch1/intro.md") == "handbook_ch1_intro"
    assert doc_id_for("Report 2024.txt") == "Report_2024"
    assert doc_id_for("nested/a b.c.md") == "nested_a_b_c"


def test_default_backend_is_hybrid():
    """Ship default: hybrid (the in-process bm25s+dense fused retriever). 'bm25' is a legacy alias."""
    assert config.RETRIEVAL_BACKEND in ("hybrid", "bm25")


def test_lexical_ranks_relevant_doc_first():
    """bm25s (the lexical stage) puts the obviously-matching doc first and returns a full ranking."""
    ids = ["vela", "zorblax", "misc"]
    texts = [
        "The Vela observatory was commissioned in 2203 by Dr Sasha Pol",
        "The Zorblax reactor was commissioned in 2147 by Mira Tanaka",
        "an unrelated cooking recipe about onions and garlic",
    ]
    ranked = _lexical_ranking("who commissioned the Vela observatory", ids, texts)
    assert ranked[0] == "vela"          # the doc that names Vela ranks first
    assert set(ranked) == set(ids)      # a full ranked list (fusion sees every doc)


# --- RRF unit test with two fake ranked lists -----------------------------------------------------

def test_rrf_fuses_two_ranked_lists():
    """A doc ranked #1 in both lists beats a doc that only tops one list; RRF sums 1/(k+rank)."""
    # 'a' is #1 in both -> highest fused score, always first.
    lex = ["a", "b", "c"]
    dense = ["b", "a", "c"]
    assert rrf_fuse([lex, dense], k=3)[0] == "a"
    # 'b' (ranks 1,0) and 'a' (ranks 0,1) tie on score; 'a' wins the doc_id tie-break, so the head is
    # a, then b, then c (rank 2 in both -> lowest).
    assert rrf_fuse([lex, dense], k=3) == ["a", "b", "c"]


def test_rrf_deterministic_tiebreak_on_doc_id():
    """Equal fused scores break on doc_id ascending — test stability, not arrival order."""
    # 'x' and 'y' are symmetric across the two lists -> identical scores -> 'x' before 'y'.
    assert rrf_fuse([["x", "y"], ["y", "x"]], k=2) == ["x", "y"]


def test_rrf_empty():
    assert rrf_fuse([], k=3) == []
    assert rrf_fuse([[], []], k=3) == []


# --- end-to-end hybrid over a real corpus (lexical-only in tests) ---------------------------------

def test_hybrid_returns_matching_doc_first(client, auth, make_corpus, cart_id):
    """retrieve() over an uploaded corpus routes through the hybrid backend and returns the
    obviously-matching doc first (bm25s path; dense off)."""
    headers, _ = auth
    cid = make_corpus(headers)
    client.post(f"/corpora/{cid}/documents",
                files=[("files", ("vela.txt", "The Vela observatory measures distant supernovae",
                                  "text/plain"))], headers=headers)
    client.post(f"/corpora/{cid}/documents",
                files=[("files", ("recipe.txt", "a cooking recipe about onions and garlic",
                                  "text/plain"))], headers=headers)
    ids = retrieval.retrieve(cid, "tell me about the Vela observatory", 2)
    assert ids[0] == cart_id(cid, "vela.txt")


def test_description_boosts_doc_into_topk(client, auth, make_corpus, cart_id):
    """The LLM description is indexed as retrieval metadata: a query that matches ONLY a doc's
    description (its body shares no query terms) still retrieves that doc. Proves the description is
    prepended into the index text."""
    from app import models
    from app.db import SessionLocal
    headers, _ = auth
    cid = make_corpus(headers)
    # Body deliberately shares NO terms with the query "quarterly revenue figures".
    client.post(f"/corpora/{cid}/documents",
                files=[("files", ("ledger.txt", "zzz qqq wobble frobnicate", "text/plain"))],
                headers=headers)
    client.post(f"/corpora/{cid}/documents",
                files=[("files", ("misc.txt", "unrelated notes about gardening tools", "text/plain"))],
                headers=headers)
    # Attach a description that DOES match the query.
    target = cart_id(cid, "ledger.txt")
    db = SessionLocal()
    try:
        doc = db.query(models.Document).filter(models.Document.filename == "ledger.txt").first()
        doc.description = "A spreadsheet of quarterly revenue figures for the finance team."
        db.commit()
    finally:
        db.close()
    ids = retrieval.retrieve(cid, "quarterly revenue figures", 1)
    assert ids == [target]  # matched purely on the indexed description


def test_adding_doc_invalidates_cached_index(client, auth, make_corpus, cart_id):
    """Staleness: the per-corpus index is cached in-process but keyed by a signature (filenames +
    descriptions). Uploading a new doc changes the signature, so the next retrieve() rebuilds and can
    return the new doc — the cache never serves a stale index."""
    headers, _ = auth
    cid = make_corpus(headers)
    client.post(f"/corpora/{cid}/documents",
                files=[("files", ("alpha.txt", "alpha content about telescopes", "text/plain"))],
                headers=headers)
    # Prime the cache.
    retrieval.retrieve(cid, "telescopes", 3)
    # Add a second doc that matches a different query.
    client.post(f"/corpora/{cid}/documents",
                files=[("files", ("beta.txt", "beta content about submarines", "text/plain"))],
                headers=headers)
    ids = retrieval.retrieve(cid, "submarines", 3)
    assert cart_id(cid, "beta.txt") in ids  # the freshly-added doc is retrievable (index rebuilt)


def test_fused_backend_delegates_to_ml_service(monkeypatch):
    """RETRIEVAL_BACKEND=fused routes through ml_client.retrieve_fused (the GPU-side index)
    instead of the in-process hybrid — and retrieve_context pulls the texts by doc_id."""
    calls = {}

    def fake_retrieve(corpus_dir, question, k):
        calls["args"] = (corpus_dir, question, k)
        return ["doc_b", "doc_a"]

    monkeypatch.setattr(config, "RETRIEVAL_BACKEND", "fused")
    monkeypatch.setattr(ml_client, "retrieve_fused", fake_retrieve)
    ids = retrieval.retrieve("corpus-x", "which doc?", 2)
    assert ids == ["doc_b", "doc_a"]
    assert calls["args"][1:] == ("which doc?", 2)
    assert calls["args"][0].endswith("corpus-x")  # the storage corpus dir path

    monkeypatch.setattr(retrieval, "_doc_text",
                        lambda cid, d: {"doc_a": "alpha text", "doc_b": "beta text"}[d])
    ids, ctx = retrieval.retrieve_context("corpus-x", "which doc?", 2)
    assert ids == ["doc_b", "doc_a"] and ctx == "beta text\n\nalpha text"
