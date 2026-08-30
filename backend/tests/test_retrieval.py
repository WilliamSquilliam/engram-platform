"""Control-plane retrieval: BM25 ranking + the doc_id slug + backend selection. No torch/GPU."""
from app import config, ml_client, retrieval
from app.retrieval import bm25_rank, doc_id_for


def test_doc_id_for_matches_onboarding_slug():
    assert doc_id_for("handbook/ch1/intro.md") == "handbook_ch1_intro"
    assert doc_id_for("Report 2024.txt") == "Report_2024"
    assert doc_id_for("nested/a b.c.md") == "nested_a_b_c"


def test_bm25_ranks_relevant_doc_first():
    docs = [
        ("vela", "The Vela observatory was commissioned in 2203 by Dr Sasha Pol".lower().split()),
        ("zorblax", "The Zorblax reactor was commissioned in 2147 by Mira Tanaka".lower().split()),
        ("misc", "an unrelated cooking recipe about onions and garlic".lower().split()),
    ]
    top = bm25_rank("who commissioned the Vela observatory", docs, k=2)
    assert top[0] == "vela"          # the doc that names Vela ranks first
    assert "misc" not in top         # the irrelevant doc is excluded from top-2


def test_bm25_empty_corpus():
    assert bm25_rank("anything", [], k=3) == []


def test_bm25_no_match_falls_back_to_first():
    # query shares no terms with any doc -> return something (the first), never empty
    docs = [("a", ["alpha"]), ("b", ["beta"])]
    assert bm25_rank("zzz qqq", docs, k=1) == ["a"]


def test_fused_backend_delegates_to_ml_service(monkeypatch):
    """RETRIEVAL_BACKEND=fused routes through ml_client.retrieve_fused (the GPU-side index)
    instead of the in-process BM25 — and retrieve_context pulls the texts by doc_id."""
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
