"""Unit tests for the fused retrieval index — CPU-only, encoders injected (no
sentence-transformers / GPU needed). Run from the repo root in a torch env:

    python -m pytest platform/ml_service/test_retrieval_fused.py -q
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval_fused import (  # noqa: E402
    BM25,
    FusedIndex,
    _doc_ranks,
    _rrf,
    _toks,
)

DOCS = {
    "reactor": "The Zorblax fusion reactor study. Commissioned 2147 by Mira Tanaka. "
               + "Plasma confinement geometry and coolant loop notes. " * 40,
    "ocean": "The Vela deep-sea observatory report. Commissioned 2203 by Sasha Pol. "
             + "Hydrophone arrays and abyssal pressure ratings. " * 40,
    "orchard": "The orchard blight ledger. Fungus spread along irrigation channels. "
               + "Spore counts and seasonal rainfall tables. " * 40,
}


def _fake_embed(texts):
    """Deterministic 'meaning' vectors from bag-of-words hashes — enough to make
    same-topic texts near and different-topic texts far."""
    out = torch.zeros(len(texts), 64)
    for i, t in enumerate(texts):
        for w in _toks(t):
            out[i, hash(w) % 64] += 1.0
    return torch.nn.functional.normalize(out, dim=-1)


def _fake_rerank(pairs):
    return [sum(1.0 for w in _toks(q) if w in _toks(d)) for q, d in pairs]


def _index():
    return FusedIndex(DOCS, embed_fn=_fake_embed, rerank_fn=_fake_rerank)


def test_bm25_scores_rare_terms_higher():
    bm = BM25([_toks(t) for t in DOCS.values()])
    s = bm.scores(_toks("hydrophone abyssal"))
    assert max(s, key=s.get) == 1  # the ocean doc


def test_rrf_prefers_multi_list_agreement():
    assert _rrf([["a", "b", "c"], ["b", "a", "c"], ["b", "x", "y"]])[0] == "b"


def test_retrieve_finds_the_right_doc():
    idx = _index()
    idx.embed()
    assert idx.retrieve("who commissioned the fusion reactor?", k=1) == ["reactor"]
    assert idx.retrieve("deep sea pressure observatory", k=1) == ["ocean"]
    assert idx.retrieve("irrigation fungus in the orchard", k=1) == ["orchard"]


def test_retrieve_lexical_only_without_embeddings():
    idx = _index()          # .embed() never called -> BM25+RRF+rerank path only
    assert idx.retrieve("hydrophone array report", k=1) == ["ocean"]


def _sequential_retrieve(idx, question, k=3, pool=50):
    """Reference retrieve WITHOUT the BM25/embed overlap: the same three rank lists assembled
    strictly in order, then RRF + rerank. The threaded retrieve() must return an identical ranking."""
    qt = _toks(question)
    lists = [_doc_ranks(*idx.bm_head, qt), _doc_ranks(*idx.bm_whole, qt)]
    if idx.mat is not None:
        qv = idx.embed_fn([idx.q_prompt + question])[0].to(idx.mat.device)
        lists.append(idx._dense_ranks(qv))
    cand = _rrf(lists, topn=pool)[:pool]
    if idx.rerank_fn is not None and cand:
        scores = idx.rerank_fn([(question, idx.head[d]) for d in cand])
        cand = [d for d, _ in sorted(zip(cand, scores, strict=True),
                                     key=lambda x: x[1], reverse=True)]
    return cand[:k]


def test_overlap_matches_sequential_ranking():
    """The threaded BM25+embed overlap must not perturb ranking vs the sequential path."""
    idx = _index()
    idx.embed()
    for q in ("who commissioned the fusion reactor?", "deep sea pressure observatory",
              "irrigation fungus in the orchard", "hydrophone arrays and rainfall"):
        assert idx.retrieve(q, k=3) == _sequential_retrieve(idx, q, k=3)


def test_save_load_roundtrip(tmp_path):
    idx = _index()
    idx.embed()
    idx.save(tmp_path / "fused_index.pt")
    idx2 = FusedIndex.load(tmp_path / "fused_index.pt", DOCS,
                           embed_fn=_fake_embed, rerank_fn=_fake_rerank)
    assert torch.equal(idx.mat.cpu(), idx2.mat.cpu())
    assert idx2.retrieve("who commissioned the fusion reactor?", k=1) == ["reactor"]
