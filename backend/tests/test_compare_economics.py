def _ready_corpus(client, headers, make_corpus, upload_doc) -> str:
    cid = make_corpus(headers)
    upload_doc(headers, cid, "alpha beta gamma")
    client.post(f"/corpora/{cid}/train", headers=headers)  # mock_ml -> ready
    return cid


def test_compare_returns_strategies(client, auth, make_corpus, upload_doc, mock_ml):
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    r = client.post(f"/corpora/{cid}/compare", json={"question": "q?", "k": 3}, headers=headers)
    assert r.status_code == 200
    data = r.json()
    # The product (Engram Smart CAG = the adaptive cartridge router) vs the only realistic baseline (RAG).
    assert [s["key"] for s in data["strategies"]] == ["everyday", "rag"]

    by = {s["key"]: s for s in data["strategies"]}
    assert by["everyday"]["cost_per_query"] is not None
    assert by["rag"]["cost_per_query"] is not None
    # Engram Smart CAG carries the adaptive routing readout; here it stayed on the cartridge alone.
    assert by["everyday"]["tier"] == "cartridge"
    assert by["everyday"]["raw_tokens"] == 0


def test_compare_requires_ready(client, auth, make_corpus, mock_ml):
    headers, _ = auth
    cid = make_corpus(headers)  # status "new"
    r = client.post(f"/corpora/{cid}/compare", json={"question": "q?"}, headers=headers)
    assert r.status_code == 400


def test_economics_shape(client, auth, make_corpus, upload_doc, mock_ml):
    headers, _ = auth
    cid = _ready_corpus(client, headers, make_corpus, upload_doc)
    e = client.get(f"/corpora/{cid}/economics", headers=headers).json()
    assert e["trained"] is True
    assert e["n_cartridges"] == 1
    assert e["per_query"]["everyday"] > 0
    assert e["per_query"]["rag"] > 0
    # break-even vs rag should be a finite, non-negative query count
    assert e["breakeven_vs_rag"] is None or e["breakeven_vs_rag"] >= 0
