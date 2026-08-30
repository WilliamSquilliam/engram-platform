def test_train_succeeds(client, auth, make_corpus, upload_doc, mock_ml):
    headers, _ = auth
    cid = make_corpus(headers)
    upload_doc(headers, cid, "alpha beta")

    r = client.post(f"/corpora/{cid}/train", headers=headers)
    assert r.status_code == 200
    # TestClient runs the BackgroundTask synchronously, so training is done here.
    c = client.get(f"/corpora/{cid}", headers=headers).json()
    assert c["status"] == "ready"
    assert c["n_cartridges"] == 1
    assert c["train_seconds"] == 1.0
    assert c["mcp_token"]  # an MCP token is minted on first success


def test_train_requires_documents(client, auth, make_corpus, mock_ml):
    headers, _ = auth
    cid = make_corpus(headers)
    assert client.post(f"/corpora/{cid}/train", headers=headers).status_code == 400


def test_cancel_without_running_job(client, auth, make_corpus, upload_doc, mock_ml):
    headers, _ = auth
    cid = make_corpus(headers)
    upload_doc(headers, cid)
    client.post(f"/corpora/{cid}/train", headers=headers)  # completes immediately
    # No running job left -> cancel is a 409.
    assert client.post(f"/corpora/{cid}/cancel", headers=headers).status_code == 409


def test_jobs_listed(client, auth, make_corpus, upload_doc, mock_ml):
    headers, _ = auth
    cid = make_corpus(headers)
    upload_doc(headers, cid)
    client.post(f"/corpora/{cid}/train", headers=headers)
    jobs = client.get(f"/corpora/{cid}/jobs", headers=headers).json()
    assert len(jobs) == 1
    assert jobs[0]["status"] == "succeeded"
