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


# --- F4: the internal progress endpoint fails CLOSED when unauthenticated -----------------

def test_progress_endpoint_503_when_token_unset(client, monkeypatch):
    """With INTERNAL_API_TOKEN unset the worker->control-plane progress callback would be
    unauthenticated, so it must 503 (mirror gc_carts) rather than skip the check and accept anyone."""
    from app import config
    monkeypatch.setattr(config, "INTERNAL_API_TOKEN", "")
    r = client.post("/internal/jobs/any-job-id/progress", json={"progress": 0.5})
    assert r.status_code == 503


def test_progress_endpoint_rejects_bad_token_when_set(client, monkeypatch):
    """With a token configured, a wrong/missing X-Internal-Token is 401 (not 503)."""
    from app import config
    monkeypatch.setattr(config, "INTERNAL_API_TOKEN", "s" * 40)
    r = client.post("/internal/jobs/any-job-id/progress", json={"progress": 0.5})
    assert r.status_code == 401
    r = client.post("/internal/jobs/any-job-id/progress", json={"progress": 0.5},
                    headers={"X-Internal-Token": "wrong"})
    assert r.status_code == 401


# --- F7: per-doc onboard status state machine; parse_status is an upload-time fact ---------

def _train_status(client, headers, cid) -> tuple[str, list[dict]]:
    """Run a train and return (corpus_status, documents)."""
    client.post(f"/corpora/{cid}/train", headers=headers)
    c = client.get(f"/corpora/{cid}", headers=headers).json()
    docs = client.get(f"/corpora/{cid}/documents", headers=headers).json()
    return c["status"], docs


def test_success_sweep_sets_ready_and_preserves_parse_status(client, auth, make_corpus,
                                                             upload_doc, mock_ml):
    """A successful onboard marks a doc with text onboard_status=ready and NEVER rewrites its
    parse_status (an upload-time fact — it stays whatever upload set)."""
    headers, _ = auth
    cid = make_corpus(headers)
    upload_doc(headers, cid, "alpha beta")
    status, docs = _train_status(client, headers, cid)
    assert status == "ready"
    assert all(d["onboard_status"] == "ready" for d in docs)
    # parse_status was set at UPLOAD to "parsed"; the success sweep must not have clobbered it to
    # anything else (it's still exactly "parsed", the upload-time value).
    assert all(d["parse_status"] == "parsed" for d in docs)


def test_empty_doc_lands_failed_not_ready(client, auth, make_corpus, mock_ml):
    """A document whose extracted text is EMPTY (an upload-time parse failure) is excluded from the
    onboard AND marked onboard_status=failed on the success sweep — not falsely reported ready."""
    headers, _ = auth
    cid = make_corpus(headers)
    # A good doc plus a corrupt PDF (parses to empty text at upload -> parse_status failed, empty sidecar).
    client.post(f"/corpora/{cid}/documents",
                files=[("files", ("good.txt", "real content here", "text/plain"))], headers=headers)
    client.post(f"/corpora/{cid}/documents",
                files=[("files", ("broken.pdf", b"%PDF-1.4 garbage", "application/pdf"))], headers=headers)

    status, docs = _train_status(client, headers, cid)
    assert status == "ready"
    by_name = {d["filename"]: d for d in docs}
    assert by_name["good.txt"]["onboard_status"] == "ready"
    # The empty-text doc is failed, and its upload-time parse_status ("failed") is untouched.
    assert by_name["broken.pdf"]["onboard_status"] == "failed"
    assert by_name["broken.pdf"]["parse_status"] == "failed"


def test_failed_onboard_marks_docs_failed_leaves_parse_status(client, auth, make_corpus,
                                                             upload_doc, mock_ml, monkeypatch):
    """When the onboard run raises, docs land onboard_status=failed and parse_status is left alone."""
    from app import ml_client
    headers, _ = auth
    cid = make_corpus(headers)
    upload_doc(headers, cid, "alpha beta")

    def boom(*a, **k):
        raise RuntimeError("gpu exploded")
    monkeypatch.setattr(ml_client, "train", boom)

    status, docs = _train_status(client, headers, cid)
    assert status == "failed"
    assert all(d["onboard_status"] == "failed" for d in docs)
    assert all(d["parse_status"] == "parsed" for d in docs)  # upload-time value preserved


def test_canceled_onboard_resets_docs_to_pending(client, auth, make_corpus, upload_doc,
                                                 mock_ml, monkeypatch):
    """A canceled onboard must reset docs to onboard_status=pending — not leave them stuck
    'onboarding'/'parsing' forever."""
    from app import ml_client
    headers, _ = auth
    cid = make_corpus(headers)
    upload_doc(headers, cid, "alpha beta")

    def canceled(*a, **k):
        return {"canceled": True}
    monkeypatch.setattr(ml_client, "train", canceled)

    status, docs = _train_status(client, headers, cid)
    assert status == "new"  # corpus returns to new so it can be retrained
    assert all(d["onboard_status"] == "pending" for d in docs)
    assert all(d["parse_status"] == "parsed" for d in docs)
