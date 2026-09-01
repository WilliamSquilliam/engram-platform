"""Resumable per-corpus onboarding wizard: step persistence + resume, the review estimate, and the
step-5 gate when no serving engine is available (the current placeholder-tier reality).

Steps 1-4 are exercised fully with no GPU; step 5's gate is asserted against the default (all-disabled)
serving registry. A monkeypatched "available" tier proves the ungated path dispatches through the
existing job machinery."""
from app import serving
from app.serving import ModelTier


def test_step_persistence_and_resume(client, auth, make_corpus):
    """PATCHing the wizard cursor + tier persists, and the corpus/onboarding responses resume it."""
    headers, _ = auth
    cid = make_corpus(headers)

    # New corpus starts at the entry step, no tier chosen.
    got = client.get(f"/corpora/{cid}", headers=headers).json()
    assert got["onboarding_step"] == "name"
    assert got["model_tier"] is None

    # Advance through steps; select a (placeholder) tier — a valid selection even though disabled.
    r = client.patch(f"/corpora/{cid}/onboarding",
                     json={"onboarding_step": "model", "model_tier": "balanced"}, headers=headers)
    assert r.status_code == 200, r.text
    state = r.json()
    assert state["onboarding_step"] == "model"
    assert state["model_tier"] == "balanced"

    # Resume: both the corpus response and the dedicated onboarding endpoint reopen at the step.
    assert client.get(f"/corpora/{cid}", headers=headers).json()["onboarding_step"] == "model"
    resumed = client.get(f"/corpora/{cid}/onboarding", headers=headers).json()
    assert resumed["onboarding_step"] == "model"
    assert resumed["model_tier"] == "balanced"


def test_patch_rejects_unknown_tier(client, auth, make_corpus):
    headers, _ = auth
    cid = make_corpus(headers)
    r = client.patch(f"/corpora/{cid}/onboarding", json={"model_tier": "nope"}, headers=headers)
    assert r.status_code == 400


def test_patch_rejects_bad_step(client, auth, make_corpus):
    headers, _ = auth
    cid = make_corpus(headers)
    # Not one of name|documents|model|review|onboarding|ready -> pydantic 422.
    r = client.patch(f"/corpora/{cid}/onboarding", json={"onboarding_step": "bogus"}, headers=headers)
    assert r.status_code == 422


def test_estimate(client, auth, make_corpus, upload_doc):
    """Review step: doc count, detected file types, total bytes, and a coarse time/cost estimate."""
    headers, _ = auth
    cid = make_corpus(headers)
    upload_doc(headers, cid, "some text here")  # uploads a.txt
    r = client.get(f"/corpora/{cid}/estimate", headers=headers)
    assert r.status_code == 200, r.text
    est = r.json()
    assert est["n_documents"] == 1
    assert est["file_types"] == {"txt": 1}
    assert est["total_bytes"] > 0
    # Coarse estimate is present and scales off the doc count.
    assert est["est_seconds"] > 0
    assert est["est_cost_ondemand"] >= 0
    assert est["est_cart_gb"] >= 0


def test_onboard_gate_no_serving_engine(client, auth, make_corpus, upload_doc, mock_ml):
    """Step 5 with the DEFAULT registry (all tiers disabled/placeholder): the gate returns a
    structured 409 {"status": "no_serving_engine"}, dispatches nothing, and leaves the cursor at
    'review' so the UI can show 'coming soon'."""
    headers, _ = auth
    cid = make_corpus(headers)
    upload_doc(headers, cid, "alpha beta")
    client.patch(f"/corpora/{cid}/onboarding",
                 json={"onboarding_step": "review", "model_tier": "balanced"}, headers=headers)

    r = client.post(f"/corpora/{cid}/onboard", headers=headers)
    assert r.status_code == 409
    assert r.json()["status"] == "no_serving_engine"

    # Nothing dispatched: no job created, corpus still 'new', cursor parked at 'review'.
    assert client.get(f"/corpora/{cid}/jobs", headers=headers).json() == []
    c = client.get(f"/corpora/{cid}", headers=headers).json()
    assert c["status"] == "new"
    assert c["onboarding_step"] == "review"


def test_onboard_requires_tier_and_docs(client, auth, make_corpus, upload_doc, mock_ml):
    headers, _ = auth
    cid = make_corpus(headers)
    # No tier selected yet -> 400.
    assert client.post(f"/corpora/{cid}/onboard", headers=headers).status_code == 400
    # Tier selected but no documents -> 400.
    client.patch(f"/corpora/{cid}/onboarding", json={"model_tier": "balanced"}, headers=headers)
    assert client.post(f"/corpora/{cid}/onboard", headers=headers).status_code == 400


def test_onboard_dispatches_when_tier_available(client, auth, make_corpus, upload_doc, mock_ml, monkeypatch):
    """When a tier IS available (engine enabled), /onboard pins model_ref, advances the cursor to the
    terminal 'ready' step via the existing worker, and marks documents onboarded — no gate."""
    headers, _ = auth
    cid = make_corpus(headers)
    upload_doc(headers, cid, "alpha beta")
    client.patch(f"/corpora/{cid}/onboarding",
                 json={"onboarding_step": "review", "model_tier": "balanced"}, headers=headers)

    # Flip the "balanced" tier to a real, enabled model for this test only.
    live = ModelTier("balanced", "Balanced", "test", model_ref="test/model-1",
                     precision="fp8", context_tokens=8192, enabled=True)
    monkeypatch.setattr(serving, "tier", lambda tid: live if tid == "balanced" else None)
    monkeypatch.setattr(serving, "model_ref_for_tier", lambda tid: live.model_ref)

    r = client.post(f"/corpora/{cid}/onboard", headers=headers)
    assert r.status_code == 200, r.text
    # TestClient runs the BackgroundTask synchronously, so the run completes here (mock_ml.train).
    state = r.json()
    assert state["model_ref"] == "test/model-1"

    c = client.get(f"/corpora/{cid}", headers=headers).json()
    assert c["status"] == "ready"
    assert c["onboarding_step"] == "ready"
    # Per-document status advanced to onboarded on success.
    docs = client.get(f"/corpora/{cid}/documents", headers=headers).json()
    assert all(d["onboard_status"] == "ready" for d in docs)
    # A job was created and succeeded via the reused job path.
    jobs = client.get(f"/corpora/{cid}/jobs", headers=headers).json()
    assert len(jobs) == 1 and jobs[0]["status"] == "succeeded"


def test_onboard_dispatch_sets_onboarding_step_only(client, auth, make_corpus, upload_doc,
                                                    mock_ml, monkeypatch):
    """F7 dispatch: /onboard flips onboard_status to 'onboarding' but must NOT write parse_status
    (an upload-time fact). We stop the run BEFORE the worker completes by having train hang the
    corpus in 'training' is hard here; instead assert the terminal success also never rewrites
    parse_status (the dispatch write is the only place parse_status was being touched)."""
    headers, _ = auth
    cid = make_corpus(headers)
    upload_doc(headers, cid, "alpha beta")
    client.patch(f"/corpora/{cid}/onboarding",
                 json={"onboarding_step": "review", "model_tier": "balanced"}, headers=headers)

    live = ModelTier("balanced", "Balanced", "test", model_ref="test/model-1",
                     precision="fp8", context_tokens=8192, enabled=True)
    monkeypatch.setattr(serving, "tier", lambda tid: live if tid == "balanced" else None)
    monkeypatch.setattr(serving, "model_ref_for_tier", lambda tid: live.model_ref)

    client.post(f"/corpora/{cid}/onboard", headers=headers)
    docs = client.get(f"/corpora/{cid}/documents", headers=headers).json()
    # parse_status is still the upload-time "parsed", never "parsing" (the removed dispatch write).
    assert all(d["parse_status"] == "parsed" for d in docs)


# --- F8: re-uploading to a ready corpus drops the wizard cursor back to "documents" --------

def test_reupload_to_ready_corpus_resets_step_to_documents(client, auth, make_corpus, upload_doc,
                                                           mock_ml, monkeypatch):
    """When a re-upload resets a ready corpus's status to 'new', the wizard cursor must also drop to
    'documents' so the two stay coherent (not left parked on the terminal 'ready' screen)."""
    headers, _ = auth
    cid = make_corpus(headers)
    upload_doc(headers, cid, "alpha beta")
    client.patch(f"/corpora/{cid}/onboarding",
                 json={"onboarding_step": "review", "model_tier": "balanced"}, headers=headers)

    live = ModelTier("balanced", "Balanced", "test", model_ref="test/model-1",
                     precision="fp8", context_tokens=8192, enabled=True)
    monkeypatch.setattr(serving, "tier", lambda tid: live if tid == "balanced" else None)
    monkeypatch.setattr(serving, "model_ref_for_tier", lambda tid: live.model_ref)

    client.post(f"/corpora/{cid}/onboard", headers=headers)
    c = client.get(f"/corpora/{cid}", headers=headers).json()
    assert c["status"] == "ready" and c["onboarding_step"] == "ready"

    # Re-upload a new document -> status back to 'new' AND cursor back to 'documents'.
    upload_doc(headers, cid, "brand new content")
    c = client.get(f"/corpora/{cid}", headers=headers).json()
    assert c["status"] == "new"
    assert c["onboarding_step"] == "documents"
