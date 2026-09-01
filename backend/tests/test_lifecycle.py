"""Cartridge lifecycle: corpus delete offboards + invalidates the right carts (excluding slugs a
sibling corpus still references), the training success path invalidates re-onboarded carts, the audit
trail is tenant-isolated, and the operator GC sweep is gated + dry-run-by-default. The GPU ML plane is
mocked (conftest.mock_ml) so these assert the CONTROL-PLANE contract, not real cart I/O."""
import uuid

from app import config
from app.db import SessionLocal
from app.models import AuditEvent


def _register(client) -> dict:
    email = f"{uuid.uuid4().hex[:8]}@t.local"
    tok = client.post(
        "/auth/register",
        json={"email": email, "password": "pw123456", "tenant_name": "t"},
    ).json()["access_token"]
    return {"Authorization": f"Bearer {tok}"}


def _upload(client, headers, corpus_id, filename, text="hello world"):
    r = client.post(
        f"/corpora/{corpus_id}/documents",
        files=[("files", (filename, text, "text/plain"))],
        headers=headers,
    )
    assert r.status_code == 200, r.text


def _audit_rows(tenant_filter=None):
    """Read audit rows straight from the DB (the /audit route is tenant-scoped; some assertions want
    the raw table, e.g. the _system GC row)."""
    db = SessionLocal()
    try:
        q = db.query(AuditEvent)
        if tenant_filter is not None:
            q = q.filter(AuditEvent.tenant_id == tenant_filter)
        return q.order_by(AuditEvent.created_at.desc(), AuditEvent.id.desc()).all()
    finally:
        db.close()


def test_delete_offboards_and_invalidates(client, auth, make_corpus, mock_ml, cart_id):
    headers, _ = auth
    cid = make_corpus(headers)
    _upload(client, headers, cid, "handbook/intro.md")  # slug: handbook_intro
    _upload(client, headers, cid, "readme.txt")          # slug: readme
    ids = sorted([cart_id(cid, "handbook/intro.md"), cart_id(cid, "readme.txt")])

    assert client.delete(f"/corpora/{cid}", headers=headers).status_code == 204

    # Both this corpus's carts were offboarded AND invalidated (order within the set is sorted()).
    assert mock_ml.offboard_ids == [ids]
    assert mock_ml.invalidate_ids == [ids]
    # ORDERING CONTRACT: invalidate (tombstone fan-out purge) is published BEFORE the durable offboard,
    # so a failed offboard degrades to a stale blob GC reaps (safe) rather than a warm-but-deleted cache
    # with no recovery. The interleaved call log proves the sequence, not just per-call presence.
    assert mock_ml.order == [("invalidate", ids), ("offboard", ids)]

    # An audit receipt exists for this tenant naming the deleted carts.
    rows = _audit_rows()
    row = next(r for r in rows if r.event == "corpus.delete" and r.corpus_id == cid)
    assert row.tenant_id  # tenant-scoped
    assert ids[0] in row.detail and ids[1] in row.detail
    assert '"n_documents": 2' in row.detail


def test_cross_tenant_same_filename_never_shares(client, mock_ml, cart_id):
    """E6 isolation: two DIFFERENT tenants each hold 'same.txt'. Because cart ids are tenant-namespaced,
    they resolve to DIFFERENT carts — so deleting one tenant's corpus offboards ITS carts in full and
    retains NOTHING as shared (the old cross-tenant 'skipped_shared' retention is gone)."""
    h1, h2 = _register(client), _register(client)
    c1 = client.post("/corpora", json={"name": "A"}, headers=h1).json()["id"]
    c2 = client.post("/corpora", json={"name": "B"}, headers=h2).json()["id"]
    _upload(client, h1, c1, "same.txt")   # tenant1's 'same'
    _upload(client, h1, c1, "only1.txt")  # tenant1's 'only1'
    _upload(client, h2, c2, "same.txt")   # tenant2's 'same' — a DIFFERENT cart id now
    # Capture c1's namespaced ids BEFORE deleting it (its tenant is unresolvable once the row is gone).
    c1_same, c1_only1 = cart_id(c1, "same.txt"), cart_id(c1, "only1.txt")
    c2_same = cart_id(c2, "same.txt")  # c2 survives the delete
    assert c2_same != c1_same          # the whole point: same filename, DIFFERENT tenant -> distinct id

    assert client.delete(f"/corpora/{c1}", headers=h1).status_code == 204

    # Both of c1's carts are offboarded/invalidated; the cross-tenant 'same' held by c2 is a different
    # id and is untouched, and nothing is retained-shared.
    expect = sorted([c1_same, c1_only1])
    assert mock_ml.offboard_ids == [expect]
    assert mock_ml.invalidate_ids == [expect]
    row = next(r for r in _audit_rows() if r.event == "corpus.delete" and r.corpus_id == c1)
    assert '"cart_ids_skipped_shared": []' in row.detail  # cross-tenant sharing no longer exists


def test_intra_tenant_shared_cart_retained(client, mock_ml, cart_id):
    """Intra-tenant correctness is KEPT: one tenant reuses 'same.txt' across TWO of its corpora, so both
    resolve to the SAME namespaced cart. Deleting one corpus must NOT offboard that cart (the sibling
    corpus still serves it) — it's listed as skipped_shared; only the unique cart is offboarded."""
    h1 = _register(client)
    c1 = client.post("/corpora", json={"name": "A"}, headers=h1).json()["id"]
    c2 = client.post("/corpora", json={"name": "B"}, headers=h1).json()["id"]  # SAME tenant
    _upload(client, h1, c1, "same.txt")   # shared with c2
    _upload(client, h1, c1, "only1.txt")  # unique to c1
    _upload(client, h1, c2, "same.txt")   # sibling corpus of the same tenant still references it
    # Same tenant, so c1 and c2 namespace identically; resolve via c2 (it survives the delete).
    shared_id, only1_id = cart_id(c2, "same.txt"), cart_id(c2, "only1.txt")

    assert client.delete(f"/corpora/{c1}", headers=h1).status_code == 204

    # Only the unique cart is offboarded; the shared-with-sibling cart is retained.
    assert mock_ml.offboard_ids == [[only1_id]]
    assert mock_ml.invalidate_ids == [[only1_id]]
    row = next(r for r in _audit_rows() if r.event == "corpus.delete" and r.corpus_id == c1)
    assert f'"cart_ids_skipped_shared": ["{shared_id}"]' in row.detail


def test_ml_plane_down_still_deletes(client, auth, make_corpus, mock_ml, monkeypatch):
    """The DB/storage deletion is authoritative: if the ML plane raises, the delete still returns 204
    and the error is captured (SANITIZED) in the audit receipt for an operator to reconcile later."""
    headers, _ = auth
    cid = make_corpus(headers)
    _upload(client, headers, cid, "doc.txt")

    def boom(_ids):
        raise RuntimeError("ml service unreachable")

    monkeypatch.setattr("app.routers.corpora.ml_client.offboard", boom)

    assert client.delete(f"/corpora/{cid}", headers=headers).status_code == 204
    row = next(r for r in _audit_rows() if r.event == "corpus.delete" and r.corpus_id == cid)
    assert '"offboard"' in row.detail          # error keyed by the call that failed
    assert "RuntimeError" in row.detail        # exception CLASS is recorded...
    assert "ml service unreachable" not in row.detail  # ...but never the raw message (finding 3)


def test_ml_error_detail_is_sanitized(client, auth, make_corpus, mock_ml, monkeypatch):
    """ML-plane errors in the tenant-visible audit detail must NEVER leak internal URLs: raw httpx
    errors embed the ML-plane URL. The receipt records a class name + fixed phrase only — no 'http'."""
    headers, _ = auth
    cid = make_corpus(headers)
    _upload(client, headers, cid, "doc.txt")

    def boom(_ids):
        # Mimic an httpx error whose message embeds the internal ML-plane URL.
        raise RuntimeError("Connection refused to http://ml-internal.svc.cluster.local:8002/invalidate")

    monkeypatch.setattr("app.routers.corpora.ml_client.offboard", boom)
    monkeypatch.setattr("app.routers.corpora.ml_client.inference_invalidate", boom)

    assert client.delete(f"/corpora/{cid}", headers=headers).status_code == 204
    row = next(r for r in _audit_rows() if r.event == "corpus.delete" and r.corpus_id == cid)
    assert "http" not in row.detail  # no leaked URL/scheme anywhere in the receipt


def test_invalidate_failure_still_offboards_and_flags(client, auth, make_corpus, mock_ml,
                                                      monkeypatch, cart_id):
    """If invalidate fails, offboard STILL runs (blob deletion isn't blocked), and the receipt carries
    invalidate_failed:true plus the affected ids so an operator can replay POST /invalidate."""
    headers, _ = auth
    cid = make_corpus(headers)
    _upload(client, headers, cid, "doc.txt")  # slug 'doc'
    doc_id = cart_id(cid, "doc.txt")

    def boom(_ids):
        raise RuntimeError("inference service unreachable")

    monkeypatch.setattr("app.routers.corpora.ml_client.inference_invalidate", boom)

    assert client.delete(f"/corpora/{cid}", headers=headers).status_code == 204
    # Offboard still ran despite the invalidate failure.
    assert mock_ml.offboard_ids == [[doc_id]]
    row = next(r for r in _audit_rows() if r.event == "corpus.delete" and r.corpus_id == cid)
    assert '"invalidate_failed": true' in row.detail
    assert f'"invalidate_failed_ids": ["{doc_id}"]' in row.detail


def test_n_documents_counts_rows_not_slugs(client, auth, make_corpus, mock_ml, cart_id):
    """n_documents is a ROW count: two files colliding to the same cart id are still two documents. The
    receipt reports what the user deleted, not the de-duplicated cart-id set."""
    headers, _ = auth
    cid = make_corpus(headers)
    # Both filenames slug to 'report' (doc_id_for drops the extension), so the namespaced cart id
    # collapses to one.
    _upload(client, headers, cid, "report.md")
    _upload(client, headers, cid, "report.txt")
    report = cart_id(cid, "report.md")  # capture before delete (corpus/tenant gone after)

    assert client.delete(f"/corpora/{cid}", headers=headers).status_code == 204
    row = next(r for r in _audit_rows() if r.event == "corpus.delete" and r.corpus_id == cid)
    assert '"n_documents": 2' in row.detail  # two rows...
    assert mock_ml.offboard_ids == [[report]]  # ...but one cart offboarded


def test_mid_training_delete_compensates(client, auth, make_corpus, mock_ml, monkeypatch, cart_id):
    """A corpus deleted WHILE its vLLM onboard is in flight resurrects offboarded blobs. The worker
    re-checks after onboard, finds the corpus gone, and runs a compensating invalidate+offboard over
    this run's carts, writing a corpus.delete_compensation receipt."""
    monkeypatch.setattr(config, "INFERENCE_BACKEND", "vllm")
    headers, _ = auth
    cid = make_corpus(headers)
    _upload(client, headers, cid, "beta.txt")  # slug 'beta'
    beta = cart_id(cid, "beta.txt")

    # The mocked onboard deletes the corpus mid-run (closure over client/headers), simulating a DELETE
    # that lands while this onboard is in flight — then returns the normal success shape.
    def onboard_then_delete(corpus_dir, docs, **kw):
        assert client.delete(f"/corpora/{cid}", headers=headers).status_code == 204
        return {"n_cartridges": len(docs), "canceled": False, "train_seconds": 1.0,
                "n_built": len(docs), "cart_seconds": 1.0, "corpus_tokens": 100}

    monkeypatch.setattr("app.routers.jobs.ml_client.onboard_cag", onboard_then_delete)

    r = client.post(f"/corpora/{cid}/train", headers=headers)
    assert r.status_code == 200
    # TestClient runs the BackgroundTask synchronously. Two cleanup passes fire over the beta cart:
    # first the mid-run DELETE's own cleanup, then — after onboard resurrected the blob — the worker's
    # compensation. The LAST invalidate-then-offboard pair is the compensation, in the safe order.
    assert mock_ml.offboard_ids == [[beta], [beta]]
    assert mock_ml.invalidate_ids == [[beta], [beta]]
    assert mock_ml.order[-2:] == [("invalidate", [beta]), ("offboard", [beta])]
    # A compensation receipt exists (alongside the normal corpus.delete row from the mid-run DELETE).
    events = {row.event for row in _audit_rows() if row.corpus_id == cid}
    assert "corpus.delete_compensation" in events
    assert "corpus.delete" in events


def test_retrain_invalidates_doc_slugs(client, auth, make_corpus, mock_ml, monkeypatch, cart_id):
    """On the vLLM serve path, a successful (re)train invalidates the onboarded cart ids so a
    force-rebuilt cart under the same id isn't shadowed by a stale warm cache."""
    monkeypatch.setattr(config, "INFERENCE_BACKEND", "vllm")
    # jobs.py reads config.INFERENCE_BACKEND both for the onboard branch and the invalidate guard; the
    # branch calls ml_client.onboard_cag, so stub it to the same success shape as fake_train.
    monkeypatch.setattr(
        "app.routers.jobs.ml_client.onboard_cag",
        lambda corpus_dir, docs, **kw: {"n_cartridges": len(docs), "canceled": False,
                                        "train_seconds": 1.0, "n_built": len(docs),
                                        "cart_seconds": 1.0, "corpus_tokens": 100},
    )
    headers, _ = auth
    cid = make_corpus(headers)
    _upload(client, headers, cid, "alpha.txt")  # slug 'alpha'

    r = client.post(f"/corpora/{cid}/train", headers=headers)
    assert r.status_code == 200
    # TestClient runs the BackgroundTask synchronously — training + invalidate have already happened.
    assert mock_ml.invalidate_ids == [[cart_id(cid, "alpha.txt")]]


def test_audit_tenant_isolation_and_limit(client, mock_ml):
    """Tenant B never sees tenant A's audit rows, and ?limit bounds the page."""
    h1, h2 = _register(client), _register(client)
    # Generate three audit rows for tenant A by deleting three corpora.
    for i in range(3):
        cid = client.post("/corpora", json={"name": f"c{i}"}, headers=h1).json()["id"]
        _upload(client, h1, cid, f"d{i}.txt")
        assert client.delete(f"/corpora/{cid}", headers=h1).status_code == 204

    a_rows = client.get("/audit", headers=h1).json()
    assert len(a_rows) == 3
    assert all(r["event"] == "corpus.delete" for r in a_rows)

    # Tenant B has no rows of its own and cannot see A's.
    assert client.get("/audit", headers=h2).json() == []

    # limit caps the page.
    assert len(client.get("/audit?limit=2", headers=h1).json()) == 2
    # Out-of-range limit is rejected by the query validator.
    assert client.get("/audit?limit=0", headers=h1).status_code == 422
    assert client.get("/audit?limit=5000", headers=h1).status_code == 422


def test_gc_requires_token(client, mock_ml, monkeypatch):
    """Unset INTERNAL_API_TOKEN -> 503 (never an unauthenticated store-wide delete); wrong token -> 401."""
    monkeypatch.setattr(config, "INTERNAL_API_TOKEN", "")
    r = client.post("/internal/gc/carts", json={"confirm": False})
    assert r.status_code == 503

    monkeypatch.setattr(config, "INTERNAL_API_TOKEN", "s3cr3t-token")
    r = client.post("/internal/gc/carts", json={"confirm": False},
                    headers={"X-Internal-Token": "wrong"})
    assert r.status_code == 401


def test_gc_dry_run_then_confirm(client, auth, make_corpus, mock_ml, monkeypatch, cart_id):
    """Dry run lists orphans and deletes nothing; confirm=true offboards + invalidates them and writes
    a _system audit row. An orphan = a store cart no live document references. The 'kept' cart is
    stored under its TENANT-NAMESPACED id (what onboarding writes), so GC recognizes it as referenced."""
    monkeypatch.setattr(config, "INTERNAL_API_TOKEN", "s3cr3t-token")
    tok = {"X-Internal-Token": "s3cr3t-token"}

    # One live corpus referencing 'kept.txt'; the fake store holds that (namespaced) cart plus two
    # orphans that no document references.
    headers, _ = auth
    cid = make_corpus(headers)
    _upload(client, headers, cid, "kept.txt")
    kept = cart_id(cid, "kept.txt")
    # Orphans must be LOCALLY-ATTRIBUTABLE (known tenant prefix) to be sweepable — the
    # shared-store rule skips foreign/legacy ids (see test_namespacing.test_gc_is_tenant_safe).
    orphan_a = cart_id(cid, "gone_a.txt")
    orphan_b = cart_id(cid, "gone_b.txt")
    mock_ml.carts = [kept, orphan_a, orphan_b]

    # Dry run: orphans surfaced, nothing deleted.
    r = client.post("/internal/gc/carts", json={"confirm": False}, headers=tok)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["deleted"] is False
    assert body["orphans"] == sorted([orphan_a, orphan_b])
    assert body["n_orphans"] == 2
    assert mock_ml.offboard_ids == [] and mock_ml.invalidate_ids == []

    # Confirm: the orphans (not 'kept') are offboarded + invalidated, and a _system receipt is written.
    r = client.post("/internal/gc/carts", json={"confirm": True}, headers=tok)
    assert r.status_code == 200, r.text
    assert r.json()["deleted"] is True
    assert mock_ml.offboard_ids == [sorted([orphan_a, orphan_b])]
    assert mock_ml.invalidate_ids == [sorted([orphan_a, orphan_b])]
    assert any(row.event == "carts.gc" and row.tenant_id == "_system"
               for row in _audit_rows(tenant_filter="_system"))
