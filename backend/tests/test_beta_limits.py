"""Beta usage limits — invisible until hit, graceful (429) when hit.

Covers:
  document limit at the upload boundary (exact-boundary 429; override wins over the global cap; 0 =
  unlimited); monthly query limit at chat dispatch (this-month scoping — an old-month measurement does
  NOT count); the structured 429 body {"error","limit","message"}; and the platform-admin limits PATCH
  (works for a platform_admin, 403 for a tenant admin).

Caps are made small via monkeypatch on config attributes (the existing per-test pattern), so the real
generous defaults are never relied on.
"""
import datetime
import uuid

from app import config
from app.db import SessionLocal
from app.models import Corpus, Measurement, Tenant, User


def _register(client, tenant_name: str = "Acme") -> tuple[dict, str]:
    email = f"user-{uuid.uuid4().hex[:8]}@test.local"
    r = client.post("/auth/register",
                    json={"email": email, "password": "pw123456", "tenant_name": tenant_name})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}, email


def _tenant_id(email: str) -> str:
    db = SessionLocal()
    try:
        return db.query(User).filter(User.email == email).first().tenant_id
    finally:
        db.close()


def _make_platform_admin(email: str) -> None:
    db = SessionLocal()
    try:
        db.query(User).filter(User.email == email).first().platform_admin = True
        db.commit()
    finally:
        db.close()


def _upload(client, hdr, cid, name):
    return client.post(f"/corpora/{cid}/documents",
                       files=[("files", (name, "hello world", "text/plain"))], headers=hdr)


# --- document limit at upload ---------------------------------------------

def test_doc_limit_429_at_exact_boundary(client, monkeypatch):
    """Cap of 2: two uploads succeed, the third (which would make 3 > 2) is rejected 429 BEFORE saving."""
    monkeypatch.setattr(config, "BETA_MAX_DOCS_PER_TENANT", 2)
    hdr, _ = _register(client)
    cid = client.post("/corpora", json={"name": "KB"}, headers=hdr).json()["id"]

    assert _upload(client, hdr, cid, "a.txt").status_code == 200
    assert _upload(client, hdr, cid, "b.txt").status_code == 200
    r = _upload(client, hdr, cid, "c.txt")
    assert r.status_code == 429
    body = r.json()["detail"]
    assert body == {"error": "beta_limit", "limit": "documents", "message": body["message"]}
    assert "beta limit" in body["message"].lower()
    # Nothing was saved past the cap: still exactly two documents.
    docs = client.get(f"/corpora/{cid}/documents", headers=hdr).json()
    assert len(docs) == 2


def test_doc_limit_override_wins_over_global(client, monkeypatch):
    """A per-tenant max_docs_override beats the (smaller) global cap — the 'contact us to raise it' lever."""
    monkeypatch.setattr(config, "BETA_MAX_DOCS_PER_TENANT", 1)
    hdr, email = _register(client)
    tid = _tenant_id(email)
    db = SessionLocal()
    try:
        db.get(Tenant, tid).max_docs_override = 3
        db.commit()
    finally:
        db.close()
    cid = client.post("/corpora", json={"name": "KB"}, headers=hdr).json()["id"]
    # Global cap is 1, but the override of 3 lets three through.
    for n in ("a.txt", "b.txt", "c.txt"):
        assert _upload(client, hdr, cid, n).status_code == 200
    assert _upload(client, hdr, cid, "d.txt").status_code == 429


def test_doc_limit_zero_is_unlimited(client, monkeypatch):
    """0 means unlimited — the cap is off even far past what a small number would block."""
    monkeypatch.setattr(config, "BETA_MAX_DOCS_PER_TENANT", 0)
    hdr, _ = _register(client)
    cid = client.post("/corpora", json={"name": "KB"}, headers=hdr).json()["id"]
    for n in ("a.txt", "b.txt", "c.txt", "d.txt", "e.txt"):
        assert _upload(client, hdr, cid, n).status_code == 200


# --- monthly query limit at chat ------------------------------------------

def _ready_corpus(tenant_id: str) -> str:
    """Seed a ready corpus with one document directly (chat's limit check runs before the engine, so a
    ready corpus is all that's needed to reach the gate)."""
    db = SessionLocal()
    try:
        c = Corpus(tenant_id=tenant_id, name="KB", status="ready")
        db.add(c)
        db.commit()
        return c.id
    finally:
        db.close()


def _seed_measurement(tenant_id: str, when: datetime.datetime) -> None:
    db = SessionLocal()
    try:
        db.add(Measurement(side="cart", tenant_id=tenant_id, created_at=when))
        db.commit()
    finally:
        db.close()


def test_query_limit_429_this_month_only(client, monkeypatch, mock_ml):
    """Cap of 1/month. A measurement from LAST month must NOT count (this-month scoping), so the first
    query still goes through; a this-month measurement at the cap trips the 429 before the engine."""
    monkeypatch.setattr(config, "BETA_MAX_QUERIES_PER_MONTH", 1)
    hdr, email = _register(client)
    tid = _tenant_id(email)
    cid = _ready_corpus(tid)

    now = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
    last_month = (now.replace(day=1) - datetime.timedelta(days=1))
    _seed_measurement(tid, last_month)  # old month — must not count

    # Under the cap this-month (0 counted), so the chat succeeds (hf path via mock_ml).
    ok = client.post(f"/corpora/{cid}/chat", json={"question": "hi"}, headers=hdr)
    assert ok.status_code == 200

    # Now push this-month usage to the cap and confirm the next call is refused.
    _seed_measurement(tid, now)
    r = client.post(f"/corpora/{cid}/chat", json={"question": "again"}, headers=hdr)
    assert r.status_code == 429
    body = r.json()["detail"]
    assert body["error"] == "beta_limit" and body["limit"] == "queries"
    assert "beta limit" in body["message"].lower()


def test_query_limit_zero_is_unlimited(client, monkeypatch, mock_ml):
    monkeypatch.setattr(config, "BETA_MAX_QUERIES_PER_MONTH", 0)
    hdr, email = _register(client)
    tid = _tenant_id(email)
    cid = _ready_corpus(tid)
    now = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
    for _ in range(5):
        _seed_measurement(tid, now)
    assert client.post(f"/corpora/{cid}/chat", json={"question": "hi"}, headers=hdr).status_code == 200


# --- platform-admin limits PATCH ------------------------------------------

def test_platform_admin_sets_limits(client):
    founder_hdr, founder_email = _register(client, "HQ")
    _make_platform_admin(founder_email)
    _hdr, email = _register(client, "Payer")
    tid = _tenant_id(email)

    r = client.patch(f"/platform-admin/tenants/{tid}/limits",
                     json={"max_docs_override": 42, "max_queries_override": 0}, headers=founder_hdr)
    assert r.status_code == 200
    body = r.json()
    assert body["tenant_id"] == tid
    assert body["max_docs_override"] == 42
    assert body["max_queries_override"] == 0  # 0 persists (unlimited), distinct from null

    db = SessionLocal()
    try:
        t = db.get(Tenant, tid)
        assert t.max_docs_override == 42 and t.max_queries_override == 0
    finally:
        db.close()


def test_limits_patch_forbidden_to_tenant_admin(client):
    _founder_hdr, _ = _register(client, "HQ2")
    tenant_hdr, email = _register(client, "Nosy")  # a tenant admin, NOT a platform admin
    tid = _tenant_id(email)
    r = client.patch(f"/platform-admin/tenants/{tid}/limits",
                     json={"max_docs_override": 99}, headers=tenant_hdr)
    assert r.status_code == 403
