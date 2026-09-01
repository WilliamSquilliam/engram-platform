"""E10 tenant Admin Dashboard + E11 platform-admin console (usage + billing).

Covers the shared frontend contract:
  /auth/me         -> now carries platform_admin (+ role)
  /admin/usage     -> tenant-scoped rollup; tenant A can't see tenant B's corpora/docs/storage
  /admin/billing   -> plan + limits + usage + estimated_cost from the pricing rate card
  /admin/*         -> 403 for a plain member
  /platform-admin/tenants, /usage -> 403 for a non-platform-admin; 200 (all tenants) for one
  /platform-admin/usage totals    -> sum of the per-tenant rows

Corpora/documents/storage are seeded directly in the DB (deterministic), and served-query counts
come from measurements.record (the same signal the serving path writes).
"""
import uuid

import pytest
from app import measurements
from app.db import SessionLocal
from app.models import Corpus, Document, Measurement, Tenant, User

# A measured head-to-head (mirrors test_measurements) so /admin/usage has a query signal to count.
CART = {"latency_ms": 10.0, "ttft_ms": 4.0, "decode_tps": 40.0, "prompt_tokens": 8,
        "resident_kv_tokens": 5000, "gen_tokens": 3, "confidence": -0.1}
RAG = {"latency_ms": 30.0, "ttft_ms": 12.0, "decode_tps": 40.0, "prompt_tokens": 500,
       "gen_tokens": 5, "confidence": None}


def _email() -> str:
    return f"user-{uuid.uuid4().hex[:8]}@test.local"


def _register(client, tenant_name: str) -> tuple[dict, str, str]:
    """Register a fresh tenant admin; return (auth_headers, email, tenant_id)."""
    email = _email()
    r = client.post("/auth/register",
                    json={"email": email, "password": "pw123456", "tenant_name": tenant_name})
    assert r.status_code == 200, r.text
    hdr = {"Authorization": f"Bearer {r.json()['access_token']}"}
    tenant_id = client.get("/auth/me", headers=hdr).json()["tenant_id"]
    return hdr, email, tenant_id


def _make_platform_admin(email: str) -> None:
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.email == email).first()
        u.platform_admin = True
        db.commit()
    finally:
        db.close()


def _seed_corpus(tenant_id: str, name: str, doc_sizes: list[int], train_seconds: float = 0.0) -> str:
    """Create a corpus with documents of the given byte sizes (drives storage_gb) and an optional
    train_seconds (drives gpu_seconds). Returns the corpus id."""
    db = SessionLocal()
    try:
        c = Corpus(tenant_id=tenant_id, name=name, train_seconds=train_seconds or None)
        db.add(c)
        db.flush()
        for i, size in enumerate(doc_sizes):
            db.add(Document(corpus_id=c.id, filename=f"doc_{i}.txt",
                            storage_key=f"k/{c.id}/{i}", size=size))
        db.commit()
        return c.id
    finally:
        db.close()


@pytest.fixture()
def clean_measurements():
    """Reset the measured ring buffer + table so query counts are deterministic per test."""
    def _reset():
        measurements._RECORDS.clear()
        measurements._WARMED = False
        s = SessionLocal()
        try:
            s.query(Measurement).delete()
            s.commit()
        finally:
            s.close()
    _reset()
    yield
    _reset()


# --- /auth/me now carries platform_admin ---------------------------------

def test_me_reports_platform_admin(client):
    hdr, email, _ = _register(client, "Acme")
    me = client.get("/auth/me", headers=hdr).json()
    assert me["role"] == "admin"
    assert me["platform_admin"] is False  # a plain tenant admin is NOT a platform admin

    _make_platform_admin(email)
    assert client.get("/auth/me", headers=hdr).json()["platform_admin"] is True


# --- E10 /admin/usage: tenant-scoped aggregation + isolation --------------

def test_admin_usage_aggregates_and_isolates(client, clean_measurements):
    a_hdr, _, a_tid = _register(client, "TenantA")
    b_hdr, _, b_tid = _register(client, "TenantB")

    # A: two corpora, 3 docs totalling 3 GiB, 12 gpu-seconds of onboarding.
    gib = 1024 ** 3
    _seed_corpus(a_tid, "A-KB1", [gib, gib], train_seconds=8.0)
    _seed_corpus(a_tid, "A-KB2", [gib], train_seconds=4.0)
    # B: one corpus, 1 doc of 5 GiB — must never appear in A's usage.
    _seed_corpus(b_tid, "B-KB", [5 * gib], train_seconds=2.0)

    # Per-tenant served-query signal (F6): each query is attributed to the tenant whose corpus it ran
    # against. Two for A, one for B, plus one NULL-tenant (legacy/demo) row that belongs to NEITHER.
    measurements.record(CART, RAG, tenant_id=a_tid)
    measurements.record(CART, RAG, tenant_id=a_tid)
    measurements.record(CART, RAG, tenant_id=b_tid)
    measurements.record(CART, RAG)  # NULL tenant_id — must not count toward any tenant

    a = client.get("/admin/usage", headers=a_hdr)
    assert a.status_code == 200
    ua = a.json()
    assert ua["n_corpora"] == 2
    assert ua["documents"] == 3
    assert ua["storage_gb"] == pytest.approx(3.0, abs=1e-6)
    assert ua["gpu_seconds"] == pytest.approx(12.0, abs=1e-6)
    # ~30-day daily series that sums to the windowed query total.
    assert len(ua["series"]) == 30
    assert sum(p["queries"] for p in ua["series"]) == ua["queries"]
    # Only A's OWN two queries — B's query and the NULL-tenant row never leak into A's count.
    assert ua["queries"] == 2
    # by_corpus is A's corpora only.
    names = {c["name"] for c in ua["by_corpus"]}
    assert names == {"A-KB1", "A-KB2"}

    # ISOLATION: B's 5 GiB / 1 doc corpus is invisible to A, and vice versa. B's query count is its
    # own one query (not A's two, not the NULL row).
    ub = client.get("/admin/usage", headers=b_hdr).json()
    assert ub["n_corpora"] == 1
    assert ub["documents"] == 1
    assert ub["storage_gb"] == pytest.approx(5.0, abs=1e-6)
    assert ub["queries"] == 1
    assert "B-KB" not in names
    assert all(c["name"].startswith("A-") for c in ua["by_corpus"])
    assert all(c["name"].startswith("B-") for c in ub["by_corpus"])


def test_admin_usage_empty_is_zeros_not_error(client, clean_measurements):
    hdr, _, _ = _register(client, "Empty")
    r = client.get("/admin/usage", headers=hdr)
    assert r.status_code == 200
    u = r.json()
    assert u["queries"] == 0 and u["documents"] == 0 and u["storage_gb"] == 0.0
    assert u["n_corpora"] == 0 and u["by_corpus"] == []
    assert len(u["series"]) == 30 and sum(p["queries"] for p in u["series"]) == 0


# --- E10 /admin/billing: the shell ---------------------------------------

def test_admin_billing_shell(client, clean_measurements):
    hdr, _, tid = _register(client, "Biller")
    gib = 1024 ** 3
    _seed_corpus(tid, "KB", [2 * gib, 2 * gib])  # 4 GiB, 2 docs

    r = client.get("/admin/billing", headers=hdr)
    assert r.status_code == 200
    b = r.json()
    assert b["plan"] == "beta"            # default plan for a new tenant
    assert b["currency"] == "usd"
    assert "storage_gb" in b["limits"]    # limits dict present (beta = uncapped Nones)
    assert b["usage"]["documents"] == 2
    assert b["usage"]["storage_gb"] == pytest.approx(4.0, abs=1e-6)
    # Estimated cost comes from the pricing rate card and is non-negative.
    assert b["estimated_cost_usd"] >= 0.0
    assert set(b["rate_card"]) >= {"per_1k_queries_usd", "per_gb_month_usd", "per_onboarded_doc_usd"}


def test_admin_billing_cost_matches_pricing_util(client, clean_measurements):
    """The billing estimate must equal pricing.estimate_cost_usd on the same aggregates (one SSOT)."""
    from app import pricing

    hdr, _, tid = _register(client, "CostCheck")
    gib = 1024 ** 3
    _seed_corpus(tid, "KB", [3 * gib], train_seconds=5.0)  # 3 GiB, 1 doc

    b = client.get("/admin/billing", headers=hdr).json()
    expected = pricing.estimate_cost_usd(
        queries=b["usage"]["queries"],
        storage_gb=b["usage"]["storage_gb"],
        documents=b["usage"]["documents"],
    )
    assert b["estimated_cost_usd"] == pytest.approx(expected, abs=1e-6)


# --- authz gates ----------------------------------------------------------

def test_member_forbidden_from_admin_dashboards(client):
    owner_hdr, _, _ = _register(client, "Acme")
    mate = _email()
    token = client.post("/admin/invites", json={"email": mate, "role": "member"},
                        headers=owner_hdr).json()["invite_link"].split("token=")[1]
    mate_tok = client.post("/auth/accept-invite",
                           json={"token": token, "password": "matepass1", "name": "Dash Mate"}).json()["access_token"]
    mate_hdr = {"Authorization": f"Bearer {mate_tok}"}

    assert client.get("/admin/usage", headers=mate_hdr).status_code == 403
    assert client.get("/admin/billing", headers=mate_hdr).status_code == 403


def test_admin_dashboards_require_auth(client):
    assert client.get("/admin/usage").status_code == 401
    assert client.get("/admin/billing").status_code == 401
    assert client.get("/platform-admin/tenants").status_code == 401
    assert client.get("/platform-admin/usage").status_code == 401


# --- E11 platform-admin console: cross-tenant, gated hard -----------------

def test_platform_admin_endpoints_forbidden_to_non_platform_admin(client):
    hdr, _, _ = _register(client, "Acme")  # a tenant admin, but NOT platform_admin
    assert client.get("/platform-admin/tenants", headers=hdr).status_code == 403
    assert client.get("/platform-admin/usage", headers=hdr).status_code == 403


def test_platform_admin_sees_all_tenants(client, clean_measurements):
    a_hdr, _, a_tid = _register(client, "FleetA")
    _b_hdr, _, b_tid = _register(client, "FleetB")
    founder_hdr, founder_email, _ = _register(client, "HQ")
    _make_platform_admin(founder_email)

    gib = 1024 ** 3
    _seed_corpus(a_tid, "A-KB", [gib, gib], train_seconds=6.0)   # 2 GiB, 2 docs
    _seed_corpus(b_tid, "B-KB", [3 * gib], train_seconds=4.0)    # 3 GiB, 1 doc

    # Per-tenant served queries (F6): A gets 3, B gets 1, plus 2 NULL-tenant (legacy/demo) rows that
    # belong to no tenant. The fleet queries total must be per-tenant sum (4) PLUS the NULL remainder (2).
    for _ in range(3):
        measurements.record(CART, RAG, tenant_id=a_tid)
    measurements.record(CART, RAG, tenant_id=b_tid)
    measurements.record(CART, RAG)
    measurements.record(CART, RAG)

    # /tenants: every tenant, with counts + plan/status.
    r = client.get("/platform-admin/tenants", headers=founder_hdr)
    assert r.status_code == 200
    tenants = {t["name"]: t for t in r.json()}
    assert {"FleetA", "FleetB", "HQ"} <= set(tenants)
    assert tenants["FleetA"]["n_corpora"] == 1 and tenants["FleetA"]["n_users"] == 1
    assert tenants["FleetA"]["plan"] == "beta" and tenants["FleetA"]["status"] == "active"

    # /usage: per-tenant rows + fleet totals; totals == sum of the per-tenant rows.
    u = client.get("/platform-admin/usage", headers=founder_hdr)
    assert u.status_code == 200
    data = u.json()
    rows = {t["name"]: t for t in data["tenants"]}
    assert rows["FleetA"]["storage_gb"] == pytest.approx(2.0, abs=1e-6)
    assert rows["FleetA"]["documents"] == 2
    assert rows["FleetB"]["storage_gb"] == pytest.approx(3.0, abs=1e-6)
    assert rows["FleetA"]["est_cost_usd"] >= 0.0
    # Per-tenant query rows carry each tenant's OWN count (F6) — no more queries=0 workaround.
    assert rows["FleetA"]["queries"] == 3
    assert rows["FleetB"]["queries"] == 1

    totals = data["totals"]
    assert totals["n_tenants"] == len(data["tenants"])
    # Storage / gpu / cost totals tie out to the exact sum of the per-tenant line items.
    assert totals["storage_gb"] == pytest.approx(sum(t["storage_gb"] for t in data["tenants"]), abs=1e-6)
    assert totals["gpu_seconds"] == pytest.approx(sum(t["gpu_seconds"] for t in data["tenants"]), abs=1e-6)
    assert totals["est_cost_usd"] == pytest.approx(
        round(sum(t["est_cost_usd"] for t in data["tenants"]), 2), abs=1e-6)
    # Fleet queries = sum of per-tenant counts (4) PLUS the NULL-tenant remainder (2 legacy/demo rows).
    assert totals["queries"] == sum(t["queries"] for t in data["tenants"]) + 2


def test_platform_usage_cost_matches_tenant_billing(client, clean_measurements):
    """A tenant's per-tenant cost line in /platform-admin/usage matches what pricing.estimate_cost_usd
    yields on that tenant's storage+documents (the shared util both views call)."""
    from app import pricing

    founder_hdr, founder_email, _ = _register(client, "HQ2")
    _make_platform_admin(founder_email)
    _hdr, _, tid = _register(client, "Payer")
    gib = 1024 ** 3
    _seed_corpus(tid, "KB", [4 * gib, gib])  # 5 GiB, 2 docs

    data = client.get("/platform-admin/usage", headers=founder_hdr).json()
    row = next(t for t in data["tenants"] if t["name"] == "Payer")
    expected = pricing.estimate_cost_usd(
        queries=0, storage_gb=row["storage_gb"], documents=row["documents"])
    assert row["est_cost_usd"] == pytest.approx(expected, abs=1e-6)
