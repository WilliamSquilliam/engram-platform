"""Durable persistence for the measured cart-vs-RAG buffer (BYOC A6).

Covers: record() writes rows to the measurements table; the in-memory buffer warms from the DB after a
simulated process restart (module state cleared); GET /metrics/savings returns correct lifetime
aggregates for a seeded set; and a broken session factory never raises out of record() (the serve path
must survive a DB failure). The GPU service isn't touched — record() takes plain metrics dicts."""
import pytest
from app import measurements
from app.db import SessionLocal
from app.models import Measurement

# One measured head-to-head, shaped like what the Inference Service returns (see vllm_inference.py).
# ttft/decode present so cost_per_query is the length-normalized number, not the latency fallback.
CART = {"latency_ms": 10.0, "ttft_ms": 4.0, "decode_tps": 40.0,
        "prompt_tokens": 8, "resident_kv_tokens": 5000, "gen_tokens": 3, "confidence": -0.1}
RAG = {"latency_ms": 30.0, "ttft_ms": 12.0, "decode_tps": 40.0,
       "prompt_tokens": 500, "gen_tokens": 5, "confidence": None}


@pytest.fixture()
def clean_measurements():
    """Isolate module + table state per test: clear the ring buffer, force a fresh lazy-warm, and
    truncate the measurements table (the DB is shared across the session)."""
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


def _count_rows(side: str | None = None) -> int:
    s = SessionLocal()
    try:
        q = s.query(Measurement)
        if side is not None:
            q = q.filter(Measurement.side == side)
        return q.count()
    finally:
        s.close()


def test_record_persists_rows(clean_measurements):
    """A full head-to-head record() writes one cart row + one rag row with the measured fields."""
    measurements.record(CART, RAG)
    assert _count_rows() == 2
    assert _count_rows("cart") == 1 and _count_rows("rag") == 1

    s = SessionLocal()
    try:
        cart = s.query(Measurement).filter(Measurement.side == "cart").one()
    finally:
        s.close()
    assert cart.latency_ms == 10.0
    assert cart.resident_kv_tokens == 5000
    # cost_per_query is computed at record time (length-normalized) -> a positive number, not None.
    assert cart.cost_per_query is not None and cart.cost_per_query > 0


def test_record_tolerates_missing_side(clean_measurements):
    """cart-only record (the second half of a sequential pair) persists just the cart row."""
    measurements.record(CART, None)
    assert _count_rows("cart") == 1 and _count_rows("rag") == 0


def test_buffer_warms_after_restart(clean_measurements):
    """Persist a record, then simulate a process restart (clear the buffer + re-arm lazy warm). The
    next summary() must reflect the persisted numbers instead of {measured: False}."""
    measurements.record(CART, RAG)

    # Simulate a fresh process: nothing in memory, warm not yet done.
    measurements._RECORDS.clear()
    measurements._WARMED = False
    assert len(measurements._RECORDS) == 0

    out = measurements.summary()
    assert out["measured"] is True
    assert out["cart"]["latency_ms"] == 10.0
    assert out["rag"]["latency_ms"] == 30.0
    # Cart is faster + cheaper than RAG in the seeded numbers.
    assert out["savings"]["faster_than_rag_x"] == 3.0
    assert out["savings"]["cheaper_than_rag_x"] is not None


def test_warm_respects_buffer_capacity(clean_measurements, monkeypatch):
    """Warm loads at most the buffer's capacity from the DB (newest rows), not the whole table."""
    monkeypatch.setattr(measurements, "_BUFFER_MAX", 3)
    for _ in range(5):
        measurements.record(CART, None)   # 5 cart rows persisted
    assert _count_rows("cart") == 5

    measurements._RECORDS.clear()
    measurements._WARMED = False
    measurements.summary()                # triggers warm
    assert len(measurements._RECORDS) == 3


def test_record_survives_db_failure(clean_measurements, monkeypatch):
    """Break the session factory: record() must NOT raise (best-effort persistence) and the in-memory
    buffer must still update so the hot summary keeps working."""
    def boom():
        raise RuntimeError("db is down")

    # Patch where _persist looks it up (it imports from app.db at call time).
    import app.db as appdb
    monkeypatch.setattr(appdb, "SessionLocal", boom)
    measurements._WARMED = True  # skip warm so the failure is isolated to the write path

    measurements.record(CART, RAG)  # must not raise
    # In-memory buffer still advanced even though the DB write failed.
    assert len(measurements._RECORDS) == 1
    out = measurements.summary()
    assert out["measured"] is True
    # Nothing was persisted.
    assert _count_rows() == 0


def test_savings_endpoint_aggregates(client, auth, clean_measurements):
    """Seed a known set and assert the /metrics/savings rollup: per-side counts/averages, the
    per-query cost delta, cumulative savings, and a monthly bucket. Authenticated since the
    2026-09 security sweep — deployment-wide economics are not public demo data."""
    hdr, _email = auth
    # 3 head-to-heads with identical numbers -> deterministic averages.
    for _ in range(3):
        measurements.record(CART, RAG)

    assert client.get("/metrics/savings").status_code == 401  # anonymous is refused
    r = client.get("/metrics/savings", headers=hdr)
    assert r.status_code == 200
    data = r.json()

    assert data["totals"]["cart"]["count"] == 3
    assert data["totals"]["rag"]["count"] == 3
    assert data["totals"]["cart"]["avg_latency_ms"] == 10.0
    assert data["totals"]["rag"]["avg_latency_ms"] == 30.0

    cart_q = data["totals"]["cart"]["avg_cost_per_query"]
    rag_q = data["totals"]["rag"]["avg_cost_per_query"]
    assert cart_q is not None and rag_q is not None and rag_q > cart_q

    delta = data["savings"]["per_query_cost_delta"]
    assert delta == pytest.approx(rag_q - cart_q, abs=1e-6)
    # cumulative = per-query delta x cart queries served (3).
    assert data["savings"]["cumulative_savings"] == pytest.approx(delta * 3, abs=1e-3)
    assert data["savings"]["queries_served"] == 6

    # Monthly breakdown: one bucket (all seeded this month), both sides counted.
    assert len(data["monthly"]) == 1
    bucket = data["monthly"][0]
    assert bucket["cart"]["count"] == 3 and bucket["rag"]["count"] == 3


# --- F6: per-tenant metering (measurements carry tenant_id; counts don't cross) --------

def test_record_stamps_tenant_id(clean_measurements):
    """record(..., tenant_id=X) persists the tenant on both sides; the default (no kwarg) is NULL."""
    measurements.record(CART, RAG, tenant_id="tenant-A")
    measurements.record(CART, None)  # no tenant -> NULL (legacy/demo)

    s = SessionLocal()
    try:
        stamped = s.query(Measurement).filter(Measurement.tenant_id == "tenant-A").all()
        null_rows = s.query(Measurement).filter(Measurement.tenant_id.is_(None)).all()
    finally:
        s.close()
    assert len(stamped) == 2 and {m.side for m in stamped} == {"cart", "rag"}
    assert len(null_rows) == 1 and null_rows[0].side == "cart"


def test_tenant_query_count_isolates_and_total_includes_null(clean_measurements):
    """usage.tenant_query_count is each tenant's OWN cart count; total_query_count is the fleet count
    = per-tenant sum + the NULL-tenant remainder."""
    from app import usage

    measurements.record(CART, RAG, tenant_id="A")
    measurements.record(CART, RAG, tenant_id="A")
    measurements.record(CART, RAG, tenant_id="B")
    measurements.record(CART, RAG)  # NULL tenant

    s = SessionLocal()
    try:
        assert usage.tenant_query_count(s, "A") == 2   # A's cart rows only
        assert usage.tenant_query_count(s, "B") == 1   # B's, not A's, not the NULL row
        # Fleet total = 2 (A) + 1 (B) + 1 (NULL) cart rows.
        assert usage.total_query_count(s) == 4
        # Tenant-scoped series counts only that tenant's queries.
        _series, a_total = usage.query_series(s, tenant_id="A")
        assert a_total == 2
    finally:
        s.close()


def test_savings_endpoint_empty(client, auth, clean_measurements):
    """With no recorded measurements the endpoint returns zeroed totals and no savings (not an error)."""
    hdr, _email = auth
    r = client.get("/metrics/savings", headers=hdr)
    assert r.status_code == 200
    data = r.json()
    assert data["totals"]["cart"]["count"] == 0
    assert data["savings"]["cumulative_savings"] is None
    assert data["monthly"] == []
