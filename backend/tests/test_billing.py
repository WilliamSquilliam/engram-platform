"""Stripe billing — DARK-LAUNCHED (fully wired, disabled by BILLING_ENABLED).

Covers the dark-launch contract:
  /billing/status         -> always 200; disabled shape; two-meter rate card visible
  /billing/portal         -> 503 while disabled; happy path when enabled (stripe mocked)
  /billing/webhook        -> 503 while disabled; 400 on bad signature when enabled (stripe mocked)
  /internal/billing/report-usage -> no-op when disabled; internal-token 401/503 when enabled

The Stripe SDK is NEVER hit for real: routers/billing.py imports it lazily and these tests monkeypatch
config.BILLING_ENABLED + inject a fake `stripe` module for the enabled paths.
"""
import sys
import types
import uuid

import pytest
from app import config
from app.db import SessionLocal
from app.models import Tenant, User


def _register(client, tenant_name: str = "Biller") -> tuple[dict, str]:
    email = f"user-{uuid.uuid4().hex[:8]}@test.local"
    r = client.post("/auth/register",
                    json={"email": email, "password": "pw123456", "tenant_name": tenant_name})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}, email


def _tenant_id_for(email: str) -> str:
    db = SessionLocal()
    try:
        return db.query(User).filter(User.email == email).first().tenant_id
    finally:
        db.close()


class _FakeCustomer:
    id = "cus_fake123"


class _FakeSession:
    url = "https://billing.stripe.test/session/abc"


def _fake_stripe(*, construct_raises: bool = False, event: dict | None = None):
    """A stand-in `stripe` module with the surface billing.py touches (Customer, billing_portal,
    billing.MeterEvent, Webhook.construct_event). Installed into sys.modules so the router's lazy
    `import stripe` picks it up — the real SDK is never imported or called."""
    stripe = types.ModuleType("stripe")
    stripe.api_key = None

    stripe.Customer = types.SimpleNamespace(create=lambda **kw: _FakeCustomer())
    stripe.billing_portal = types.SimpleNamespace(
        Session=types.SimpleNamespace(create=lambda **kw: _FakeSession())
    )
    meter_calls: list[dict] = []
    stripe.billing = types.SimpleNamespace(
        MeterEvent=types.SimpleNamespace(
            create=lambda **kw: meter_calls.append(kw)
        )
    )
    stripe._meter_calls = meter_calls

    def _construct(payload, sig, secret):
        if construct_raises:
            raise ValueError("bad signature")
        return event or {"type": "customer.updated", "data": {"object": {}}}

    stripe.Webhook = types.SimpleNamespace(construct_event=_construct)
    return stripe


@pytest.fixture()
def enable_billing(monkeypatch):
    """Turn billing on + set the four Stripe values, and install a fake stripe module. Returns the
    fake so a test can assert meter calls."""
    monkeypatch.setattr(config, "BILLING_ENABLED", True)
    monkeypatch.setattr(config, "STRIPE_SECRET_KEY", "sk_test_x")
    monkeypatch.setattr(config, "STRIPE_WEBHOOK_SECRET", "whsec_x")
    monkeypatch.setattr(config, "STRIPE_PRICE_MEMORY_ID", "meter_memory")
    monkeypatch.setattr(config, "STRIPE_PRICE_INFERENCE_ID", "meter_inference")

    def _install(**kw):
        fake = _fake_stripe(**kw)
        monkeypatch.setitem(sys.modules, "stripe", fake)
        return fake

    return _install


# --- /billing/status: always safe -----------------------------------------

def test_status_disabled_shape(client):
    hdr, _ = _register(client)
    r = client.get("/billing/status", headers=hdr)
    assert r.status_code == 200
    b = r.json()
    assert b["enabled"] is False
    assert b["portal_available"] is False
    # Two-meter rate card surfaced even while disabled (pricing visible in one place).
    assert set(b["rate_card"]) >= {"per_1k_queries_usd", "per_doc_month_usd", "per_onboarded_doc_usd"}
    assert b["rate_card"]["per_onboarded_doc_usd"] == 0.0


def test_status_requires_admin(client):
    assert client.get("/billing/status").status_code == 401


# --- /billing/portal: 503 disabled, happy path enabled ---------------------

def test_portal_503_when_disabled(client):
    hdr, _ = _register(client)
    r = client.post("/billing/portal", headers=hdr)
    assert r.status_code == 503
    assert r.json()["detail"] == "Billing is not enabled during the beta."


def test_portal_creates_customer_and_returns_url_when_enabled(client, enable_billing):
    enable_billing()
    hdr, email = _register(client)
    r = client.post("/billing/portal", headers=hdr)
    assert r.status_code == 200
    assert r.json()["url"] == _FakeSession.url
    # The customer id was persisted on the tenant (lazily created on first portal open).
    db = SessionLocal()
    try:
        tid = _tenant_id_for(email)
        assert db.get(Tenant, tid).stripe_customer_id == _FakeCustomer.id
    finally:
        db.close()


# --- /billing/webhook: 503 disabled, 400 bad signature enabled -------------

def test_webhook_503_when_disabled(client):
    r = client.post("/billing/webhook", content=b"{}",
                    headers={"Stripe-Signature": "t=1,v1=abc"})
    assert r.status_code == 503


def test_webhook_400_on_bad_signature_when_enabled(client, enable_billing):
    enable_billing(construct_raises=True)
    r = client.post("/billing/webhook", content=b"{}",
                    headers={"Stripe-Signature": "bad"})
    assert r.status_code == 400


def test_webhook_acks_invoice_paid_when_enabled(client, enable_billing):
    enable_billing(event={"type": "invoice.paid", "data": {"object": {"customer": "cus_x"}}})
    r = client.post("/billing/webhook", content=b"{}",
                    headers={"Stripe-Signature": "ok"})
    assert r.status_code == 200
    assert r.json()["received"] is True


# --- /internal/billing/report-usage ----------------------------------------

def test_report_usage_noop_when_disabled(client):
    """Disabled -> clean 200 no-op (a scheduler can call it harmlessly during the beta). No token needed
    on this path."""
    r = client.post("/internal/billing/report-usage")
    assert r.status_code == 200
    assert r.json() == {"reported": False, "reason": "billing_disabled"}


def test_report_usage_503_without_internal_token_when_enabled(client, enable_billing, monkeypatch):
    enable_billing()
    monkeypatch.setattr(config, "INTERNAL_API_TOKEN", "")
    r = client.post("/internal/billing/report-usage")
    assert r.status_code == 503


def test_report_usage_401_on_bad_internal_token_when_enabled(client, enable_billing, monkeypatch):
    enable_billing()
    monkeypatch.setattr(config, "INTERNAL_API_TOKEN", "s" * 40)
    assert client.post("/internal/billing/report-usage").status_code == 401
    assert client.post("/internal/billing/report-usage",
                       headers={"X-Internal-Token": "wrong"}).status_code == 401


def test_report_usage_pushes_meter_events_when_enabled(client, enable_billing, monkeypatch):
    """With a valid token + a tenant that has a stripe_customer_id, the reporter pushes meter events and
    advances the query high-water mark."""
    fake = enable_billing()
    monkeypatch.setattr(config, "INTERNAL_API_TOKEN", "s" * 40)
    _hdr, email = _register(client)
    tid = _tenant_id_for(email)
    db = SessionLocal()
    try:
        db.get(Tenant, tid).stripe_customer_id = "cus_payer"
        db.commit()
    finally:
        db.close()

    r = client.post("/internal/billing/report-usage", headers={"X-Internal-Token": "s" * 40})
    assert r.status_code == 200
    assert r.json()["reported"] is True
    # At least the memory meter fired for the tenant with a customer id.
    names = {c["event_name"] for c in fake._meter_calls}
    assert config.STRIPE_PRICE_MEMORY_ID in names
