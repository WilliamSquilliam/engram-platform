"""Tests for the web-exposure hardening: registration gate, auth rate limiting,
and the upload size cap. These guard the behaviours added before the app goes on
a network (see docs/PRE_PRODUCTION.md)."""
import uuid

from app import config
from app.ratelimit import limiter


def _reg_body():
    return {
        "email": f"u-{uuid.uuid4().hex[:8]}@test.local",
        "password": "pw123456",
        "tenant_name": "Acme",
    }


def test_registration_gate_blocks_when_disabled(client, monkeypatch):
    # The router binds ALLOW_REGISTRATION at import, so patch it there.
    monkeypatch.setattr("app.routers.auth.ALLOW_REGISTRATION", False)
    r = client.post("/auth/register", json=_reg_body())
    assert r.status_code == 403


def test_registration_allowed_by_default(client):
    r = client.post("/auth/register", json=_reg_body())
    assert r.status_code == 200


def test_auth_rate_limit_trips(client, monkeypatch):
    # Temporarily enable the limiter (the suite disables it globally) and confirm
    # the 11th call within the window is throttled.
    monkeypatch.setattr(limiter, "enabled", True)
    limiter.reset()
    codes = [client.post("/auth/register", json=_reg_body()).status_code for _ in range(12)]
    limiter.reset()
    assert 429 in codes, codes
    assert codes[:10].count(200) == 10  # first 10 allowed, then throttled


def test_upload_rejects_oversize_file(client, auth, make_corpus, monkeypatch):
    headers, _ = auth
    cid = make_corpus(headers, "Big")
    monkeypatch.setattr("app.routers.corpora.MAX_UPLOAD_MB", 0)  # any non-empty file too big
    r = client.post(
        f"/corpora/{cid}/documents",
        files=[("files", ("big.txt", "x" * 1024, "text/plain"))],
        headers=headers,
    )
    assert r.status_code == 413


def test_internal_token_required_in_prod(monkeypatch):
    # config.validate() must refuse to boot in prod without a strong internal token.
    monkeypatch.setattr(config, "IS_PROD", True)
    monkeypatch.setattr(config, "JWT_SECRET", "x" * 40)
    monkeypatch.setattr(config, "SESSION_SECRET", "y" * 40)
    monkeypatch.setattr(config, "DATABASE_URL", "postgresql+psycopg://h/db")
    monkeypatch.setattr(config, "CORS_ORIGINS", ["https://app.example.com"])
    monkeypatch.setattr(config, "INTERNAL_API_TOKEN", "")
    try:
        config.validate()
        raised = False
    except RuntimeError as exc:
        raised = "INTERNAL_API_TOKEN" in str(exc)
    assert raised
