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


def _valid_prod_config(monkeypatch):
    """Set every prod invariant to a VALID value, so a single subsequent override is the only thing
    that can trip config.validate()."""
    monkeypatch.setattr(config, "IS_PROD", True)
    monkeypatch.setattr(config, "JWT_SECRET", "x" * 40)
    monkeypatch.setattr(config, "SESSION_SECRET", "y" * 40)
    monkeypatch.setattr(config, "DATABASE_URL", "postgresql+psycopg://h/db")
    monkeypatch.setattr(config, "CORS_ORIGINS", ["https://app.example.com"])
    monkeypatch.setattr(config, "INTERNAL_API_TOKEN", "i" * 40)
    monkeypatch.setattr(config, "EMAIL_BACKEND", "ses")
    monkeypatch.setattr(config, "ML_AUTH_TOKEN", "m" * 40)


def _validate_error(monkeypatch) -> str:
    try:
        config.validate()
        return ""
    except RuntimeError as exc:
        return str(exc)


def test_internal_token_required_in_prod(monkeypatch):
    # config.validate() must refuse to boot in prod without a strong internal token.
    _valid_prod_config(monkeypatch)
    monkeypatch.setattr(config, "INTERNAL_API_TOKEN", "")
    assert "INTERNAL_API_TOKEN" in _validate_error(monkeypatch)


def test_email_backend_none_rejected_in_prod(monkeypatch):
    # F2: EMAIL_BACKEND=none in prod would return invite/reset LINKS in API responses -> refuse boot.
    _valid_prod_config(monkeypatch)
    monkeypatch.setattr(config, "EMAIL_BACKEND", "none")
    assert "EMAIL_BACKEND" in _validate_error(monkeypatch)


def test_ml_auth_token_required_in_prod(monkeypatch):
    # F2: an unset (or short) ML_AUTH_TOKEN leaves the ML/vLLM planes unauthenticated -> refuse boot.
    _valid_prod_config(monkeypatch)
    monkeypatch.setattr(config, "ML_AUTH_TOKEN", "")
    assert "ML_AUTH_TOKEN" in _validate_error(monkeypatch)
    monkeypatch.setattr(config, "ML_AUTH_TOKEN", "short")  # under 32 chars is also rejected
    assert "ML_AUTH_TOKEN" in _validate_error(monkeypatch)


def test_valid_prod_config_passes(monkeypatch):
    # The fully-valid prod config must NOT raise (guards against a check that can never be satisfied).
    _valid_prod_config(monkeypatch)
    assert _validate_error(monkeypatch) == ""
