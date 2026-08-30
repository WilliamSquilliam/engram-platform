"""ML-plane shared-token auth (DEFAULT OFF).

Two halves:
  * SERVICE side — a FastAPI TestClient on platform/ml_service/app.py (imported CPU-only; torch is
    already installed in the ML env the suite runs in). No ML_AUTH_TOKEN -> routes are open exactly as
    today; token set -> 401 without a correct Bearer header, allowed with it, and /health stays open
    unconditionally.
  * BACKEND side — ml_client attaches `Authorization: Bearer <token>` to every ML-plane request when
    config.ML_AUTH_TOKEN is set, and attaches nothing when it is empty (today's behavior). We capture
    the outgoing request by stubbing httpx inside the client module.

Run:  python -m pytest platform/backend/tests/test_ml_auth.py -q
"""
import sys
from pathlib import Path

import pytest

# app package importable (mirrors conftest); this test does not need the DB fixtures.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, ml_client  # noqa: E402


# --------------------------------------------------------------------------- backend ml_client side
class _CapturedResp:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_ml_client_attaches_bearer_when_configured(monkeypatch):
    """With config.ML_AUTH_TOKEN set, every ml_client call carries the Bearer header."""
    monkeypatch.setattr(config, "ML_AUTH_TOKEN", "s3cr3t-token")
    captured = {}

    def fake_post(url, **kw):
        captured["headers"] = kw.get("headers") or {}
        return _CapturedResp({"answer": "ok", "used_docs": [], "doc_ids": []})

    monkeypatch.setattr(ml_client.httpx, "post", fake_post)
    ml_client.query("/tmp/corpus", "q?", k=3)
    assert captured["headers"].get("Authorization") == "Bearer s3cr3t-token"

    ml_client.inference_query(["d1"], "q?")
    assert captured["headers"].get("Authorization") == "Bearer s3cr3t-token"


def test_ml_client_no_header_when_unset(monkeypatch):
    """Empty token = today's behavior: no Authorization header attached at all."""
    monkeypatch.setattr(config, "ML_AUTH_TOKEN", "")
    captured = {}

    def fake_post(url, **kw):
        captured["headers"] = kw.get("headers") or {}
        return _CapturedResp({"answer": "ok", "used_docs": []})

    monkeypatch.setattr(ml_client.httpx, "post", fake_post)
    ml_client.query("/tmp/corpus", "q?", k=3)
    assert "Authorization" not in captured["headers"]


def test_ml_headers_merges_extra(monkeypatch):
    """_ml_headers merges the token with any per-call extra headers (used by progress callbacks)."""
    monkeypatch.setattr(config, "ML_AUTH_TOKEN", "tok")
    h = ml_client._ml_headers({"X-Internal-Token": "abc"})
    assert h == {"X-Internal-Token": "abc", "Authorization": "Bearer tok"}


# ------------------------------------------------------------------------------- ML service side
@pytest.fixture()
def ml_app_client(monkeypatch):
    """TestClient over platform/ml_service/app.py. Loaded under a UNIQUE module name via importlib
    (a plain `import app` would resolve to the BACKEND app package that conftest already put on
    sys.path). Imported lazily so a box without torch skips these SERVICE-side tests rather than
    erroring, while the BACKEND-side tests above still run."""
    import importlib.util
    ml_app_path = Path(__file__).resolve().parents[2] / "ml_service" / "app.py"
    sys.path.insert(0, str(ml_app_path.parent))     # so `import cartridges` etc. resolve
    spec = importlib.util.spec_from_file_location("ml_service_app_under_test", ml_app_path)
    ml_app = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(ml_app)
    except Exception as e:  # noqa: BLE001 — torch/deps missing on this box
        pytest.skip(f"ml_service app import unavailable: {e}")
    from starlette.testclient import TestClient
    return ml_app, TestClient(ml_app.app)


def test_service_open_when_token_unset(ml_app_client, monkeypatch):
    """No ML_AUTH_TOKEN -> health AND a normal route are reachable with no header (today's behavior).
    We probe /health (cheap, no GPU) and assert a protected route is NOT rejected for auth reasons."""
    ml_app, client = ml_app_client
    monkeypatch.delenv("ML_AUTH_TOKEN", raising=False)
    assert client.get("/health").status_code == 200
    # A route that would need auth if the token were set: with no token it must not 401.
    r = client.post("/retrieve", json={"corpus_dir": "/nope", "question": "q"})
    assert r.status_code != 401


def test_service_401_without_header_when_token_set(ml_app_client, monkeypatch):
    """Token set -> a protected route without a Bearer header is 401; /health stays open."""
    ml_app, client = ml_app_client
    monkeypatch.setenv("ML_AUTH_TOKEN", "svc-token")
    assert client.get("/health").status_code == 200          # liveness never gated
    r = client.post("/retrieve", json={"corpus_dir": "/nope", "question": "q"})
    assert r.status_code == 401


def test_service_401_with_wrong_header(ml_app_client, monkeypatch):
    ml_app, client = ml_app_client
    monkeypatch.setenv("ML_AUTH_TOKEN", "svc-token")
    r = client.post("/retrieve", json={"corpus_dir": "/nope", "question": "q"},
                    headers={"Authorization": "Bearer WRONG"})
    assert r.status_code == 401


def test_service_passes_with_correct_bearer(ml_app_client, monkeypatch):
    """Correct Bearer -> the auth gate lets it through (the handler then fails on the bogus corpus_dir,
    NOT on auth) — a non-401 status proves auth passed on a cheap route."""
    ml_app, client = ml_app_client
    monkeypatch.setenv("ML_AUTH_TOKEN", "svc-token")
    r = client.post("/retrieve", json={"corpus_dir": "/nope", "question": "q"},
                    headers={"Authorization": "Bearer svc-token"})
    assert r.status_code != 401
