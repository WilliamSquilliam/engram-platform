"""Test harness: a throwaway SQLite DB + temp storage, the GPU ML service mocked.
Env is configured BEFORE importing the app (config.py reads env at import time)."""
import os
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

_TMP = tempfile.mkdtemp(prefix="cartridge-test-")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_TMP, 'test.db').as_posix()}"
os.environ["PLATFORM_DATA_DIR"] = _TMP
os.environ["PLATFORM_STORAGE_DIR"] = str(Path(_TMP, "storage"))
os.environ["JWT_SECRET"] = "test-secret"
# Force local-only auth (don't pick up a developer's real .env Google creds).
os.environ["GOOGLE_CLIENT_ID"] = ""
os.environ["GOOGLE_CLIENT_SECRET"] = ""
# Source connectors OFF by default: pin the OAuth creds empty so a developer's .env can't flip a
# connector on mid-suite (the connector tests set creds per-test via monkeypatch + oauth.register_all).
# CONNECTOR_ENC_KEY is pinned to a FIXED valid Fernet key so token encrypt/decrypt round-trips
# deterministically in tests (a real key is required — the crypto layer refuses to store plaintext).
os.environ["GDRIVE_CLIENT_ID"] = ""
os.environ["GDRIVE_CLIENT_SECRET"] = ""
os.environ["SHAREPOINT_CLIENT_ID"] = ""
os.environ["SHAREPOINT_CLIENT_SECRET"] = ""
os.environ["SHAREPOINT_TENANT_ID"] = ""
os.environ["CONNECTOR_ENC_KEY"] = "hSY0m6cX3n8b1cS8yQnZ0mDqk5b3Kd1n7oFwq2vJ0Xg="
# Hermetic email: a developer's .env may set EMAIL_BACKEND=ses; tests assume the
# link-in-response (none) behavior and must never attempt real sends.
os.environ["EMAIL_BACKEND"] = "none"
# GPU controls default OFF in tests: a developer's repo-root .env may set real LAMBDA_API_KEY /
# CLOUDFLARE_* creds; pin them empty so the suite never touches Lambda Cloud or Cloudflare and
# GPU_CONTROL_ENABLED starts false (the gpu_admin tests set these per-test via monkeypatch).
os.environ["LAMBDA_API_KEY"] = ""
os.environ["CLOUDFLARE_API_TOKEN"] = ""
os.environ["CLOUDFLARE_API_KEY"] = ""
os.environ["CLOUDFLARE_ZONE_ID"] = ""
# Billing is DARK-LAUNCHED (disabled): pin BILLING_ENABLED off + the Stripe secrets empty so a
# developer's .env can't flip billing on mid-suite or make the SDK reach the network. Tests that
# exercise the enabled path monkeypatch config.BILLING_ENABLED (and the stripe module) per-test.
os.environ["BILLING_ENABLED"] = ""
os.environ["STRIPE_SECRET_KEY"] = ""
os.environ["STRIPE_WEBHOOK_SECRET"] = ""
os.environ["STRIPE_PRICE_MEMORY_ID"] = ""
os.environ["STRIPE_PRICE_INFERENCE_ID"] = ""
# BETA_MAX_* are left at their config defaults (generous); the limit tests monkeypatch config attrs
# to small values, following the existing per-test monkeypatch pattern.
# The suite makes many register/login calls from one client IP — turn off the
# auth rate limiter so it doesn't trip mid-suite.
os.environ["RATELIMIT_ENABLED"] = "false"
# Hybrid retriever: run lexical-only in tests so fastembed never downloads its ONNX model (hermetic,
# fast). The dense stage is exercised separately/behind the same RRF fusion the lexical path uses.
os.environ["RETRIEVAL_DENSE"] = "off"

# Make the `app` package importable (platform/backend on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import ml_client, serving  # noqa: E402
from app.main import app  # noqa: E402
from app.ratelimit import limiter  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

# Belt-and-suspenders: force the auth rate limiter off for the whole suite so its
# many register/login calls from one client IP never trip the per-IP limit.
limiter.enabled = False


@pytest.fixture(autouse=True)
def _reset_connector_crypto():
    """The connector token-crypto layer caches its Fernet (keyed by CONNECTOR_ENC_KEY). A test that
    rotates the key must not leak the cached one into the next test — clear the cache around every
    test so each starts from the pinned conftest key."""
    from app.connectors import crypto

    crypto.reset()
    yield
    crypto.reset()


@pytest.fixture(autouse=True)
def serving_up_true(monkeypatch):
    """Pin serving.serving_up -> True for every test (mirrors the env-leak pinning above): the real
    probe would hit ML_SERVICE_URL/health, so without this the onboard tests would depend on a live
    box. Tests covering the serving-offline path override this back to False."""
    monkeypatch.setattr(serving, "serving_up", lambda: True)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def auth(client: TestClient):
    """Register a fresh tenant; return (auth_headers, email)."""
    email = f"user-{uuid.uuid4().hex[:8]}@test.local"
    r = client.post(
        "/auth/register",
        json={"email": email, "password": "pw123456", "tenant_name": "Acme"},
    )
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}, email


class _MLCalls:
    """Records the lifecycle ML-plane calls the control plane makes (offboard / invalidate /
    list_carts), so a test can assert exactly which cart ids were offboarded/invalidated. `carts` is
    the fake durable store the sweep lists — a test seeds it to simulate orphaned blobs."""

    def __init__(self):
        self.offboard_ids: list[list[str]] = []
        self.invalidate_ids: list[list[str]] = []
        # Interleaved call log ("invalidate"/"offboard", ids) in ACTUAL call order, so a test can assert
        # invalidate is published BEFORE the durable offboard (the tombstone-first ordering contract).
        self.order: list[tuple[str, list[str]]] = []
        self.list_carts_called = 0
        self.carts: list[str] = []  # what list_carts() returns (the fake store's contents)


@pytest.fixture()
def mock_ml(monkeypatch):
    """Deterministic stand-ins for the GPU ML service (no torch/CUDA in tests). The `.calls` attribute
    on the returned function records the lifecycle calls (see _MLCalls)."""
    calls = _MLCalls()

    def fake_train(corpus_dir, docs, **kw):
        return {
            "n_cartridges": len(docs),
            "canceled": False,
            "train_seconds": 1.0,
            "corpus_tokens": 100 * max(1, len(docs)),
        }

    def fake_query(corpus_dir, question, k=3):
        return {"answer": "stub answer", "used_docs": ["doc_0"]}

    def fake_compare(corpus_dir, question, k=3):
        return {
            "results": {
                # everyday = the adaptive router; here it stayed on the cartridge alone (0 raw tok).
                "everyday": {"answer": "e", "latency_ms": 10.0, "prompt_tokens": 50,
                             "raw_tokens": 0, "cart_tokens": 50, "gen_tokens": 5, "feasible": True,
                             "tier": "cartridge", "confidence": -0.2, "theta": -0.7, "used_docs": ["d"]},
                "rag": {"answer": "r", "latency_ms": 20.0, "prompt_tokens": 500,
                        "gen_tokens": 5, "feasible": True, "used_docs": ["d"]},
            },
            "k": k,
            "corpus_tokens": 9000,
        }

    def fake_offboard(doc_ids):
        calls.offboard_ids.append(list(doc_ids))
        calls.order.append(("offboard", list(doc_ids)))
        return {"deleted": list(doc_ids), "missing": []}

    def fake_invalidate(cart_ids):
        calls.invalidate_ids.append(list(cart_ids))
        calls.order.append(("invalidate", list(cart_ids)))
        return {"invalidated": len(cart_ids), "backend": "memory"}

    def fake_list_carts():
        calls.list_carts_called += 1
        return {"cart_ids": list(calls.carts)}

    monkeypatch.setattr(ml_client, "train", fake_train)
    monkeypatch.setattr(ml_client, "query", fake_query)
    monkeypatch.setattr(ml_client, "compare", fake_compare)
    monkeypatch.setattr(ml_client, "offboard", fake_offboard)
    monkeypatch.setattr(ml_client, "inference_invalidate", fake_invalidate)
    monkeypatch.setattr(ml_client, "list_carts", fake_list_carts)
    # The control plane imports these functions via the `ml_client` module namespace (module.attr),
    # so patching the module attributes above covers routers/corpora.py and routers/jobs.py both.
    # Existing tests take mock_ml as a bare fixture arg (ignore the return); lifecycle tests read it.
    return calls


@pytest.fixture()
def make_corpus(client: TestClient):
    def _make(headers: dict, name: str = "KB") -> str:
        r = client.post("/corpora", json={"name": name}, headers=headers)
        assert r.status_code == 200, r.text
        return r.json()["id"]

    return _make


@pytest.fixture()
def upload_doc(client: TestClient):
    def _upload(headers: dict, corpus_id: str, text: str = "hello world") -> None:
        r = client.post(
            f"/corpora/{corpus_id}/documents",
            files=[("files", ("a.txt", text, "text/plain"))],
            headers=headers,
        )
        assert r.status_code == 200, r.text

    return _upload


@pytest.fixture()
def cart_id():
    """The TENANT-NAMESPACED cart id a corpus's document resolves to — the id retrieve()/onboard now
    emit after per-tenant namespacing (E6). Tests that used to hardcode the bare slug (e.g. 'a' for
    a.txt) ask for it here so they assert the SAME derivation the app uses without knowing the tenant
    uuid. `filename` defaults to a.txt (what upload_doc / the make_corpus helpers upload)."""
    from app.retrieval import _tenant_for_corpus, cart_id_for

    def _cart_id(corpus_id: str, filename: str = "a.txt") -> str:
        return cart_id_for(_tenant_for_corpus(corpus_id), filename)

    return _cart_id
