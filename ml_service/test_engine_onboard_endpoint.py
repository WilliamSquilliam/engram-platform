"""POST /onboard_cag on the vLLM serving frontend (vllm_inference.py) — the engine-side cart-build
path. GPU-FREE: vllm is imported lazily inside the module, and the wheel's engine_onboard/cartridge
imports live inside the persist helper, so this stubs all three (a fake vllm module for
SamplingParams/TokensPrompt, a fake cartridges.serve.engine_onboard for collect_rank_shards/
cart_from_kv, a fake cartridges.cartridge for the rope helpers) and seeds a fake async engine +
store. What it pins:
  * cart_id derivation == the doc's tenant-namespaced doc_id (the store key app.py uses),
  * every build submission carries extra_args={"cartridge_build_cart_id": <cart_id>} + max_tokens=1,
  * one doc's failure is isolated into response["errors"] and never sinks the batch,
  * the response schema matches app.py's onboard_cag_corpus (n_cartridges/canceled/method/
    train_seconds/corpus_tokens + the cag extras n_built/cart_seconds),
  * idempotent reuse: a cart already in the store is skipped (no engine submission),
  * engine-not-ready surfaces as 503 through the route.

Follows the sys.modules-stub style of test_sampling_helper.py (reload vllm_inference against stubs).
"""
from __future__ import annotations

import asyncio
import importlib
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


# --- stubs -----------------------------------------------------------------------------------
class _SamplingParams:
    def __init__(self, temperature=1.0, max_tokens=16, extra_args=None):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_args = extra_args


class _TokensPrompt:
    def __init__(self, prompt_token_ids=None, cache_salt=None):
        self.prompt_token_ids = prompt_token_ids
        self.cache_salt = cache_salt


class _FakeTok:
    """Word-count tokenizer: ids are 1 per whitespace token, enough to exercise truncation + counts."""
    vocab_size = 1000

    def __call__(self, text, add_special_tokens=False):
        n = len(text.split())
        return types.SimpleNamespace(input_ids=list(range(1, n + 1)))


class _FakeEngine:
    """Records every (prompt, SamplingParams, rid) submitted, and yields one RequestOutput so the
    async `generate` loop in _engine_build_kv completes."""
    def __init__(self):
        self.calls = []

    async def generate(self, prompt, sampling, rid):
        self.calls.append((prompt, sampling, rid))
        out = types.SimpleNamespace(request_id=rid,
                                    outputs=[types.SimpleNamespace(token_ids=[0], text="")])
        yield out


class _FakeStore:
    def __init__(self, existing=None):
        self._existing = set(existing or [])
        self.puts = []          # (cart_id, store_dtype)

    def exists(self, cart_id):
        return cart_id in self._existing

    def put(self, cart_id, cart, store_dtype="bf16"):
        self.puts.append((cart_id, store_dtype))
        self._existing.add(cart_id)


class _FakeCart:
    def __init__(self, doc_id, token_ids, model_ref, rope_theta, rope_scaling):
        self.doc_id = doc_id
        self.token_ids = token_ids
        self.model_ref = model_ref
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

    def num_kv_tokens(self):
        return len(self.token_ids)


def _install_wheel_stubs(monkeypatch, *, fail_ids=()):
    """Stub cartridges.serve.engine_onboard (collect_rank_shards/cart_from_kv) and
    cartridges.cartridge (rope helpers). collect_rank_shards raises for any id in `fail_ids` so a
    per-doc failure can be exercised without a GPU."""
    collected = []

    def collect_rank_shards(cart_id, registry_dir, tp_size, timeout_s=120.0):
        collected.append((cart_id, registry_dir, tp_size))
        if cart_id in fail_ids:
            raise TimeoutError(f"no shards for {cart_id}")
        return object()  # opaque kv; cart_from_kv below ignores its contents

    def cart_from_kv(kv, doc_id, *, token_ids, model_ref, rope_theta=None, rope_scaling=None):
        return _FakeCart(doc_id, token_ids, model_ref, rope_theta, rope_scaling)

    eo = types.ModuleType("cartridges.serve.engine_onboard")
    eo.collect_rank_shards = collect_rank_shards
    eo.cart_from_kv = cart_from_kv

    cart_mod = types.ModuleType("cartridges.cartridge")
    cart_mod.rope_theta_of = lambda cfg: 1_000_000.0 if cfg is not None else None
    cart_mod.rope_scaling_of = lambda cfg: None

    # Parent packages must exist for the dotted import to resolve.
    for name, mod in (("cartridges", types.ModuleType("cartridges")),
                      ("cartridges.serve", types.ModuleType("cartridges.serve"))):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or mod)
    monkeypatch.setitem(sys.modules, "cartridges.serve.engine_onboard", eo)
    monkeypatch.setitem(sys.modules, "cartridges.cartridge", cart_mod)
    return collected


@pytest.fixture()
def vi(monkeypatch):
    """vllm_inference reloaded against a stubbed vllm, with a fake async engine + store seeded and
    SERVE_ASYNC on. Returns the module."""
    stub = types.ModuleType("vllm")
    stub.SamplingParams = _SamplingParams
    stub.TokensPrompt = _TokensPrompt
    monkeypatch.setitem(sys.modules, "vllm", stub)
    monkeypatch.setenv("SERVE_ASYNC", "1")
    monkeypatch.setenv("CARTRIDGE_SERVED_MODEL_ID", "some-org/Served-Model")
    monkeypatch.setenv("FORCE_REONBOARD", "0")
    monkeypatch.setenv("CARTRIDGE_REGISTRY_DIR", "/tmp/reg-under-test")
    import vllm_inference
    V = importlib.reload(vllm_inference)
    V.SERVE_ASYNC = True
    V.TENSOR_PARALLEL = 2
    V.CAG_MAX_DOC_TOK = 5           # small so truncation is testable
    V.ONBOARD_ENGINE_CONCURRENCY = 4
    V.CART_STORE_DTYPE = "cc_safe"
    return V


def _seed_engine(V, store, engine=None):
    engine = engine or _FakeEngine()
    V._astate.clear()
    V._astate.update(engine=engine, tok=_FakeTok(), reg=None, store=store,
                     # a truthy hf-config path so _onboard_rope returns theta (via the stubbed helper)
                     )
    # _engine_hf_config walks engine.model_config.hf_config; give it something non-None.
    engine.model_config = types.SimpleNamespace(hf_config=types.SimpleNamespace(rope_theta=1e6),
                                                hf_text_config=None)
    return engine


def test_build_submits_key_and_persists_with_correct_cart_id(vi):
    with pytest.MonkeyPatch.context() as mp:
        collected = _install_wheel_stubs(mp)
        store = _FakeStore()
        engine = _seed_engine(vi, store)
        docs = [{"doc_id": "tenantA__doc_one", "text": "alpha beta gamma"},
                {"doc_id": "tenantA__doc_two", "text": "one two three four five six seven"}]
        res = asyncio.run(vi.onboard_cag_via_engine("/data/c", docs))

    # Every doc built one cart, keyed by its EXACT doc_id (== the store key app.py uses).
    assert res["n_cartridges"] == 2 and res["n_built"] == 2 and res["canceled"] is False
    assert {cid for cid, _ in store.puts} == {"tenantA__doc_one", "tenantA__doc_two"}
    assert all(dt == "cc_safe" for _, dt in store.puts)   # store dtype mirrors app.py's tier default
    # Each engine submission carried the build key + a 1-token greedy generation.
    keys = [sp.extra_args["cartridge_build_cart_id"] for _p, sp, _r in engine.calls]
    assert sorted(keys) == ["tenantA__doc_one", "tenantA__doc_two"]
    assert all(sp.max_tokens == 1 and sp.temperature == 0.0 for _p, sp, _r in engine.calls)
    # collect_rank_shards was called per built cart with the module's registry dir + tp size.
    assert {cid for cid, _rd, _tp in collected} == {"tenantA__doc_one", "tenantA__doc_two"}
    assert all(tp == 2 for _cid, _rd, tp in collected)


def test_doc_truncated_to_max_doc_tok(vi):
    with pytest.MonkeyPatch.context() as mp:
        _install_wheel_stubs(mp)
        store = _FakeStore()
        _seed_engine(vi, store)
        # 7 whitespace tokens; CAG_MAX_DOC_TOK=5 -> corpus_tokens counts the truncated 5.
        res = asyncio.run(vi.onboard_cag_via_engine(
            "/data/c", [{"doc_id": "tenantA__d", "text": "a b c d e f g"}]))
    assert res["corpus_tokens"] == 5


def test_per_doc_failure_is_isolated(vi):
    with pytest.MonkeyPatch.context() as mp:
        _install_wheel_stubs(mp, fail_ids={"tenantA__bad"})
        store = _FakeStore()
        _seed_engine(vi, store)
        docs = [{"doc_id": "tenantA__good", "text": "alpha beta"},
                {"doc_id": "tenantA__bad", "text": "gamma delta"}]
        res = asyncio.run(vi.onboard_cag_via_engine("/data/c", docs))
    # The good doc built + persisted; the bad one is recorded in errors, batch still succeeds.
    assert res["n_cartridges"] == 1 and res["n_built"] == 1 and res["canceled"] is False
    assert [cid for cid, _ in store.puts] == ["tenantA__good"]
    assert "tenantA__bad" in res["errors"] and "TimeoutError" in res["errors"]["tenantA__bad"]
    assert "tenantA__good" not in res["errors"]


def test_idempotent_reuse_skips_existing(vi):
    with pytest.MonkeyPatch.context() as mp:
        _install_wheel_stubs(mp)
        store = _FakeStore(existing={"tenantA__already"})
        engine = _seed_engine(vi, store)
        docs = [{"doc_id": "tenantA__already", "text": "x y"},
                {"doc_id": "tenantA__fresh", "text": "p q r"}]
        res = asyncio.run(vi.onboard_cag_via_engine("/data/c", docs))
    # Reused cart counts toward n_cartridges but not n_built, and never hit the engine.
    assert res["n_cartridges"] == 2 and res["n_built"] == 1
    assert [cid for cid, _ in store.puts] == ["tenantA__fresh"]
    assert [sp.extra_args["cartridge_build_cart_id"] for _p, sp, _r in engine.calls] == ["tenantA__fresh"]
    # corpus_tokens counts EVERY valid doc (reused included) — the read-once economics unchanged.
    assert res["corpus_tokens"] == 2 + 3


def test_response_schema_matches_app_py(vi):
    """The response carries the fields the control plane (jobs.py) reads off app.py's onboard result."""
    with pytest.MonkeyPatch.context() as mp:
        _install_wheel_stubs(mp)
        store = _FakeStore()
        _seed_engine(vi, store)
        res = asyncio.run(vi.onboard_cag_via_engine(
            "/data/c", [{"doc_id": "tenantA__d", "text": "a b c"}]))
    for key in ("n_cartridges", "canceled", "method", "train_seconds",
                "cart_seconds", "n_built", "corpus_tokens"):
        assert key in res, f"missing {key}"
    assert res["method"] == "cag_engine"


def test_route_503_when_engine_not_ready(vi, monkeypatch):
    fastapi = pytest.importorskip("fastapi")   # noqa: F841
    from fastapi.testclient import TestClient
    monkeypatch.setenv("ML_AUTH_TOKEN", "")
    vi._astate.clear()                          # no warm engine
    vi._state.clear()
    client = TestClient(vi.app)
    r = client.post("/onboard_cag",
                    json={"corpus_dir": "/data/c", "docs": [{"doc_id": "d", "text": "x"}]})
    assert r.status_code == 503


def test_route_400_on_empty_docs(vi, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    monkeypatch.setenv("ML_AUTH_TOKEN", "")
    _seed_engine(vi, _FakeStore())              # engine ready, but no docs
    client = TestClient(vi.app)
    r = client.post("/onboard_cag", json={"corpus_dir": "/data/c", "docs": []})
    assert r.status_code == 400
