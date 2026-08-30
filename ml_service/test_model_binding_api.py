"""API-layer cart-to-model binding — the LOUD front door (platform/ml_service/vllm_inference.py).

CPU-only: no GPU, no vLLM import. The point of the feature is that a strict-policy MISMATCH is
refused with HTTP 409 BEFORE the query ever reaches the engine, so these tests seed the module's
engine state with a fake tokenizer + a FakeStore and install a sentinel "engine" whose .generate
blows up if it is ever touched — proving the refusal happens during prompt resolution, upstream of
any generation. warn/off policies must still serve (backward compatible).

Run from the repo root:  python -m pytest platform/ml_service/test_model_binding_api.py -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vllm_inference as vi  # noqa: E402

try:
    from fastapi import HTTPException  # noqa: E402
    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    _HAS_FASTAPI = False

ENGINE_ID = "Qwen/Qwen3-30B-A3B"


class FakeStore:
    """Minimal CartridgeStore stand-in: get_meta returns p / token_ids / model_ref just like the
    real store's fast path, so _cart_meta reads the stamp WITHOUT any KV decode or torch."""

    def __init__(self, metas: dict):
        self._metas = metas   # cart_id -> {"p", "token_ids", "model_ref"}

    def exists(self, cart_id: str) -> bool:
        return cart_id in self._metas

    def get_meta(self, cart_id: str) -> dict:
        return self._metas[cart_id]


class _ExplodingEngine:
    """If any binding refusal leaks past to real generation, this makes it loud instead of silent."""

    def generate(self, *a, **k):  # noqa: D401
        raise AssertionError("engine.generate reached — binding refusal did NOT gate before the engine")


class _FakeTok:
    vocab_size = 1000

    def apply_chat_template(self, *a, **k):
        return [1, 2, 3]


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Fresh per-cart caches + a no-real-engine state each test; restore the env the module read."""
    vi._kv_len.clear()
    vi._cart_ids.clear()
    vi._cart_binding_ok.clear()
    # Seed BOTH engine states so serve_query / serve_query_async find a tok+store and an exploding
    # engine without building anything. _get()/_aget() return early when the dict is truthy.
    for st in (vi._state, vi._astate):
        st.clear()
        st.update(engine=_ExplodingEngine(), llm=_ExplodingEngine(),
                  tok=_FakeTok(), reg=None, store=None)
    monkeypatch.setenv("CARTRIDGE_SERVED_MODEL_ID", ENGINE_ID)
    yield
    vi._kv_len.clear()
    vi._cart_ids.clear()
    vi._cart_binding_ok.clear()
    for st in (vi._state, vi._astate):
        st.clear()


def _store(model_ref):
    return FakeStore({"cartA": {"p": 3, "token_ids": [1, 2, 3], "model_ref": model_ref}})


def test_strict_mismatch_refuses_409_before_engine(monkeypatch):
    """strict + a cart stamped for OTHER weights -> 409 at prompt resolution, engine never touched."""
    monkeypatch.setenv("CARTRIDGE_MODEL_ENFORCE", "strict")
    store = _store("some-other-org/Other-Model#bf16")
    with pytest.raises(HTTPException if _HAS_FASTAPI else RuntimeError) as ei:
        vi._cart_prefix_ids(store, ["cartA"], vocab=1000)
    if _HAS_FASTAPI:
        assert ei.value.status_code == 409
        assert "cartA" in ei.value.detail and ENGINE_ID in ei.value.detail


def test_strict_unstamped_refuses_409(monkeypatch):
    """strict + an unstamped cart (predates the stamp) -> 409 too (strict trusts only a match)."""
    monkeypatch.setenv("CARTRIDGE_MODEL_ENFORCE", "strict")
    store = _store(None)
    with pytest.raises(HTTPException if _HAS_FASTAPI else RuntimeError) as ei:
        vi._cart_prefix_ids(store, ["cartA"], vocab=1000)
    if _HAS_FASTAPI:
        assert ei.value.status_code == 409
        assert "cartA" in ei.value.detail


def test_strict_match_serves(monkeypatch):
    """strict + a cart stamped for THIS engine's weights -> no refusal (prefix builds normally)."""
    monkeypatch.setenv("CARTRIDGE_MODEL_ENFORCE", "strict")
    store = _store(f"{ENGINE_ID}#bf16")   # exact weights match (precision tag ignored by design)
    prefix, total_p = vi._cart_prefix_ids(store, ["cartA"], vocab=1000)
    assert total_p == 3
    assert vi._cart_binding_ok.get("cartA") is True


def test_warn_mismatch_serves(monkeypatch, capfd):
    """warn (DEFAULT) + a mismatch -> serves, one warning line, no raise (backward compatible)."""
    monkeypatch.setenv("CARTRIDGE_MODEL_ENFORCE", "warn")
    store = _store("some-other-org/Other-Model#bf16")
    prefix, total_p = vi._cart_prefix_ids(store, ["cartA"], vocab=1000)   # must NOT raise
    assert total_p == 3
    assert "model-binding" in capfd.readouterr().out


def test_off_skips_check(monkeypatch):
    """off -> no classification at all; even a mismatch serves silently."""
    monkeypatch.setenv("CARTRIDGE_MODEL_ENFORCE", "off")
    store = _store("some-other-org/Other-Model#bf16")
    prefix, total_p = vi._cart_prefix_ids(store, ["cartA"], vocab=1000)
    assert total_p == 3


def test_verdict_memoized_per_cart(monkeypatch):
    """The verdict is decided once: after the first pass the cart is in _cart_binding_ok and a
    second resolution never re-reads the store (get_meta would raise if hit again)."""
    monkeypatch.setenv("CARTRIDGE_MODEL_ENFORCE", "strict")
    store = _store(f"{ENGINE_ID}#bf16")
    vi._cart_prefix_ids(store, ["cartA"], vocab=1000)

    class OneShot(FakeStore):
        def get_meta(self, cart_id):
            raise AssertionError("get_meta re-hit — meta/verdict was not cached")

    # kv_len + binding are cached now, so this store's get_meta must never be called again.
    vi._cart_prefix_ids(OneShot(store._metas), ["cartA"], vocab=1000)


@pytest.mark.skipif(not _HAS_FASTAPI, reason="route test needs fastapi TestClient")
def test_query_route_returns_409_on_mismatch(monkeypatch):
    """End-to-end through the FastAPI /query route: strict mismatch surfaces as a 409 response, and
    the exploding engine is never generated against. The serve function imports SamplingParams from
    vllm at its top; on a real box vllm IS installed (and the 409 still fires before .generate), so
    here we stub a bare `vllm` module to exercise the SAME ordering CPU-only."""
    import types
    if "vllm" not in sys.modules:
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.SamplingParams = lambda *a, **k: None
        fake_vllm.TokensPrompt = lambda *a, **k: None
        monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    from starlette.testclient import TestClient
    monkeypatch.setenv("CARTRIDGE_MODEL_ENFORCE", "strict")
    monkeypatch.setenv("ML_AUTH_TOKEN", "")          # auth off for this route test
    vi._astate["store"] = _store("some-other-org/Other-Model#bf16")
    vi._state["store"] = vi._astate["store"]
    client = TestClient(vi.app)
    r = client.post("/query", json={"doc_ids": ["cartA"], "question": "hi", "max_tokens": 4})
    assert r.status_code == 409, r.text
    assert "cartA" in r.json()["detail"]
