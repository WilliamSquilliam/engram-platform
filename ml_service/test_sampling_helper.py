"""_sampling (vllm_inference): the cart-routing SamplingParams builder, GPU-free.

vLLM is imported LAZILY inside functions in vllm_inference (the module must load in
vLLM-less envs), so these tests stub sys.modules['vllm'] with both SamplingParams shapes
— with and without extra_args — and verify: the routing dict lands under extra_args when
supported, is omitted (not crashed on) when not, and the helper resolves SamplingParams
at CALL time (v026qual run 7's selftest died on a module-scope NameError here)."""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


class _SPModern:
    def __init__(self, temperature=1.0, max_tokens=16, logprobs=None, extra_args=None):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logprobs = logprobs
        self.extra_args = extra_args


class _SPLegacy:
    def __init__(self, temperature=1.0, max_tokens=16, logprobs=None):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logprobs = logprobs


def _vi_with_stub(monkeypatch, sp_cls):
    stub = types.ModuleType("vllm")
    stub.SamplingParams = sp_cls
    monkeypatch.setitem(sys.modules, "vllm", stub)
    import vllm_inference
    V = importlib.reload(vllm_inference)
    V._SP_HAS_EXTRA_ARGS = None   # re-probe against this stub
    return V


def test_sampling_embeds_cart_ids_when_extra_args_supported(monkeypatch):
    V = _vi_with_stub(monkeypatch, _SPModern)
    sp = V._sampling(["doc-1", "doc-2"], max_tokens=8)
    assert sp.extra_args == {"cartridge_cart_ids": ["doc-1", "doc-2"]}
    assert sp.temperature == 0.0 and sp.max_tokens == 8


def test_sampling_no_carts_leaves_extra_args_unset(monkeypatch):
    V = _vi_with_stub(monkeypatch, _SPModern)
    assert V._sampling([], max_tokens=8).extra_args is None


def test_sampling_degrades_cleanly_without_extra_args(monkeypatch):
    V = _vi_with_stub(monkeypatch, _SPLegacy)
    sp = V._sampling(["doc-1"], max_tokens=8)
    assert not hasattr(sp, "extra_args") or sp.extra_args is None
    assert sp.temperature == 0.0
