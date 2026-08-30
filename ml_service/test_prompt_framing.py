"""Cart prompt framing (_compose_cart_prompt) — splice MECHANICS only, GPU-free.

These tests pin the tokenizer-level behavior of both styles against the real Qwen3
tokenizer. They deliberately say nothing about whether the engine can SERVE a framed
prompt: it can't today — the v1 connector contract is prefix-only and cart keys are
position-locked (see the CART_PROMPT_STYLE comment in vllm_inference.py; v026qual
run-5 selftest showed framed serving confabulates). Default is 'legacy'; 'framed'
stays experimental until carts bake the chat head in at build time."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
except Exception as e:  # pragma: no cover - offline CI
    _TOK = None
    _ERR = str(e)

pytestmark = pytest.mark.skipif(_TOK is None, reason="tokenizer download unavailable")


def _vi(monkeypatch, style: str):
    monkeypatch.setenv("CART_PROMPT_STYLE", style)
    import vllm_inference
    return importlib.reload(vllm_inference)


PREFIX = list(range(1_000_000, 1_000_040))  # unmistakable placeholder ids
Q = "What year was the reactor commissioned?"


def test_framed_places_prefix_inside_user_turn(monkeypatch):
    V = _vi(monkeypatch, "framed")
    ids, n_chat = V._compose_cart_prompt(_TOK, PREFIX, Q)
    s = ids.index(PREFIX[0])
    assert ids[s:s + len(PREFIX)] == PREFIX, "prefix must be spliced contiguously"
    # The span sits INSIDE the template: real chat tokens both before and after it,
    # and the tokens BEFORE it contain the system prompt (i.e. we are post-BOS/system).
    assert s > 0 and s + len(PREFIX) < len(ids)
    before = _TOK.decode(ids[:s])
    after = _TOK.decode(ids[s + len(PREFIX):])
    assert "Documents:" in before, "doc header must precede the resident span"
    assert V._SYSTEM.split(".")[0] in before, "system prompt must PRECEDE the documents"
    assert "Question:" in after and Q in after, "question must follow the resident span"
    assert V._DOC_MARK not in before + after, "marker token must not survive the splice"
    assert n_chat == len(ids) - len(PREFIX), "prefill accounting = non-resident tokens"


def test_framed_no_prefix_degenerates_to_plain_chat(monkeypatch):
    V = _vi(monkeypatch, "framed")
    ids, n_chat = V._compose_cart_prompt(_TOK, [], Q)
    assert ids == V._chat_ids(_TOK, Q) and n_chat == len(ids)


def test_legacy_is_byte_identical_prepend(monkeypatch):
    V = _vi(monkeypatch, "legacy")
    ids, n_chat = V._compose_cart_prompt(_TOK, PREFIX, Q)
    chat = V._chat_ids(_TOK, Q)
    assert ids == PREFIX + chat and n_chat == len(chat)


def test_framed_history_rides_after_documents(monkeypatch):
    V = _vi(monkeypatch, "framed")
    hist = [{"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"}]
    ids, _ = V._compose_cart_prompt(_TOK, PREFIX, Q, history=hist)
    s = ids.index(PREFIX[0])
    assert ids[s:s + len(PREFIX)] == PREFIX
    text = _TOK.decode(ids[:s]) + _TOK.decode(ids[s + len(PREFIX):])
    assert "earlier question" in text and "earlier answer" in text
