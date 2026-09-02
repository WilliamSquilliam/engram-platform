"""vLLM-backed Inference Service — serve resident-KV cartridges (CAG / QRC) by doc_id through the
proven vLLM connector. This is the C2 "Inference Service" container: an OpenAI-shaped serving process
that runs in the **vLLM env** (supported stack: vLLM >= 0.26, transformers 5.14).

It is deliberately SEPARATE from ml_service/app.py (training + onboarding + HF inference) as a
division of labour, not because of a transformers version split: on the supported vLLM >= 0.26
stack (transformers 5.14) serving and onboarding can share one env. (Legacy note: the older
vLLM 0.11 path pinned transformers<5, which onboarding's transformers>=5 could not share; that
split no longer applies on the supported stack.) Division of labour:
  * control plane (backend): retrieval — pick which doc_id(s) answer a query (pgvector).
  * onboarding worker (app.py): build carts, put them in the cartridge store by doc_id.
  * THIS service: given doc_id(s) + a question, serve the answer from the resident KV — no per-query
    document prefill. Per-request routing uses the cross-process CartridgeRequestRegistry so one
    engine serves a whole corpus (validated: one L40S engine served a 2-doc corpus, 16/16 each).

Deploy: set the cart store (CARTRIDGE_STORE_BACKEND=s3 + CARTRIDGE_STORE_BUCKET, or =local +
CARTRIDGE_STORE_DIR) and CARTRIDGE_REGISTRY_DIR (a shared dir the EngineCore subprocess inherits),
then `uvicorn vllm_inference:app`. Self-test (on a GPU box, after build-corpus):
  python platform/ml_service/vllm_inference.py --selftest --workdir /opt/work
"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import inspect
import os
import random
import threading
import time
import uuid
from pathlib import Path

import httpx

# `cartridges` is a pip dependency (engram-cartridge); import it normally.
_PLATFORM_DIR = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(_PLATFORM_DIR / ".env")
    load_dotenv(_PLATFORM_DIR / ".env.local")
except ImportError:
    pass

# FastAPI is the web layer only; the core serve_query() works without it (so --selftest runs in the
# bare vLLM env). Guard the import so the module loads where FastAPI isn't installed.
try:
    from fastapi import FastAPI, HTTPException  # noqa: E402
    from pydantic import BaseModel  # noqa: E402
    from starlette.concurrency import run_in_threadpool  # noqa: E402
    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    _HAS_FASTAPI = False

# The connector reads the registry in the EngineCore subprocess, which inherits this env var; set a
# default BEFORE the vLLM engine is built so frontend writes and subprocess reads share the dir.
REGISTRY_DIR = os.environ.setdefault("CARTRIDGE_REGISTRY_DIR", "/tmp/cartridge_registry")
MODEL = os.environ.get("CARTRIDGES_MODEL", "Qwen/Qwen3-30B-A3B")
# The engine's served model id, pinned into the env BEFORE the EngineCore subprocess is spawned so
# the connector's defense-in-depth binding check (cartridges/model_binding.py) sees the SAME id this
# frontend enforces against — vLLM's own model_config.model would otherwise be the only source, and
# it can differ from CARTRIDGES_MODEL when weights are a local path. setdefault (mirrors REGISTRY_DIR
# above) so an installer-pinned canonical id is never clobbered. This is the engine side of the
# by-IDENTITY match the API layer below refuses on.
os.environ.setdefault("CARTRIDGE_SERVED_MODEL_ID", MODEL)
MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192"))
GPU_MEM_UTIL = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.85"))
# Tensor parallelism: 1 for 8B on a single L40S; 4 for Qwen3-30B-A3B bf16 across a g6e.12xlarge
# (4× L40S). KV injection under TP shards KV heads per GPU: the connector's _slice_heads slices
# each rank's kv-head shard and start_load_kv asserts the shard matches vLLM's paged cache (fails
# loud on mismatch) — proven token-exact on a live 4-GPU TP=4 box 2026-07-08 (selftest + 8-way
# serve-async). In a real deploy CARTRIDGES_MODEL is always set by the tier; this default only
# matters for a manual/local ml-serve run and now matches cartridges/config.py (30B).
TENSOR_PARALLEL = int(os.environ.get("VLLM_TP", "1"))
# "bfloat16" for bf16 checkpoints (the historical default); set "auto" for pre-quantized
# checkpoints (FP8-dynamic / compressed-tensors) where forcing bf16 would up-cast or fail.
TORCH_DTYPE = os.environ.get("VLLM_TORCH_DTYPE", "bfloat16")
# Mirrors the connector's CARTRIDGE_ON_ERROR (see vllm_cartridge_connector). "abort" makes the
# serving layer fail a request whose cart(s) couldn't hydrate instead of returning a blank-KV
# answer; "degrade" (default) returns the answer but flags it `degraded` in the done-frame.
CARTRIDGE_ON_ERROR = os.environ.get("CARTRIDGE_ON_ERROR", "degrade").lower()

# ML-plane shared-token auth (DEFAULT OFF). When ML_AUTH_TOKEN is set and non-empty, every route
# EXCEPT the liveness/readiness endpoints requires `Authorization: Bearer <token>` (constant-time
# compared); unset/empty reproduces today's behavior exactly (no auth — trust is SG-only). The
# backend attaches this same token on every ML-plane request (platform/backend/app/ml_client.py).
ML_AUTH_TOKEN = os.environ.get("ML_AUTH_TOKEN", "")
# Paths that stay open even when the token is set: load balancers / orchestrators probe these with
# no credentials. Kept as a set so app.py and this service exempt exactly the same shape.
_AUTH_OPEN_PATHS = frozenset({"/health", "/healthz", "/ready", "/readiness", "/readyz"})

# ----- serve-side knobs (all OFF/unchanged by default; new capability is opt-in) ---------------
# CART_PLACEHOLDER: how the sum(p) placeholder prefix is filled.
#   'random' (DEFAULT, today's behavior) — never-repeating ids from the advancing PRNG below.
#   'real'   — build the prefix via cartridges.client.build_prompt_ids, which submits each cart's
#              REAL token_ids when the blob carries them (so vLLM's automatic prefix cache hashes
#              to a CORRECT hit — identical tokens at identical positions imply byte-identical CAG
#              KV) and falls back to the same advancing-PRNG ids per cart otherwise.
CART_PLACEHOLDER = os.environ.get("CART_PLACEHOLDER", "random").lower()
# SERVE_CACHE_SALT: when set, passed as TokensPrompt(cache_salt=...) on BOTH cart and RAG paths.
# Empty (DEFAULT) = today's behavior (no salt). This service is single-corpus-per-deployment, so
# one salt is a fine per-deployment partition for a single-tenant demo. A MULTI-TENANT frontend
# must instead pass a PER-TENANT salt (cartridges.client.tenant_cache_salt) so two tenants'
# identical document text can never cross-tenant APC-hit — that routing lives in the frontend,
# not here (this process serves one corpus).
SERVE_CACHE_SALT = os.environ.get("SERVE_CACHE_SALT", "")
# SERVE_SPEC: '' (DEFAULT, off) | 'ngram'. When 'ngram', both engine builders pass a
# speculative_config; ngram (prompt-lookup) needs no draft model and helps the CAG path where the
# answer often quotes the resident context verbatim.
SERVE_SPEC = os.environ.get("SERVE_SPEC", "")
SERVE_SPEC_TOKENS = int(os.environ.get("SERVE_SPEC_TOKENS", "4"))
# prompt_lookup_min floor of 3: vLLM's documented min=2 falls into a corruption class under
# sampling (short-match false positives propose wrong drafts that verification silently accepts
# on ties), so we refuse to go below 3 even if the env asks for less.
SERVE_SPEC_LOOKUP_MIN = max(3, int(os.environ.get("SERVE_SPEC_LOOKUP_MIN", "3")))
SERVE_SPEC_LOOKUP_MAX = int(os.environ.get("SERVE_SPEC_LOOKUP_MAX", "4"))
# VLLM_ENFORCE_EAGER: '1' (DEFAULT, UNCHANGED — eager, no CUDA graph capture) | '0' (capture graphs).
# The default flips to '0' per-tier ONLY after the GPU gate (token-exact conformance + bench)
# passes on that tier; until then eager stays the safe default everywhere.
VLLM_ENFORCE_EAGER = os.environ.get("VLLM_ENFORCE_EAGER", "1") != "0"

# ----- engine-side onboarding knobs (POST /onboard_cag; app.py proxies to it when ONBOARD_VIA_ENGINE
# is set). This path builds one CAG cart per doc by harvesting the doc's prompt KV FROM the running
# vLLM engine (the connector stages per-TP-rank shards keyed by cart_id), then merges + persists —
# the only cart-build route for a model class the transformers forward can't load (quantized MoE VLM).
# CAG_MAX_DOC_TOK mirrors app.py's transformers-path default EXACTLY (same env, same 4096) so a doc is
# truncated to the same token budget whichever onboarding path builds it. ONBOARD_ENGINE_CONCURRENCY
# bounds in-flight per-doc engine submissions (~4; the engine batches internally). CART_STORE_DTYPE
# mirrors app.py's tier default (cc_aggr on the s3 durable tier, else cc_safe) so blobs are byte-for-
# byte the same store format regardless of which service wrote them.
CAG_MAX_DOC_TOK = int(os.environ.get("CAG_MAX_DOC_TOK", "4096"))
ONBOARD_ENGINE_CONCURRENCY = max(1, int(os.environ.get("ONBOARD_ENGINE_CONCURRENCY", "4")))
CART_STORE_DTYPE = os.environ.get(
    "CART_STORE_DTYPE",
    "cc_aggr" if os.environ.get("CARTRIDGE_STORE_BACKEND", "local").lower() == "s3"
    else "cc_safe").lower()


def _bearer_ok(auth_header: str | None, token: str) -> bool:
    """True when `auth_header` is exactly 'Bearer <token>' for the configured `token`. Constant-time
    (hmac.compare_digest) so a wrong token can't be discovered by timing. A missing/malformed header
    is a plain False. Shared shape with app.py so both ML-plane services authenticate identically."""
    if not auth_header:
        return False
    scheme, _, presented = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not presented:
        return False
    return hmac.compare_digest(presented, token)


def _spec_config() -> dict | None:
    """ngram speculative-decoding config for both engine builders, or None when off. Never carries
    an async-scheduling flag: vLLM 0.11 HARD-REJECTS speculative_config together with
    --async-scheduling. (SERVE_ASYNC/AsyncLLMEngine is a DIFFERENT mechanism — the async request
    API, not the async scheduler — and coexists with spec decode fine.)"""
    if SERVE_SPEC != "ngram":
        return None
    return {"method": "ngram", "num_speculative_tokens": SERVE_SPEC_TOKENS,
            "prompt_lookup_min": SERVE_SPEC_LOOKUP_MIN, "prompt_lookup_max": SERVE_SPEC_LOOKUP_MAX}


_state: dict = {}          # lazily-built {llm, tok, reg, store}
_lock = threading.Lock()   # serialize engine submission (one-at-a-time v1; async batching is a follow-up)
_counter = [0]             # mirrors vLLM's sequential request-id assignment for THIS llm instance
_kv_len: dict[str, int] = {}   # cart_id -> num_kv_tokens; saves re-loading the cart blob per query
_cart_ids: dict[str, list[int] | None] = {}  # cart_id -> real token_ids (None = use PRNG fallback);
#                                              cached beside _kv_len for CART_PLACEHOLDER='real'
# cart_id -> True once its model_ref stamp has been checked against the engine and cleared to serve.
# The binding verdict is decided ONCE per id — off the SAME get_meta fetch that fills _kv_len /
# _cart_ids — and held until POST /invalidate clears the entry (corpus delete, or a force
# re-onboard that rebuilt the blob under the same id with a possibly different stamp).
# See _verify_cart_binding.
_cart_binding_ok: dict[str, bool] = {}

# Invalidation epoch: bumped by POST /invalidate BEFORE it pops the three cart-meta dicts
# above. _cart_meta snapshots it before its store read and refuses to memoize across a bump —
# otherwise a query thread that fetched meta just before the pop would re-install the deleted
# blob's p/token_ids/binding verdict right after it (check-then-act). One shared int, no lock:
# a spurious bump only costs one uncached read.
_inval_epoch = [0]

# Placeholder token ids must NEVER repeat across requests: vLLM's automatic prefix cache
# (on by default in v1) hashes block token ids, so a repeated placeholder sequence — e.g.
# two carts with the same total length under a per-request Random(0) — can match a PREVIOUS
# request's cached blocks and silently serve that request's cart KV without ever consulting
# the connector (the collision class qualify_preemption.py documents). One shared advancing
# PRNG per process guarantees no two requests submit the same placeholder prefix.
_ph_rng = random.Random(0xC0FFEE)


def _placeholder_ids(n: int, vocab: int) -> list[int]:
    return [_ph_rng.randrange(vocab) for _ in range(n)]


def _knob_state() -> dict:
    """The serve-side knob states, surfaced on /health and /stats so ops can eyeball the config
    at a glance (a wrong CART_PLACEHOLDER/spec/eager/salt is otherwise invisible until a query
    misbehaves). cache_salt is reported as set-or-not only — never echo the salt value, it
    partitions the tenant/corpus hash space."""
    return {"placeholder": CART_PLACEHOLDER, "spec": SERVE_SPEC or "off",
            "enforce_eager": VLLM_ENFORCE_EAGER, "cache_salt_set": bool(SERVE_CACHE_SALT)}


def _tokens_prompt(prompt_ids: list[int]):
    """Build the vLLM TokensPrompt for a prompt, attaching cache_salt=SERVE_CACHE_SALT when set.
    vLLM 0.11's TokensPrompt accepts an OPTIONAL cache_salt that partitions the automatic-prefix-
    cache hash space; unset (DEFAULT) reproduces today's behavior exactly (no salt key). Kept as
    one helper so cart + RAG, sync + async + streaming all salt identically (a per-corpus/tenant
    partition is worthless if one path forgets it)."""
    from vllm import TokensPrompt
    if SERVE_CACHE_SALT:
        return TokensPrompt(prompt_token_ids=prompt_ids, cache_salt=SERVE_CACHE_SALT)
    return TokensPrompt(prompt_token_ids=prompt_ids)


def _store_extra() -> dict:
    """Cartridge-store config for the connector (and our own loads), from env. s3 = durable prod tier.
    cart_cache_dir must be a HOST mount (each cart ~0.4 GB; the in-container default fills the root
    volume at corpus scale) and cart_cache_items bounds the local mirror."""
    if os.environ.get("CARTRIDGE_STORE_BACKEND", "local").lower() == "s3":
        extra = {"cart_store_backend": "s3", "cart_store_bucket": os.environ["CARTRIDGE_STORE_BUCKET"],
                 "cart_store_prefix": os.environ.get("CARTRIDGE_STORE_PREFIX", "cartridges"),
                 "cart_cache_items": int(os.environ.get("CART_CACHE_ITEMS", "64"))}
        if os.environ.get("CART_CACHE_DIR"):
            extra["cart_cache_dir"] = os.environ["CART_CACHE_DIR"]
        return extra
    return {"cart_store_backend": "local",
            "cart_store_dir": os.environ.get("CARTRIDGE_STORE_DIR", "/opt/work/store")}


_build_lock = threading.Lock()


def _get() -> dict:
    """Lazily build the vLLM engine + connector + cart store (one model load, cached).
    Build-locked: two first-queries racing here used to construct TWO engine cores in one
    process — the second saw the first's half-loaded weights as 'used memory' and died with
    free-memory ValueError (found live on the demo box)."""
    if _state:
        return _state
    with _build_lock:
        if _state:
            return _state
        return _build_engine()


def _build_engine() -> dict:
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.config import KVTransferConfig

    from cartridges.serve.vllm_cartridge_connector import (
        _store_from_extra,
        make_request_registry,
    )
    extra = _store_extra()
    ktc = KVTransferConfig(
        kv_connector="CartridgeKVConnector", kv_role="kv_both",
        kv_connector_module_path="cartridges.serve.vllm_cartridge_connector",
        kv_connector_extra_config=extra,   # NO default_cart_ids -> every request routes via the registry
    )
    # enforce_eager + speculative_config are env-driven (defaults reproduce today's build exactly:
    # eager on, no spec). spec_config is added ONLY when SERVE_SPEC=ngram — and it never carries an
    # async-scheduling flag (v0.11 rejects that combo; see _spec_config).
    kw = {"enforce_eager": VLLM_ENFORCE_EAGER}
    spec = _spec_config()
    if spec is not None:
        kw["speculative_config"] = spec
    llm = LLM(model=MODEL, dtype=TORCH_DTYPE, gpu_memory_utilization=GPU_MEM_UTIL,
              tensor_parallel_size=TENSOR_PARALLEL, max_model_len=MAX_MODEL_LEN,
              kv_transfer_config=ktc, **kw)
    # Registry backend from env (CARTRIDGE_REGISTRY_URL redis for a fleet, else the shared dir) —
    # the same selection the EngineCore-subprocess connector makes, so both sides rendezvous.
    _state.update(llm=llm, tok=AutoTokenizer.from_pretrained(MODEL),
                  reg=make_request_registry(), store=_store_from_extra(extra))
    return _state


def _req_metrics(ro, wall_ms: float, prompt_tokens: int, resident_kv_tokens: int) -> dict:
    """Turn a vLLM RequestOutput + wall time into MEASURED per-query metrics (this is the real
    runtime data the UI shows — not a documentation constant):
      prompt_tokens      tokens actually re-prefilled THIS query (cart path: ~the question only;
                         RAG path: the whole retrieved context, every query — the read-once lever).
      resident_kv_tokens the cart KV served from the resident cache (read once at onboarding, 0 for RAG).
      ttft_ms / *_tps    from vLLM's own request timers when present; decode_tps falls back to
                         wall-clock so a number is always real, never modeled."""
    gen_tokens = len(ro.outputs[0].token_ids)
    ttft_ms = prefill_tps = decode_tps = None
    m = getattr(ro, "metrics", None)
    try:
        if m and getattr(m, "first_token_time", None) and getattr(m, "arrival_time", None):
            ttft_s = m.first_token_time - m.arrival_time
            if ttft_s > 0:
                ttft_ms = round(ttft_s * 1000.0, 1)
                if prompt_tokens:
                    prefill_tps = round(prompt_tokens / ttft_s, 1)
        if m and getattr(m, "finished_time", None) and getattr(m, "first_token_time", None) and gen_tokens > 1:
            dec_s = m.finished_time - m.first_token_time
            if dec_s > 0:
                decode_tps = round((gen_tokens - 1) / dec_s, 1)
    except Exception:  # noqa: BLE001 — metrics are best-effort; wall-clock below always yields a number
        pass
    if decode_tps is None and gen_tokens and wall_ms:
        decode_tps = round(gen_tokens / (wall_ms / 1000.0), 1)
    return {"latency_ms": wall_ms, "ttft_ms": ttft_ms, "prompt_tokens": prompt_tokens,
            "resident_kv_tokens": resident_kv_tokens, "gen_tokens": gen_tokens,
            "cached_tokens": getattr(ro, "num_cached_tokens", None),
            "prefill_tps": prefill_tps, "decode_tps": decode_tps, "measured": True}


def _generate(prompt_ids: list[int], cart_ids: list[str] | None, max_tokens: int):
    """Run ONE greedy generation on the owned engine, optionally routing `cart_ids` for KV injection.
    cart_ids given  -> register the request so the connector scatters those carts (the resident-KV path).
    cart_ids None   -> submit WITHOUT registering; the connector reports 0 external tokens, so vLLM
                       prefills the whole prompt normally = the live RAG baseline on the SAME engine.
    Serialized under `_lock` so vLLM's sequential request ids stay in step with `_counter` (16/16 path);
    returns (RequestOutput, wall_ms)."""
    st = _get()
    llm, reg = st["llm"], st["reg"]
    with _lock:
        rid = str(_counter[0])
        if cart_ids:
            reg.set(rid, list(cart_ids))       # route THIS request to THESE carts (before submission)
        t0 = time.perf_counter()
        try:
            out = llm.generate([_tokens_prompt(prompt_ids)],
                               _sampling(cart_ids, max_tokens=max_tokens), use_tqdm=False)
        finally:
            if cart_ids:
                reg.pop(rid)
        wall_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        got = out[0].request_id
        _counter[0] = int(got) + 1 if got.isdigit() else _counter[0] + 1
        if cart_ids and got != rid:            # id drift => cart not injected; fail loud (don't serve a wrong answer)
            raise RuntimeError(f"request-id drift (registered {rid}, vLLM used {got}); cart not injected")
    return out[0], wall_ms


# The serving system prompt: makes the model answer like an assistant (not a paper-summarizer)
# and name its sources — each stored document begins with its title on the first line.
_SYSTEM = ("You are a helpful assistant answering questions from the documents provided to you. "
           "Answer the question directly and conversationally. Keep answers concise — two to "
           "five sentences — unless the user explicitly asks for more detail. "
           "Each document begins with its "
           "title on its first line — when your answer draws on a document, mention that title "
           "naturally (e.g. 'According to \"...\"'). If the documents do not contain the answer, "
           "say so briefly instead of guessing.")


# The fixed instruction the /describe endpoint serves against each doc's resident cart — a single
# short sentence saying what the document is and what it contains (used as retrieval metadata by the
# control plane, and shown as a secondary line in the Documents tab).
_DESCRIBE_INSTRUCTION = ("Describe this document in one short sentence: what it is and what "
                         "information it contains.")


def _chat_ids(tok, user_text: str, history: list | None = None) -> list[int]:
    """Tokenized chat-template prompt (system + prior turns + user). Using the model's chat
    template is what makes answers END at EOS — the old raw 'Q: ... A:' completion style never
    terminated, so every answer rambled to the max_tokens ceiling (found live on the demo)."""
    msgs = [{"role": "system", "content": _SYSTEM}]
    for m in history or []:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            msgs.append({"role": m["role"], "content": str(m["content"])})
    msgs.append({"role": "user", "content": user_text})
    # The reasoning-control kwarg varies by model family and a wrong one is SILENTLY ignored (jinja just
    # sees an unused variable): Command A+'s template ignores enable_thinking and ends the
    # generation prompt with <|START_THINKING|> — every answer was the model's thinking
    # stream ("The need to answer: ...") burning the whole token budget (found live on the
    # bench). Its real control is reasoning=False, which renders a pre-closed
    # <|START_THINKING|><|END_THINKING|> block so generation goes straight to the answer.
    # Pass BOTH knobs (each family reads its own, ignores the other); fall back for
    # templates that reject unexpected kwargs outright.
    try:
        out = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      reasoning=False, enable_thinking=False)
    except TypeError:  # template that rejects unknown kwargs
        try:
            out = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                          enable_thinking=False)
        except TypeError:
            out = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    # transformers 4.x returns list[int]; 5.x returns a BatchEncoding (and some paths nest
    # a batch dim). Normalize to a flat list — every caller concatenates with plain lists,
    # and on the vLLM 0.26 stack (transformers 5.14 in the SERVE env) the raw return would
    # TypeError at prompt assembly (found 2026-08-09 during the 0.26 qualification).
    ids = getattr(out, "input_ids", out)
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


# ---- cart prompt framing --------------------------------------------------------------
# CART_PROMPT_STYLE: where the resident document span sits in the prompt.
#   'legacy' (DEFAULT) — resident span first, chat template appended after. The ONLY placement
#       the serving stack can honor today, for two independent reasons:
#       (1) the vLLM v1 connector contract is PREFIX-only: get_num_new_matched_tokens counts
#           external tokens from the START of the prompt, so a mid-prompt span misaligns the
#           claim — cart KV lands on the chat-head tokens and the span's tail is computed
#           from placeholder ids. Observed live (v026qual run-5 selftest under 'framed'):
#           every doc fact confabulated.
#       (2) cart keys keep their original-position RoPE rotation — built at positions 0..p-1,
#           no de-RoPE/re-base at serve (encoder.py compacts post-RoPE; qrc.py routing relies
#           on original-position keys) — so the span must SIT at positions 0..p-1.
#   'framed' (EXPERIMENTAL — broken for connector serving, see above) — splices the span
#       inside the chat user turn after a "Documents:" header, the slot RAG text occupies.
#       The PLACEMENT is the right target: pre-BOS docs are off-distribution, measured as a
#       large factual-accuracy gap (fact-grid 2026-08-03: lossless carts 29-54% exact-fact
#       vs 100% for the same text inside the user turn, same model). The viable route there
#       is BUILD-side: bake system prompt + doc header into the cart itself so the resident
#       span is a prefix that CONTAINS the chat head — prefix contract and RoPE both hold.
#       'framed' stays as the serve-side experiment until baked-head carts exist.
CART_PROMPT_STYLE = os.environ.get("CART_PROMPT_STYLE", "legacy").lower()
_DOC_MARK = "␞"  # SYMBOL FOR RECORD SEPARATOR: never occurs in prose, survives BPE


def _compose_cart_prompt(tok, prefix: list[int], question: str,
                         history: list | None = None) -> tuple[list[int], int]:
    """(prompt_ids, n_prefilled) — the cart placeholder span framed per CART_PROMPT_STYLE.
    n_prefilled counts the NON-resident tokens (the conversation actually prefilled),
    which is what the done-frame's prompt_tokens metric reports."""
    if CART_PROMPT_STYLE == "legacy" or not prefix:
        chat_ids = _chat_ids(tok, question, history)
        return prefix + chat_ids, len(chat_ids)
    marked = _chat_ids(tok, f"Documents:\n\n{_DOC_MARK}\n\nQuestion: {question}", history)
    mark_ids = tok(_DOC_MARK, add_special_tokens=False).input_ids
    n = len(mark_ids)
    for i in range(len(marked) - n + 1):
        if marked[i:i + n] == mark_ids:
            out = marked[:i] + prefix + marked[i + n:]
            return out, len(out) - len(prefix)
    # Tokenizer merged the marker into a neighbor (not observed on Qwen3; fail SAFE by
    # falling back to the legacy shape rather than serving a prompt with a stray marker).
    chat_ids = _chat_ids(tok, question, history)
    return prefix + chat_ids, len(chat_ids)


_SP_HAS_EXTRA_ARGS: bool | None = None   # probed on first use (SamplingParams impl varies by vLLM)


def _sampling(cart_ids, **kw):
    """SamplingParams with cart routing embedded via extra_args['cartridge_cart_ids'].
    vLLM 0.26 rewrites request ids before the scheduler (caller_rid -> 'rid-xxxxxxxx'),
    so registry-by-rid alone goes silently cartless there (v026qual run 6); the request-
    embedded channel survives any rid rewrite. Registry writes stay alongside for stacks
    whose SamplingParams lacks extra_args (and as the ops/invalidation surface)."""
    from vllm import SamplingParams  # local, like every vLLM import here (module loads GPU-free)
    global _SP_HAS_EXTRA_ARGS
    if _SP_HAS_EXTRA_ARGS is None:
        try:
            SamplingParams(extra_args=None)
            _SP_HAS_EXTRA_ARGS = True
        except TypeError:
            _SP_HAS_EXTRA_ARGS = False
    if cart_ids and _SP_HAS_EXTRA_ARGS:
        kw["extra_args"] = {"cartridge_cart_ids": [str(c) for c in cart_ids]}
    return SamplingParams(temperature=0.0, **kw)


def _strip_think(text: str) -> str:
    """Defensively drop a leading <think>...</think> block (Qwen3 emits one if the template's
    thinking switch is ignored)."""
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def _verify_cart_binding(cart_id: str, model_ref: str | None) -> None:
    """Cart-to-model binding gate at the API frontend — the LOUD front door (the connector in the
    EngineCore subprocess re-checks as defense in depth; see cartridges/model_binding.py for the
    two-layer contract). Classify the cart's model_ref stamp against this engine's served id, then
    apply the deployment policy:
      * off              — no check (returns immediately).
      * warn (DEFAULT)   — mismatch/unstamped is logged ONCE per cart_id and served (backward
                           compatible: existing demo libraries predate the stamp).
      * strict           — mismatch OR unstamped raises HTTPException(409) with the exact ids BEFORE
                           the query ever reaches the engine, so the user gets a clear refusal
                           instead of a silently-wrong answer.
    The verdict is memoized in _cart_binding_ok so it is decided once per cart_id (immutable for a
    given id) and never re-classified on a repeat query. When the engine id is unknown (no env, no
    vLLM config — unit tests / bare runs) verdict() returns 'match', so binding is simply not
    enforced. Raising here does not import fastapi at module scope (guarded)."""
    if cart_id in _cart_binding_ok:
        return
    from cartridges.model_binding import policy, verdict
    pol = policy()
    if pol == "off":
        _cart_binding_ok[cart_id] = True
        return
    engine = os.environ.get("CARTRIDGE_SERVED_MODEL_ID") or MODEL
    kind, detail = verdict(model_ref, engine)
    if kind == "match":
        _cart_binding_ok[cart_id] = True
        return
    if pol == "warn":
        # One line per cart_id (memoized below), never per query — a mismatched library would
        # otherwise flood the log on every turn.
        print(f"[model-binding] WARN cart {cart_id!r} ({kind}): {detail} — serving anyway under "
              f"CARTRIDGE_MODEL_ENFORCE=warn", flush=True)
        _cart_binding_ok[cart_id] = True
        return
    # strict + (mismatch|unstamped): refuse loudly at the front door with the exact ids.
    ref = model_ref if model_ref else "(unstamped)"
    msg = (f"cart {cart_id} was built by {ref} but this engine serves {engine}; "
           f"re-onboard the corpus on the serving model")
    if _HAS_FASTAPI:
        raise HTTPException(409, msg)
    raise RuntimeError(msg)   # bare vLLM env (--selftest) without fastapi: still refuse, loudly


def _cart_meta(store, cart_id: str) -> tuple[int, list[int] | None]:
    """(p, real token_ids-or-None) for a cart, cached — stable for a given id until a lifecycle
    action (delete / force re-onboard) clears the entry via POST /invalidate, so read them ONCE
    via the store's lightweight get_meta (a safetensors header read for cc blobs — no full KV
    decode) and cache both beside each other. Repeat queries hit the cache, never the store.
    token_ids is cached even in 'random' mode (harmless) so a mode flip needs no reload.
    The SAME get_meta fetch carries the cart's model_ref, so the model-binding gate is applied
    here (once per cart_id) — strict policy raises HTTPException(409) BEFORE any engine call."""
    n = _kv_len.get(cart_id)
    if n is not None:
        return n, _cart_ids.get(cart_id)
    epoch = _inval_epoch[0]
    if not store.exists(cart_id):
        raise LookupError(f"cartridge not found in store: {cart_id}")
    get_meta = getattr(store, "get_meta", None)
    if get_meta is not None:
        m = get_meta(cart_id)
        n, tids = m["p"], m["token_ids"]
        _verify_cart_binding(cart_id, m.get("model_ref"))
    else:  # older store without the fast path: full load (materializes KV)
        cart = store.get(cart_id)
        n, tids = cart.num_kv_tokens(), cart.token_ids
        _verify_cart_binding(cart_id, getattr(cart, "model_ref", None))
    if _inval_epoch[0] != epoch:
        # An /invalidate raced this fetch: what we just read (and the binding verdict
        # _verify_cart_binding memoized) may describe the pre-invalidation blob. Serve
        # this one request from the local values but memoize NOTHING — the next request
        # re-reads the store fresh.
        _cart_binding_ok.pop(cart_id, None)
        return n, tids
    _kv_len[cart_id] = n
    _cart_ids[cart_id] = tids
    return n, tids


def _cart_kv_len(store, cart_id: str) -> int:
    """num_kv_tokens for a cart (see _cart_meta for the caching rationale)."""
    return _cart_meta(store, cart_id)[0]


def _cart_prefix_ids(store, doc_ids: list[str], vocab: int) -> tuple[list[int], int]:
    """The sum(p) placeholder prefix for `doc_ids` + total_p, honoring CART_PLACEHOLDER:
      'real' — each cart contributes its REAL token_ids when the blob carries them (correct-by-hash
               APC reuse), else the same never-repeating PRNG fill per cart. Built from the cached
               per-cart meta so a repeat query never re-hits the store.
      'random' (DEFAULT) — sum(p) never-repeating PRNG ids (today's behavior, byte-for-byte)."""
    total_p = 0
    prefix: list[int] = []
    for d in doc_ids:
        p, tids = _cart_meta(store, d)
        total_p += p
        if CART_PLACEHOLDER == "real" and tids is not None:
            if len(tids) != p:  # a corrupt/mismatched blob must never scatter KV under wrong ids
                raise ValueError(
                    f"cart {d!r} has {len(tids)} token_ids but p={p} — refusing to build a "
                    "prompt whose placeholder count disagrees with the cart's KV length")
            prefix.extend(int(t) for t in tids)
        else:
            prefix.extend(_placeholder_ids(p, vocab))
    return prefix, total_p


def serve_query(doc_ids: list[str], question: str, max_tokens: int = 64,
                history: list | None = None) -> dict:
    """Answer `question` from the resident KV of `doc_ids` (CAG carts) — the product serve path. The
    placeholder prefix is sum(cart.p) random tokens (KV overwritten by the connector) + the templated
    conversation; only the conversation is actually prefilled (the cart KV is resident), which is
    what prompt_tokens records. `history` = prior chat turns — they ride as small per-turn prefill
    ON TOP of the resident corpus KV. Returns {answer, metrics} with MEASURED tokens + latency."""
    if not doc_ids:
        raise ValueError("doc_ids required")
    st = _get()
    tok, store = st["tok"], st["store"]
    vocab = getattr(tok, "vocab_size", None) or len(tok)
    prefix, total_p = _cart_prefix_ids(store, list(doc_ids), vocab)
    prompt_ids, n_chat = _compose_cart_prompt(tok, prefix, question, history)
    ro, wall_ms = _generate(prompt_ids, list(doc_ids), max_tokens)
    ans = _strip_think(tok.decode(list(ro.outputs[0].token_ids), skip_special_tokens=True))
    return {"answer": ans, "metrics": _req_metrics(ro, wall_ms, n_chat, total_p)}


def serve_rag(context: str, question: str, max_tokens: int = 64,
              history: list | None = None) -> dict:
    """The RAG baseline, measured on the SAME engine/hardware: prefill the retrieved `context` + the
    templated conversation (no cart), then generate. This is the honest head-to-head — RAG
    re-prefills the whole context every query (prompt_tokens), the cart path does not."""
    st = _get()
    tok = st["tok"]
    user = (f"Documents:\n\n{context}\n\nQuestion: {question}") if context else question
    prompt_ids = _chat_ids(tok, user, history)
    if not prompt_ids:
        raise ValueError("empty RAG prompt")
    ro, wall_ms = _generate(prompt_ids, None, max_tokens)
    ans = _strip_think(tok.decode(list(ro.outputs[0].token_ids), skip_special_tokens=True))
    return {"answer": ans, "metrics": _req_metrics(ro, wall_ms, len(prompt_ids), 0)}


# ----- async batching (throughput) -------------------------------------------------------------
# vLLM's async engine batches concurrent requests (its strength); explicit per-request UUIDs route
# the cross-process registry correctly under concurrency, so there's no _lock and no request-id
# counter (the fragile part of the sync path). VALIDATED 2026-07-03 (vLLM 0.11 / Qwen3-0.6B, WSL
# RTX4070): validate_cag_qrc --stage serve-async — 8 concurrent interleaved requests batched by
# AsyncLLMEngine, every one answered from ITS cart, deterministic across rounds, routing
# negative-control clean. DEFAULT is now ON: the async path is validated 8-concurrent and the
# deployed demo already runs it, so it is the right default for fleet-style concurrency. A
# single-user local run that wants the simplest one-at-a-time path can set SERVE_ASYNC=0. When on,
# the FastAPI handler is async so requests batch.
SERVE_ASYNC = os.environ.get("SERVE_ASYNC", "1") == "1"
_astate: dict = {}


def _aget() -> dict:
    if _astate:
        return _astate
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    from transformers import AutoTokenizer
    from vllm import AsyncEngineArgs, AsyncLLMEngine
    from vllm.config import KVTransferConfig

    from cartridges.serve.vllm_cartridge_connector import (
        _store_from_extra,
        make_request_registry,
    )
    extra = _store_extra()
    ktc = KVTransferConfig(
        kv_connector="CartridgeKVConnector", kv_role="kv_both",
        kv_connector_module_path="cartridges.serve.vllm_cartridge_connector",
        kv_connector_extra_config=extra,
    )
    # Same env-driven enforce_eager + spec_config as the sync builder. NOTE: AsyncLLMEngine is
    # the async REQUEST API — NOT vLLM's --async-scheduling. speculative_config is fine here;
    # it is --async-scheduling that v0.11 hard-rejects together with a spec config (see _spec_config).
    kw = {"enforce_eager": VLLM_ENFORCE_EAGER}
    spec = _spec_config()
    if spec is not None:
        kw["speculative_config"] = spec
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=MODEL, dtype=TORCH_DTYPE, gpu_memory_utilization=GPU_MEM_UTIL,
        tensor_parallel_size=TENSOR_PARALLEL, max_model_len=MAX_MODEL_LEN,
        kv_transfer_config=ktc, **kw))
    _astate.update(engine=engine, tok=AutoTokenizer.from_pretrained(MODEL),
                   reg=make_request_registry(), store=_store_from_extra(extra))
    return _astate


def _prompt_ids(tok, store, doc_ids: list[str], question: str,
                history: list | None = None) -> tuple[list[int], int, int]:
    """(prompt token ids, prefilled-conversation length, resident cart tokens) for a cart request."""
    if not doc_ids:
        raise ValueError("doc_ids required")
    prefix, total_p = _cart_prefix_ids(store, list(doc_ids),
                                       getattr(tok, "vocab_size", None) or len(tok))
    prompt_ids, n_chat = _compose_cart_prompt(tok, prefix, question, history)
    return prompt_ids, n_chat, total_p


async def serve_query_async(doc_ids: list[str], question: str, max_tokens: int = 64,
                            history: list | None = None) -> dict:
    """Concurrent-safe serve: explicit uuid request_id keys the registry, so vLLM can batch many
    in-flight requests and each still routes to its own cart. No lock, no counter. Returns
    {answer, metrics} with the same MEASURED shape as the sync path (wall-clock under concurrency
    includes queueing — that's the honest number a fleet sees)."""
    st = _aget()
    tok, reg, store = st["tok"], st["reg"], st["store"]
    prompt, q_len, total_p = _prompt_ids(tok, store, doc_ids, question, history)
    rid = uuid.uuid4().hex
    reg.set(rid, list(doc_ids))
    final = None
    t0 = time.perf_counter()
    try:
        async for out in st["engine"].generate(
                _tokens_prompt(prompt),
                _sampling(doc_ids, max_tokens=max_tokens), rid):
            final = out
    finally:
        reg.pop(rid)
    degrade_reason = reg.pop_degraded(rid)
    if degrade_reason and CARTRIDGE_ON_ERROR == "abort":
        raise RuntimeError(f"cart degraded ({degrade_reason}): KV could not be served "
                           f"(e.g. GPU OOM); answer withheld under CARTRIDGE_ON_ERROR=abort")
    wall_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    ans = _strip_think(tok.decode(list(final.outputs[0].token_ids), skip_special_tokens=True))
    m = _req_metrics(final, wall_ms, q_len, total_p)
    m["degraded"], m["degrade_reason"] = bool(degrade_reason), degrade_reason
    return {"answer": ans, "metrics": m}


async def serve_rag_async(context: str, question: str, max_tokens: int = 64,
                          history: list | None = None) -> dict:
    """RAG baseline on the ASYNC engine. On a SERVE_ASYNC=1 box the sync serve_rag would
    lazily construct a second (sync) engine core beside the running async one — the exact
    'Free memory ...' OOM the build-lock comment documents. Same measured shape."""
    from vllm import SamplingParams
    st = _aget()
    tok = st["tok"]
    user = (f"Documents:\n\n{context}\n\nQuestion: {question}") if context else question
    prompt_ids = _chat_ids(tok, user, history)
    if not prompt_ids:
        raise ValueError("empty RAG prompt")
    rid = uuid.uuid4().hex
    final = None
    t0 = time.perf_counter()
    async for out in st["engine"].generate(
            _tokens_prompt(prompt_ids),
            SamplingParams(temperature=0.0, max_tokens=max_tokens), rid):
        final = out
    wall_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    ans = _strip_think(tok.decode(list(final.outputs[0].token_ids), skip_special_tokens=True))
    return {"answer": ans, "metrics": _req_metrics(final, wall_ms, len(prompt_ids), 0)}


async def _stream_generate(prompt_ids: list[int], cart_ids: list[str] | None, max_tokens: int,
                           prefill_tokens: int, resident_tokens: int, want_conf: bool = False):
    """SSE token stream on the async engine: {'delta': text} per new fragment, then a final
    {'done': true, 'metrics': {...}} with MEASURED numbers — TTFT here is the real time to the
    first visible token, the number streaming makes user-facing. When want_conf, the metrics also
    carry `confidence` = mean per-token logprob of the answer (the adaptive router's escalation
    signal; logprobs=1 adds negligible overhead at greedy decode)."""
    import json as _json

    st = _aget()
    reg = st["reg"]
    rid = uuid.uuid4().hex
    if cart_ids:
        reg.set(rid, list(cart_ids))
    t0 = time.perf_counter()
    first = None
    sent = 0
    final = None
    try:
        async for out in st["engine"].generate(
                _tokens_prompt(prompt_ids),
                _sampling(cart_ids, max_tokens=max_tokens,
                          logprobs=1 if want_conf else None), rid):
            final = out
            text = out.outputs[0].text
            if len(text) > sent:
                if first is None:
                    first = time.perf_counter()
                yield f"data: {_json.dumps({'delta': text[sent:]})}\n\n"
                sent = len(text)
    finally:
        if cart_ids:
            reg.pop(rid)
    # Did the connector serve this request over BLANK KV (cart failed to hydrate)? The reverse
    # channel tells us here, at the done-frame — so a degraded answer is never returned silently.
    degrade_reason = reg.pop_degraded(rid) if cart_ids else None
    if degrade_reason and CARTRIDGE_ON_ERROR == "abort":
        yield f"data: {_json.dumps({'error': 'cart_degraded', 'reason': degrade_reason, 'detail': 'the cartridge KV could not be served (e.g. GPU OOM); answer withheld under CARTRIDGE_ON_ERROR=abort'})}\n\n"
        return
    now = time.perf_counter()
    gen = len(final.outputs[0].token_ids) if final else 0
    ttft_s = (first - t0) if first else None
    # confidence = mean per-token logprob (closer to 0 = more confident). The adaptive router
    # escalates the cart side to the RAG backup when this drops below ADAPTIVE_THETA.
    conf = None
    if want_conf and final and gen:
        cl = getattr(final.outputs[0], "cumulative_logprob", None)
        if cl is not None:
            conf = round(cl / gen, 3)
    metrics = {"latency_ms": round((now - t0) * 1000.0, 1),
               "ttft_ms": round(ttft_s * 1000.0, 1) if ttft_s else None,
               "prompt_tokens": prefill_tokens, "resident_kv_tokens": resident_tokens,
               # Tokens vLLM's own prefix cache reused (repeat context) — surfaced so a
               # cache-hit RAG TTFT labels itself instead of looking like magic.
               "cached_tokens": getattr(final, "num_cached_tokens", None),
               "gen_tokens": gen, "confidence": conf,
               "prefill_tps": round(prefill_tokens / ttft_s, 1) if ttft_s and prefill_tokens else None,
               "decode_tps": round((gen - 1) / (now - first), 1) if first and gen > 1 and now > first else None,
               # True when the cart(s) failed to hydrate and this answer was computed over blank KV
               # (unreliable). Surfaced so a client/judge never treats a degraded answer as valid.
               "degraded": bool(degrade_reason),
               "degrade_reason": degrade_reason,
               "measured": True}
    yield f"data: {_json.dumps({'done': True, 'metrics': metrics})}\n\n"


# ============================================================================
# Engine-side onboarding — POST /onboard_cag
# ----------------------------------------------------------------------------
# The onboarding worker's transformers forward (app.py) CANNOT load this model class (a quantized
# MoE VLM). The ONLY process that can produce this model's KV is the vLLM engine already serving it,
# so onboarding runs THROUGH the engine: submit each doc's prompt with a build key in
# SamplingParams.extra_args, the connector harvests that request's full prompt KV at completion and
# stages per-TP-rank shards under $CARTRIDGE_REGISTRY_DIR/builds/<cart_id>/; we then merge the shards
# (collect_rank_shards) into a full-head KV, wrap it in a Cartridge (cart_from_kv) and persist via the
# SAME store write app.py uses. Request/response schema is IDENTICAL to app.py's /onboard_cag: app.py
# proxies verbatim when ONBOARD_VIA_ENGINE is set, so the control plane sees no difference — including
# the progress-report mechanism (mirrored below from app.py's _make_reporter).


def _make_reporter(progress_url, token):
    """Non-blocking, failure-proof progress heartbeat + cancel Event — IDENTICAL mechanism to
    app.py's _make_reporter so the control plane's progress bar/cancel behave the same whichever
    service ran the onboard. Each POST carries {progress, eta_seconds, detail} with the
    X-Internal-Token header, runs in a daemon thread with a short timeout, and swallows any error
    (a slow/down control plane must never stall GPU work). The heartbeat RESPONSE may carry
    {"cancel": true} (the worker's only inbound channel); when it does we set the Event."""
    cancel_event = threading.Event()
    if not progress_url:
        return (lambda *a, **k: None), cancel_event

    def report(progress, eta_seconds=None, detail=None):
        def _send():
            try:
                resp = httpx.post(
                    progress_url,
                    json={"progress": float(progress), "eta_seconds": eta_seconds, "detail": detail},
                    headers={"X-Internal-Token": token} if token else {},
                    timeout=3.0,
                )
                if resp.json().get("cancel"):
                    cancel_event.set()
            except Exception:  # noqa: BLE001  (heartbeats are best-effort)
                pass

        threading.Thread(target=_send, daemon=True).start()

    return report, cancel_event


def _engine_hf_config(st: dict):
    """The engine's HF model config (for rope_theta / rope_scaling), reached across the sync-LLM and
    async-engine handle shapes vLLM exposes. Returns the config object or None (None -> rope helpers
    return None, and a plain-RoPE model needs neither). Kept defensive: the attribute path to
    model_config differs between vLLM's LLM (llm_engine.model_config) and AsyncLLMEngine
    (model_config), and the HF config sits under .hf_config on the vLLM ModelConfig."""
    eng = st.get("engine") or st.get("llm")
    if eng is None:
        return None
    mc = (getattr(eng, "model_config", None)
          or getattr(getattr(eng, "llm_engine", None), "model_config", None))
    if mc is None:
        return None
    # vLLM's ModelConfig exposes the transformers config as .hf_config (and .hf_text_config for
    # multimodal, which is the text tower carrying rope_*). Prefer the text config when present.
    return getattr(mc, "hf_text_config", None) or getattr(mc, "hf_config", None) or mc


def _onboard_rope() -> tuple[float | None, dict | None]:
    """(rope_theta, rope_scaling) for every cart this service builds — the cc store dtypes de-rotate
    K and need theta at save time, so this must be authoritative for the SERVED weights. The wheel's
    rope_theta_of/rope_scaling_of read them across transformers versions (config.rope_theta on <5,
    config.rope_parameters on 5.x). Source order: the engine handle's HF config first (the exact
    config the engine loaded); if the vLLM handle shape doesn't surface it, fall back to a weights-
    free AutoConfig.from_pretrained(MODEL) — cheap, CPU-only, and authoritative for rope. Only if
    BOTH miss do we return None, and then a cc save raises loudly rather than mis-de-RoPE silently."""
    from cartridges.cartridge import rope_scaling_of, rope_theta_of
    cfg = _engine_hf_config(_get_engine_state())
    theta, scaling = rope_theta_of(cfg), rope_scaling_of(cfg)
    if theta is None:
        try:
            from transformers import AutoConfig
            hf = AutoConfig.from_pretrained(MODEL)
            hf = getattr(hf, "text_config", None) or hf   # multimodal: rope_* on the text tower
            theta, scaling = rope_theta_of(hf), rope_scaling_of(hf)
        except Exception as e:  # noqa: BLE001 — fall through to None; cc save raises if theta needed
            print(f"[onboard_cag] WARN could not read rope config for {MODEL!r}: {e}", flush=True)
    return theta, scaling


def _get_engine_state() -> dict:
    """The warm engine state (async when SERVE_ASYNC, else sync). Building lazily here is the same
    contract every serve route follows; the startup warmup usually has it ready already."""
    return _aget() if SERVE_ASYNC else _get()


def _engine_ready() -> bool:
    """True once the engine core is built. /onboard_cag needs a live engine to harvest KV; if the
    warmup hasn't finished we 503 (consistent with streaming's _require_async and the pre-warm
    behavior of the other routes) rather than block a long build inside the request."""
    return bool(_astate if SERVE_ASYNC else _state)


async def _engine_build_kv(cart_id: str, prompt_ids: list[int]) -> None:
    """Submit ONE build request to the engine: a 1-token greedy generation over `prompt_ids` whose
    SamplingParams.extra_args carries the connector's build key. The connector harvests this
    request's full prompt KV at completion and stages per-TP-rank shards under
    $CARTRIDGE_REGISTRY_DIR/builds/<cart_id>/. We don't care about the generated token — only that
    the request runs to completion so the KV is captured. Works on both engine shapes: the async
    engine yields RequestOutputs, the sync LLM.generate returns a list; both are driven off the
    threadpool by the caller for the sync path."""
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=1, temperature=0.0,
                        extra_args={"cartridge_build_cart_id": cart_id})
    st = _get_engine_state()
    if SERVE_ASYNC:
        rid = uuid.uuid4().hex
        async for _out in st["engine"].generate(_tokens_prompt(prompt_ids), sp, rid):
            pass
    else:
        # Sync LLM: serialize on the serve lock so vLLM's sequential request ids stay consistent
        # with the query path sharing this engine (mirrors _generate's discipline).
        def _run():
            with _lock:
                st["llm"].generate([_tokens_prompt(prompt_ids)], sp, use_tqdm=False)
        await run_in_threadpool(_run)


def _persist_cart_from_shards(cart_id: str, doc_id: str, token_ids: list[int],
                              model_ref: str, rope_theta, rope_scaling) -> int:
    """Merge the staged per-rank KV shards for `cart_id`, wrap them in a Cartridge, and persist via
    the same store write app.py uses. Returns the cart's kv-token count (p) for the response stats.
    collect_rank_shards blocks until every TP rank has staged its shard (or times out), merges them
    to a full-head KV, and cleans up the staging dir; cart_from_kv stamps token_ids/model_ref/rope so
    the served-side binding gate + cc de-RoPE both hold. Runs off the event loop (blocking torch)."""
    from cartridges.serve.engine_onboard import cart_from_kv, collect_rank_shards
    kv = collect_rank_shards(cart_id, REGISTRY_DIR, TENSOR_PARALLEL)
    cart = cart_from_kv(kv, doc_id, token_ids=token_ids, model_ref=model_ref,
                        rope_theta=rope_theta, rope_scaling=rope_scaling)
    _get_engine_state()["store"].put(cart_id, cart, store_dtype=CART_STORE_DTYPE)
    return cart.num_kv_tokens()


async def onboard_cag_via_engine(corpus_dir: str, docs: list[dict], *, build_index: bool = False,
                                 report=None, should_cancel=None) -> dict:
    """Build one CAG cart per doc THROUGH the running engine and persist it to the cartridge store —
    the engine-side mirror of app.py's onboard_cag_corpus, with an IDENTICAL response shape so the
    control plane stores n_cartridges / seconds / corpus_tokens unchanged.

    Per doc: tokenize with the served tokenizer (add_special_tokens=False, truncated to CAG_MAX_DOC_TOK
    — the SAME budget the transformers path uses), submit to the engine with the build key, collect
    the staged shards, build + persist the cart. cart_id == d["doc_id"] exactly as app.py derives it
    (the control plane already hands us the tenant-namespaced doc id; app.py uses it directly as the
    store key). Idempotent by doc_id (a cart already in the store is reused, no engine call), and
    FORCE_REONBOARD=1 rebuilds. Bounded concurrency (~ONBOARD_ENGINE_CONCURRENCY in flight; the engine
    batches internally). One doc's failure is isolated into the response's per-doc errors and never
    fails the batch. build_index is accepted for schema parity; the engine box builds no fused index
    (the retrieval index is the transformers/app.py path's concern), so it is a no-op here."""
    report = report or (lambda *a, **k: None)
    should_cancel = should_cancel or (lambda: False)
    st = _get_engine_state()
    tok, store = st["tok"], st["store"]
    model_ref = os.environ.get("CARTRIDGE_SERVED_MODEL_ID") or MODEL
    rope_theta, rope_scaling = _onboard_rope()
    force = os.environ.get("FORCE_REONBOARD", "0") == "1"
    t0 = time.perf_counter()
    cart_build_s = 0.0  # wall-clock ACTUALLY building carts (excludes idempotent reuse)

    valid = [d for d in docs if d.get("text", "").strip()]
    corpus_tokens = 0
    prepared: list[tuple[dict, list[int]]] = []   # (doc, token_ids) for docs that need building
    n_skipped = 0
    for d in valid:
        ids = tok(d["text"], add_special_tokens=False).input_ids[:CAG_MAX_DOC_TOK]
        corpus_tokens += len(ids)
        if not force and store.exists(d["doc_id"]):
            n_skipped += 1
            continue
        prepared.append((d, ids))

    n_done = n_skipped                 # carts now in the store (reused + built)
    errors: dict[str, str] = {}        # doc_id -> error string (per-doc isolation, same shape as batch reporting)
    n_valid = len(valid)
    build_lock = asyncio.Lock()        # serialize the shared counters + store puts across tasks
    sem = asyncio.Semaphore(ONBOARD_ENGINE_CONCURRENCY)
    canceled = {"v": False}

    report(0.02, None, f"Onboarding {n_valid} document(s) via engine"
                       + (f" ({n_skipped} reused)" if n_skipped else ""))

    async def _one(doc: dict, ids: list[int]) -> None:
        nonlocal n_done, cart_build_s
        if canceled["v"] or should_cancel():
            canceled["v"] = True
            return
        cart_id = doc["doc_id"]        # == the tenant-namespaced id; app.py's store key derivation
        async with sem:
            if canceled["v"]:
                return
            _bs = time.perf_counter()
            try:
                await _engine_build_kv(cart_id, ids)
                await run_in_threadpool(_persist_cart_from_shards, cart_id, doc["doc_id"], ids,
                                        model_ref, rope_theta, rope_scaling)
            except Exception as exc:  # noqa: BLE001 — one bad doc must not sink the batch
                errors[doc["doc_id"]] = f"{type(exc).__name__}: {exc}"
                print(f"[onboard_cag] WARN doc {doc['doc_id']!r} failed: {exc}", flush=True)
                return
            async with build_lock:
                n_done += 1
                cart_build_s += time.perf_counter() - _bs
                report(0.02 + 0.96 * min(n_done, n_valid) / max(n_valid, 1), None,
                       f"Onboarded {n_done} cartridge(s)"
                       + (f" ({n_skipped} reused)" if n_skipped else ""))

    await asyncio.gather(*(_one(d, ids) for d, ids in prepared))

    n_built = n_done - n_skipped
    if canceled["v"]:
        return {"n_cartridges": n_done, "canceled": True, "method": "cag_engine",
                "train_seconds": round(time.perf_counter() - t0, 1),
                "cart_seconds": round(cart_build_s, 1), "n_built": n_built,
                "corpus_tokens": corpus_tokens, "errors": errors}
    return {"n_cartridges": n_done, "canceled": False, "method": "cag_engine",
            "train_seconds": round(time.perf_counter() - t0, 1),
            "cart_seconds": round(cart_build_s, 1), "n_built": n_built,
            "corpus_tokens": corpus_tokens, "errors": errors}


if _HAS_FASTAPI:
    app = FastAPI(title="Cartridge vLLM Inference Service")

    from fastapi.responses import JSONResponse
    from starlette.requests import Request as _Request

    @app.middleware("http")
    async def _ml_auth(request: _Request, call_next):
        """Shared-token gate (DEFAULT OFF). ML_AUTH_TOKEN unset/empty -> today's behavior (open).
        Set -> every route except the liveness/readiness paths needs a correct Bearer token; a
        missing/wrong one gets 401 before the handler runs. Read live from the env (not the
        import-time value) so a test or deploy that sets the env after import still takes effect."""
        token = os.environ.get("ML_AUTH_TOKEN", "")
        if token and request.url.path not in _AUTH_OPEN_PATHS:
            if not _bearer_ok(request.headers.get("authorization"), token):
                return JSONResponse({"detail": "missing or invalid ML auth token"}, status_code=401)
        return await call_next(request)

    class QueryReq(BaseModel):
        doc_ids: list[str]        # which cartridge(s) to serve (control plane retrieved these)
        question: str
        max_tokens: int = 64
        history: list[dict] = []  # prior turns [{role, content}] — small prefill atop resident KV

    class RagReq(BaseModel):
        context: str              # the retrieved documents RAG re-prefills (control plane assembled these)
        question: str
        max_tokens: int = 64
        history: list[dict] = []

    from fastapi.responses import StreamingResponse

    @app.on_event("startup")
    def _warm_engine():
        """Build the vLLM engine NOW (background thread), not on the first user query — the
        cold build takes minutes, which blows through ALB/client timeouts and (pre-lock) let
        concurrent first-queries race two engine cores onto one GPU.

        FAIL LOUD, NOT ZOMBIE: if the build throws (partial weight files mid-seed, a worker
        shm race, an NCCL flake), a swallowed exception used to leave the service answering
        /health with engine_ready:false FOREVER — indistinguishable from a slow warm, found
        live when a provision restarted serve during the FS weight seed. Exit the process
        instead: systemd (Restart=on-failure, RestartSec=15) relaunches a CLEAN process with
        no leaked CUDA/worker state, and the restart loop keeps retrying until the build
        succeeds. In-process retry was rejected — a failed engine build can leave zombie
        worker procs holding GPU memory that would OOM the retry."""
        if os.environ.get("SERVE_WARMUP", "1") != "1":
            return

        def _warm_or_die():
            try:
                (_aget if SERVE_ASYNC else _get)()
            except Exception:  # noqa: BLE001 — anything fatal to the build
                import traceback
                print("[warm] ENGINE BUILD FAILED — exiting so systemd restarts a clean "
                      "process:\n" + traceback.format_exc(), flush=True)
                os._exit(3)

        threading.Thread(target=_warm_or_die, daemon=True).start()

    @app.get("/health")
    def health():
        return {"ok": True, "model": MODEL, "async": SERVE_ASYNC,
                "engine_ready": bool(_astate if SERVE_ASYNC else _state),
                "store": _store_extra().get("cart_store_backend"),
                "knobs": _knob_state()}

    @app.get("/stats")
    def stats():
        """Fleet-facing serving stats. The connector lives in the EngineCore subprocess, so its
        counters arrive via the JSON mirrors it writes to the shared registry dir (one per
        connector role/process: cache hits/misses = the multiplexing story, load latency = the
        store hydration cost). Frontend-side state is reported directly."""
        import glob as _glob
        import json as _json
        connectors = []
        for f in sorted(_glob.glob(str(Path(REGISTRY_DIR) / "connector_stats-*.json"))):
            try:
                connectors.append(_json.loads(Path(f).read_text()))
            except (OSError, ValueError):
                continue
        # Roll-up across worker PIDs: under TP/multi-process the connector runs in the
        # scheduler AND each worker, so these counters are SPLIT across several mirror
        # files (e.g. the degrade signal — load_errors/registry_misses/degraded_requests —
        # lands in whichever process hit it). Sum the numeric counters so the operator gets
        # the whole-request health in one place. Missing keys are tolerated (older mirrors
        # predate some counters); non-numeric fields (role/pid/last_load_ms) are skipped.
        _SUM_KEYS = ("cart_loads", "cache_hits", "cache_evictions", "load_seconds_total",
                     "requests_served", "load_errors", "registry_misses", "degraded_requests",
                     "invalidations")
        totals = {k: 0 for k in _SUM_KEYS}
        for c in connectors:
            for k in _SUM_KEYS:
                v = c.get(k)
                if isinstance(v, (int, float)):
                    totals[k] += v
        totals["load_seconds_total"] = round(totals["load_seconds_total"], 3)
        return {"model": MODEL, "async": SERVE_ASYNC, "tensor_parallel": TENSOR_PARALLEL,
                "store": _store_extra().get("cart_store_backend"),
                "knobs": _knob_state(),
                "frontend": {"kv_len_cache_entries": len(_kv_len)},
                "totals": totals, "connectors": connectors}

    class InvalidateReq(BaseModel):
        cart_ids: list[str]      # ids to purge from serving (deleted, or force-rebuilt in place)

    def _registry():
        """The frontend's registry handle: the built engine state's instance when one exists,
        else a fresh env-configured one — /invalidate must work BEFORE the first query builds
        an engine (an offboard right after boot), and building a registry needs no GPU."""
        st = _astate if SERVE_ASYNC else _state
        if st:
            return st["reg"]
        from cartridges.serve.vllm_cartridge_connector import make_request_registry
        return make_request_registry()

    @app.post("/invalidate")
    async def invalidate(req: InvalidateReq):
        """Serving-side half of the data-deletion path (DATA_LIFECYCLE.md, engram-dynamics-landing repo): purge this
        frontend's per-cart caches (_kv_len/_cart_ids/_cart_binding_ok), evict this box's
        store MIRROR copies (mirror-first reads would otherwise resurrect a deleted cart —
        or keep serving the OLD blob after a force re-onboard), publish registry tombstones
        that every EngineCore-subprocess connector (scheduler + each TP-rank worker) polls
        and purges on within ~CARTRIDGE_INVALIDATE_POLL_S of its next request, and
        best-effort reset vLLM's automatic prefix cache (under CART_PLACEHOLDER=real, a
        re-onboarded cart's unchanged leading blocks could otherwise APC-hit blocks whose
        KV was injected from the old blob). Durable-blob deletion is the control plane's
        job (ml_service POST /offboard); the two calls together are the complete deletion.
        Idempotent — ids never served (or already purged) tombstone harmlessly. NOTE: the
        frontend memo purge is per-replica; a multi-replica frontend fleet must fan this
        call out (the ENGINE purge fans out via tombstones regardless — a stale memo alone
        cannot serve deleted KV, it only mis-predicts a 404, and the epoch guard plus the
        engine's containment turn that into a degraded/failed request, never stale text)."""
        from cartridges.cart_store import validate_cart_id
        try:
            for cid in req.cart_ids:
                validate_cart_id(cid)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        reg = _registry()
        st = _astate if SERVE_ASYNC else _state
        store = st["store"] if st else None
        if store is None:
            from cartridges.serve.vllm_cartridge_connector import _store_from_extra
            store = _store_from_extra(_store_extra())
        _inval_epoch[0] += 1   # BEFORE the pops: _cart_meta must refuse to re-memoize
        for cid in req.cart_ids:
            _kv_len.pop(cid, None)
            _cart_ids.pop(cid, None)
            _cart_binding_ok.pop(cid, None)
            try:
                store.evict_local(cid)
            except Exception as e:  # noqa: BLE001 — evict is best-effort; tombstones still go out
                print(f"[invalidate] WARN mirror evict failed for {cid!r}: {e}", flush=True)
            reg.request_invalidation(cid)
        if st:
            # Best-effort APC reset on the built engine (v1 exposes reset_prefix_cache on
            # both the sync and async engines; the async form returns an awaitable).
            try:
                eng = st.get("engine") or st.get("llm")
                fn = (getattr(eng, "reset_prefix_cache", None)
                      or getattr(getattr(eng, "llm_engine", None), "reset_prefix_cache", None))
                if fn is not None:
                    r = fn()
                    if inspect.isawaitable(r):
                        await r
            except Exception as e:  # noqa: BLE001 — never fail the deletion on a cache reset
                print(f"[invalidate] WARN prefix-cache reset failed: {e}", flush=True)
        return {"invalidated": len(req.cart_ids), "backend": reg.backend}

    @app.post("/query")
    async def query(req: QueryReq):
        try:
            if SERVE_ASYNC:                       # batched concurrent serving (measured, incl. queueing)
                result = await serve_query_async(req.doc_ids, req.question, req.max_tokens,
                                                 req.history)
            else:                                 # proven one-at-a-time path (measured)
                # serve_query blocks on the GPU for the whole generation; run it in
                # the threadpool so the event loop (and /health) stays responsive.
                result = await run_in_threadpool(
                    serve_query, req.doc_ids, req.question, req.max_tokens, req.history)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except LookupError as e:
            raise HTTPException(404, str(e)) from e
        return {**result, "doc_ids": req.doc_ids}

    class DescribeReq(BaseModel):
        doc_ids: list[str]        # carts to describe (control plane onboarded these)
        max_tokens: int = 60

    @app.post("/describe")
    async def describe(req: DescribeReq):
        """Generate a one-sentence description PER doc_id from its resident CAG cart — same
        resolve/serve path as /query (each doc's cart is served against a FIXED instruction, so the
        model reads only that document's KV). Returns {descriptions: {doc_id: text-or-null}}. Per-doc
        failures (missing cart, GPU error) map that id to null and NEVER fail the batch — the control
        plane's describe pass is best-effort and onboarding must still succeed. The async engine
        batches the concurrent per-doc requests; the sync path runs them one at a time in a
        threadpool so the event loop stays responsive."""
        async def _one(doc_id: str) -> str | None:
            try:
                if SERVE_ASYNC:
                    out = await serve_query_async([doc_id], _DESCRIBE_INSTRUCTION, req.max_tokens)
                else:
                    out = await run_in_threadpool(
                        serve_query, [doc_id], _DESCRIBE_INSTRUCTION, req.max_tokens)
                text = (out.get("answer") or "").strip()
                return text or None
            except Exception as e:  # noqa: BLE001 — one bad cart must not sink the batch
                print(f"[describe] WARN doc {doc_id!r} failed: {e}", flush=True)
                return None

        results = await asyncio.gather(*(_one(d) for d in req.doc_ids))
        return {"descriptions": dict(zip(req.doc_ids, results))}

    def _safe_corpus_dir_or_400(corpus_dir: str) -> str:
        """Mirror of app.py's corpus_dir confinement (kept local — importing app.py would drag its
        transformers stack into this process). Same allowlist semantics: ML_ALLOWED_CORPUS_ROOTS
        (comma-separated) overrides; default is PLATFORM_DATA_DIR (when set) and /data."""
        env = os.environ.get("ML_ALLOWED_CORPUS_ROOTS", "")
        if env.strip():
            roots = [Path(p).resolve() for p in env.split(",") if p.strip()]
        else:
            roots = [Path(os.environ.get("PLATFORM_DATA_DIR", "/data")).resolve(), Path("/data").resolve()]
        p = Path(corpus_dir).resolve()
        for root in roots:
            if p == root or p.is_relative_to(root):
                return str(p)
        raise HTTPException(400, f"corpus_dir outside the allowed data roots: {corpus_dir}")

    class OnboardCagReq(BaseModel):
        # Schema IDENTICAL to app.py's OnboardCagReq — app.py proxies the body verbatim, so this
        # must accept exactly the same fields (any drift breaks the control-plane contract).
        corpus_dir: str
        docs: list[dict]                 # [{doc_id, text}]
        build_index: bool = False        # accepted for parity; the engine box builds no fused index
        job_id: str | None = None
        progress_url: str | None = None
        progress_token: str | None = None

    @app.post("/onboard_cag")
    async def onboard_cag(req: OnboardCagReq):
        """Build CAG carts for a corpus THROUGH the running vLLM engine (the only process that can
        produce this model's KV) and persist them to the cartridge store the serve path reads from.
        Same request/response shape as app.py's /onboard_cag; app.py proxies here when
        ONBOARD_VIA_ENGINE is set, so the control plane sees no difference (progress reporting
        included). Engine not yet warm -> 503 (consistent with the pre-warm behavior of the serve
        routes); the control plane can retry once the box reports engine_ready."""
        if not req.docs:
            raise HTTPException(400, "no documents")
        # Defense-in-depth corpus_dir confinement (2026-09 security sweep M1): this endpoint never
        # writes under corpus_dir today (carts persist by cart_id through the store), but the
        # contract says confinement holds on BOTH onboarding paths — and a future filesystem use
        # of the field must not become a traversal. Mirrors app.py's allowlist exactly.
        _safe_corpus_dir_or_400(req.corpus_dir)
        if not _engine_ready():
            raise HTTPException(503, "engine not ready; retry once the box reports engine_ready")
        report, cancel_event = _make_reporter(req.progress_url, req.progress_token)
        return await onboard_cag_via_engine(
            req.corpus_dir, req.docs, build_index=req.build_index,
            report=report, should_cancel=cancel_event.is_set)

    _SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

    def _require_async() -> dict:
        """Streaming needs the async engine. On a SERVE_ASYNC=0 box the warmed engine is the
        SYNC one, and calling _aget() here would lazily build a SECOND engine core beside it —
        the same 'Free memory ...' OOM the build-lock comment documents. Fail loud instead."""
        if not SERVE_ASYNC:
            raise HTTPException(409, "streaming requires the async engine; start with SERVE_ASYNC=1")
        return _aget()

    @app.post("/query_stream")
    async def query_stream(req: QueryReq):
        """Token-streaming cart serve (SSE). Requires the async engine (SERVE_ASYNC=1)."""
        st = _require_async()
        prompt, q_len, total_p = _prompt_ids(st["tok"], st["store"], req.doc_ids,
                                             req.question, req.history)
        return StreamingResponse(
            _stream_generate(prompt, list(req.doc_ids), req.max_tokens, q_len, total_p,
                             want_conf=True),   # cart side reports confidence for the adaptive router
            media_type="text/event-stream", headers=_SSE_HEADERS)

    @app.post("/rag_query_stream")
    async def rag_query_stream(req: RagReq):
        """Token-streaming RAG baseline (SSE), same engine — honest side-by-side streaming."""
        st = _require_async()
        tok = st["tok"]
        user = (f"Documents:\n\n{req.context}\n\nQuestion: {req.question}"
                if req.context else req.question)
        ids = _chat_ids(tok, user, req.history)
        return StreamingResponse(
            _stream_generate(ids, None, req.max_tokens, len(ids), 0),
            media_type="text/event-stream", headers=_SSE_HEADERS)

    @app.post("/rag_query")
    async def rag_query(req: RagReq):
        """Live RAG baseline (measured on the same engine) — no cart, prefill the retrieved
        context + question. Routes to whichever engine this box runs: the async engine when
        SERVE_ASYNC=1 (the old unconditional sync call built a SECOND engine core beside it),
        threadpool-wrapped sync otherwise (off the event loop)."""
        try:
            if SERVE_ASYNC:
                return await serve_rag_async(req.context, req.question, req.max_tokens,
                                             req.history)
            return await run_in_threadpool(serve_rag, req.context, req.question,
                                           req.max_tokens, req.history)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e


def _selftest(workdir: str) -> None:
    """End-to-end service check on a GPU box: load the build-corpus carts and serve each doc's question
    (+ a routing cross-check). Confirms the request->registry->generate path serves the right cart."""
    import json
    os.environ.setdefault("CARTRIDGE_STORE_DIR", str(Path(workdir) / "store"))
    manifest = json.loads((Path(workdir) / "corpus_manifest.json").read_text())
    docs = manifest["docs"]
    keys = list(docs)
    for doc_id in keys:
        out = serve_query([doc_id], docs[doc_id]["question"])
        print(f"[selftest:{doc_id}] {docs[doc_id]['question']}\n  -> {out['answer']!r}  {out['metrics']}")
    if len(keys) >= 2:                          # routing cross-check: doc-a's Q, doc-b's cart
        out = serve_query([keys[1]], docs[keys[0]]["question"])
        print(f"[selftest:routing-check] {keys[0]}'s Q served with {keys[1]}'s cart -> {out['answer']!r} "
              f"(should answer {keys[1]}'s fact, not {keys[0]}'s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--workdir", default="/opt/work")
    args = ap.parse_args()
    if args.selftest:
        _selftest(args.workdir)
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8002")))
