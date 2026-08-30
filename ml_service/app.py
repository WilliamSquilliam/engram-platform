"""ML service — the GPU worker + inference server from the C2/C3 architecture.

Wraps the proven `cartridges` primitives behind a small HTTP API so the
control plane stays torch-free. Two endpoints:

  POST /train  {corpus_dir, docs:[{doc_id,text}], ...}  -> trains one cartridge
               per document (generated self-study + mixed-visibility KL loop) and
               saves them under corpus_dir/cartridges/.
  POST /query  {corpus_dir, question, k}                -> TF-IDF-selects the top-k
               cartridges, generates an answer with them prefixed, returns text.

Local default model is small (Qwen3-0.6B) so the whole flow runs in ~minutes on a
laptop GPU/CPU; set CARTRIDGES_MODEL to scale up. This process is the only one that
imports torch — it maps to the "Training Worker" + "Inference Service" containers
and would split into those two on AWS (EKS GPU node group).
"""
from __future__ import annotations

import gc
import hmac
import json
import os
import random
import threading
import time
import traceback
from pathlib import Path

# `cartridges` is a pip dependency (engram-cartridge); import it normally.

# Environment management: load the platform env file BEFORE importing
# cartridges.config (which resolves CARTRIDGES_MODEL at import time). Local
# auto-loads .env.local; a .env overrides it; real env (compose/ECS) always wins.
_PLATFORM_DIR = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    # Layered, override=False so the first value set wins: real process env >
    # .env (gitignored, your secrets/overrides) > .env.local (committed defaults).
    load_dotenv(_PLATFORM_DIR / ".env")
    load_dotenv(_PLATFORM_DIR / ".env.local")
except ImportError:
    pass

import httpx  # noqa: E402
import torch  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from cartridges.budget_manager import BudgetManager  # noqa: E402
from cartridges.config import DEFAULT_MODEL, Cfg  # noqa: E402
from cartridges.encoder import (  # noqa: E402
    encode_to_cartridge,
    encode_to_cartridge_perlayer,
    load_encoder,
)
from cartridges.infer import (  # noqa: E402
    DenseRetriever,
    TfidfRetriever,
    answer_question,
    load_carts_from,
    select_carts,
)
from cartridges.model_patch import (  # noqa: E402
    forward_with_cartridges,
    generate_with_cartridges,
)
from cartridges.self_study import (  # noqa: E402
    encode_pair,
    generate_questions,
    question_prompt,
    teacher_answer,
)
from cartridges.train import (  # noqa: E402
    initialize_cartridges,
    sparse_kl_loss,
)

# Base model: single source of truth is cartridges.config.DEFAULT_MODEL
# (env CARTRIDGES_MODEL; production default Qwen/Qwen3-30B-A3B). Locally, start
# this service with CARTRIDGES_MODEL=Qwen/Qwen3-1.7B (or 0.6B) — a laptop can't
# hold 30B. Compute dtype is its own knob (bf16 default; fp16 for older GPUs); for
# tight VRAM point CARTRIDGES_MODEL at a pre-quantized repo (e.g. an AWQ/GPTQ/FP8
# build of the same model).
MODEL_NAME = DEFAULT_MODEL
_DTYPE = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
          "float16": torch.float16, "fp16": torch.float16}
COMPUTE_DTYPE = _DTYPE.get(os.environ.get("CARTRIDGES_DTYPE", "bfloat16").lower(), torch.bfloat16)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# WEIGHT-MATCHED onboarding (opt-in; default '' = load CARTRIDGES_MODEL exactly as today).
# Set ONBOARD_WEIGHTS to an HF repo id (e.g. RedHatAI/Qwen3-30B-A3B-FP8-dynamic) to onboard AGAINST
# the fp8 serve weights instead of the bf16 base, so a cart's KV matches the tokens the fp8 serve
# engine will actually produce. The cart is stamped model_ref accordingly (see _onboard_model_ref)
# so a frontend can reject a cart built under weights that don't match the serve tier.
ONBOARD_WEIGHTS = os.environ.get("ONBOARD_WEIGHTS", "").strip()


def _onboard_model_ref() -> str:
    """The model_ref stamped on every cart this service builds. Names the weights AND how they were
    loaded: '#dequant-bf16' when we onboard a compressed-tensors FP8 checkpoint (loaded with
    run_compressed=False, so the forward runs in the compute dtype), '#bf16' for the default base."""
    if ONBOARD_WEIGHTS:
        return f"{ONBOARD_WEIGHTS}#dequant-bf16"
    return f"{MODEL_NAME}#bf16"

_model = None
_tokenizer = None
_lock = threading.Lock()  # serialize GPU work; one model, one device

# Retrieval gets its OWN lock so a chat turn's /retrieve doesn't queue behind an hour-long
# /onboard_cag or /train holding the global _lock. Retrieval still serializes against itself
# (the fused encoders are shared, mutable ST state). WHY only narrow, not remove: the historical
# embedding OOM (see _unload_model, ~2026-07-05, 226 MiB free) means running /retrieve's encoders
# concurrently with an onboarding embed can exhaust VRAM on a small GPU — RETRIEVE_EXCLUSIVE=1
# restores the old global-lock serialization for those boxes. Default '0' (narrowed) is safe in the
# co-located deployment: the vLLM serve container already shares this GPU concurrently with us, so
# cross-process GPU concurrency exists regardless of which lock /retrieve takes.
_retr_lock = threading.Lock()
RETRIEVE_EXCLUSIVE = os.environ.get("RETRIEVE_EXCLUSIVE", "0") == "1"

# Progress-bar bookkeeping: self-study/prep occupies the first _PREP_FRAC of the
# bar, the training loop the rest (up to ~0.97), saving the tail.
_PREP_FRAC = 0.15

# Teacher distillation targets are stored sparsely: only the top-K vocab
# log-probs per answer token (the teacher is sharply peaked, so K≈64 captures
# essentially all the mass). Storing the full V≈152k distribution for every QA
# is what OOM'd the host at paper scale. Tunable via TEACHER_TOPK.
TEACHER_TOPK = int(os.environ.get("TEACHER_TOPK", "64"))


class _Cancelled(Exception):
    """Raised internally when the control plane requests training cancellation."""


# --- corpus_dir confinement -------------------------------------------------
# The API takes corpus_dir from the request and reads/writes under it (including
# deleting stale *.pt). This service is internal-only, but defense-in-depth:
# refuse any corpus_dir outside the known data roots so a compromised/misrouted
# caller can't read or delete files elsewhere on the box.
# Roots: ML_ALLOWED_CORPUS_ROOTS (comma-separated) overrides; default covers the
# native local layout (platform/.data), PLATFORM_DATA_DIR when set, and /data
# (the docker-compose / ECS volume mount).
def _allowed_corpus_roots() -> list[Path]:
    env = os.environ.get("ML_ALLOWED_CORPUS_ROOTS", "")
    if env.strip():
        return [Path(p).resolve() for p in env.split(",") if p.strip()]
    roots = [Path(os.environ.get("PLATFORM_DATA_DIR", _PLATFORM_DIR / ".data")), Path("/data")]
    return [r.resolve() for r in roots]


def _safe_corpus_dir(corpus_dir: str) -> str:
    p = Path(corpus_dir).resolve()
    for root in _allowed_corpus_roots():
        if p == root or p.is_relative_to(root):
            return str(p)
    raise HTTPException(400, f"corpus_dir outside the allowed data roots: {corpus_dir}")


def _fp8_dequant_kwargs(weights: str) -> dict:
    """Extra from_pretrained kwargs to load a compressed-tensors FP8 checkpoint DEQUANTIZED to the
    compute dtype (run_compressed=False). Onboarding needs a real bf16/fp16 forward to extract KV
    (llmcompressor block-FP8 has no dequantized cart path otherwise), and run_compressed=False
    sidesteps the transformers#42915 block-FP8 MoE shape bug that fires when the weights stay packed.
    Only fp8 weight ids get this; the default bf16 base gets a plain load (empty dict)."""
    if weights and "fp8" in weights.lower():
        # transformers >=5 requires the CONFIG CLASS — a plain dict raises "quantized with
        # CompressedTensorsConfig but you are passing a dict config" (hit live, tf 5.14).
        from transformers import CompressedTensorsConfig
        return {"quantization_config": CompressedTensorsConfig(run_compressed=False)}
    return {}


def get_model():
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # ONBOARD_WEIGHTS (opt-in) loads a DIFFERENT repo than the served CARTRIDGES_MODEL — the
        # weight-matched onboarding path (fp8 serve weights). Default '' keeps loading MODEL_NAME.
        weights = ONBOARD_WEIGHTS or MODEL_NAME
        _tokenizer = AutoTokenizer.from_pretrained(weights)
        # For a compressed-tensors FP8 checkpoint, ask transformers to DEQUANTIZE to the compute
        # dtype (run_compressed=False). fp8_kw is empty for the (default) bf16 base, so that path is
        # byte-identical to before. `_from_pretrained` tries the fp8-aware kwargs and, if this
        # transformers is too old to accept quantization_config as a dict (TypeError) or trips the
        # block-FP8 MoE shape bug (ValueError), falls back to a plain load — then, since a bare load
        # of an fp8 repo can't produce a usable bf16 forward, raises a clear RuntimeError naming the
        # transformers version needed rather than letting a cryptic conversion crash surface.
        fp8_kw = _fp8_dequant_kwargs(weights)

        def _from_pretrained(**base):
            try:
                return AutoModelForCausalLM.from_pretrained(weights, **base, **fp8_kw)
            except (TypeError, ValueError) as e:
                if not fp8_kw:
                    raise  # not the fp8 path — a real load error, propagate as before
                raise RuntimeError(
                    f"onboarding weights {weights!r} are a compressed-tensors FP8 checkpoint, but "
                    f"this transformers could not DEQUANTIZE them (run_compressed=False). This needs "
                    f"a transformers with the compressed-tensors integration that handles block-FP8 "
                    f"MoE (>=4.56 with the transformers#42915 fix, or a build where run_compressed "
                    f"is accepted). Pin ONBOARD_WEIGHTS to a bf16 repo, or upgrade transformers. "
                    f"Underlying error: {type(e).__name__}: {e}") from e

        # CARTRIDGES_DEVICE_MAP shards a model too big for one GPU (e.g. Qwen3-30B-A3B
        # ~60GB bf16) across all visible GPUs via accelerate. When set, accelerate's
        # dispatch hooks place inputs on the right shard, so we must NOT call
        # .to(DEVICE). Empty/unset (the 8B default) = single-device load as before.
        device_map = os.environ.get("CARTRIDGES_DEVICE_MAP", "").strip()
        if device_map:
            # A device_map load places layers across GPUs by *free* memory. If the GPUs are
            # already largely occupied — e.g. a co-resident vLLM serve engine holding ~15 GB/GPU
            # of 30B weights on a 46 GB card — accelerate can't place the last few layers and
            # transformers 5.x mislabels them "missing from the checkpoint," raising a cryptic
            # conversion error (measured on the 30B demo box 2026-07-09). Surface the real cause
            # (VRAM exhaustion) with actionable guidance, and NEVER build carts from a partially
            # placed model. At 30B, onboard on dedicated/idle GPUs (the decoupled pre-onboard
            # pattern) or before the serve engine warms, or lower VLLM_GPU_MEM_UTIL.
            import torch
            try:
                _model = _from_pretrained(
                    dtype=COMPUTE_DTYPE, attn_implementation="sdpa", device_map=device_map,
                ).eval()
            except RuntimeError:
                raise  # the fp8-dequant guidance above — don't mask it as VRAM exhaustion
            except Exception as e:  # noqa: BLE001 — re-raise with the real (VRAM) cause
                free_gb = (sum(torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count()))
                           / 1e9) if torch.cuda.is_available() else 0.0
                raise RuntimeError(
                    f"onboarding could not load {weights} across {torch.cuda.device_count()} GPU(s) "
                    f"({free_gb:.0f} GB free). At 30B this is almost always VRAM exhaustion from a "
                    f"co-resident serve engine (transformers reports it as 'weights missing from the "
                    f"checkpoint'). Onboard on dedicated/idle GPUs or lower VLLM_GPU_MEM_UTIL. "
                    f"Underlying error: {type(e).__name__}: {e}") from e
            if any(p.device.type == "meta" for p in _model.parameters()):
                _model = None  # partial placement -> would yield garbage carts; refuse
                raise RuntimeError(
                    f"{weights} loaded only partially (params left unplaced on 'meta') — treat as "
                    f"out-of-VRAM; onboard on dedicated/idle GPUs. Refusing to build carts from an "
                    f"incompletely loaded model.")
        else:
            _model = (
                _from_pretrained(dtype=COMPUTE_DTYPE, attn_implementation="sdpa")
                .to(DEVICE)
                .eval()
            )
        for p in _model.parameters():
            p.requires_grad = False
    return _model, _tokenizer


# Amortized encoder (the production onboarding path): trained ONCE, then compresses any doc
# into a cartridge in ~2 frozen forward passes — no per-doc gradient descent. Loaded lazily and
# cached by checkpoint path. ENCODER_CKPT (or the request's encoder_ckpt) points at an encoder.pt
# trained for THIS base model via `python -m cartridges.encoder`.
_encoder = None
_encoder_variant = None
_encoder_path = None


def get_encoder(ckpt_path: str):
    global _encoder, _encoder_variant, _encoder_path
    if _encoder is None or _encoder_path != ckpt_path:
        model, _ = get_model()
        _encoder, _encoder_variant = load_encoder(ckpt_path, model)
        _encoder_path = ckpt_path
    return _encoder, _encoder_variant


def onboard_with_encoder(corpus_dir, docs, encoder_ckpt=None, report=None):
    """Amortized-encoder onboarding: materialize one cartridge per doc in ~2 frozen forward
    passes each (NO per-doc gradient descent — the ~54× cheaper path, COST_COMPARISON.md §2).
    Same on-disk output as train_corpus (corpus_dir/cartridges/*.pt + docs.json), so query and
    compare work identically. Requires a pre-trained encoder checkpoint."""
    report = report or (lambda *a, **k: None)
    ckpt = encoder_ckpt or os.environ.get("ENCODER_CKPT", "")
    if not ckpt or not Path(ckpt).exists():
        raise HTTPException(
            400,
            "encoder onboarding requested but no encoder checkpoint found — set ENCODER_CKPT "
            "(or pass encoder_ckpt) to an encoder.pt trained for this base model "
            "(`python -m cartridges.encoder`).",
        )
    model, tokenizer = get_model()
    encoder, variant = get_encoder(ckpt)
    materialize = encode_to_cartridge_perlayer if variant == "perlayer" else encode_to_cartridge
    t0 = time.time()
    corpus_tokens = sum(_ntok(tokenizer, d["text"]) for d in docs)
    cart_dir = Path(corpus_dir) / "cartridges"
    cart_dir.mkdir(parents=True, exist_ok=True)
    for old in cart_dir.glob("*.pt"):
        old.unlink()
    n = len(docs)
    report(0.02, None, f"Encoding {n} documents (amortized encoder)")
    for i, d in enumerate(docs):
        cart = materialize(encoder, model, tokenizer, d["text"], doc_id=d["doc_id"])
        cart.save(cart_dir / f"{d['doc_id']}.pt")
        report(min(0.98, (i + 1) / n), None, f"Encoding documents ({i + 1}/{n})")
    (Path(corpus_dir) / "docs.json").write_text(
        json.dumps({d["doc_id"]: d["text"] for d in docs}), encoding="utf-8"
    )
    secs = round(time.time() - t0, 2)
    return {"n_cartridges": n, "canceled": False, "train_seconds": secs,
            "corpus_tokens": corpus_tokens, "method": "encoder",
            "per_doc_seconds": round(secs / max(1, n), 3)}


# ---------------------------------------------------------------- training
def _self_study(model, tokenizer, docs, gen_qs, ans_max, report, should_cancel=lambda: False):
    """Generated self-study + teacher log-probs for each doc (in memory)."""
    device = next(model.parameters()).device
    caches: dict[str, list[dict]] = {}
    n = len(docs)
    for i, d in enumerate(docs):
        if should_cancel():
            raise _Cancelled()
        text, doc_id = d["text"], d["doc_id"]
        questions = generate_questions(model, tokenizer, text, gen_qs, str(device))
        # Always include a trivial "what does this say" QA so even a weak
        # generation pass leaves the cart with at least one training signal.
        questions = questions or ["What is this document about?"]
        recs = []
        for q in questions:
            a = teacher_answer(model, tokenizer, text, q, ans_max, str(device))
            if not a:
                continue
            answer = a if a.startswith(" ") else " " + a
            enc = encode_pair(tokenizer, text + question_prompt(q), answer)
            ids = torch.tensor([enc["full_ids"]], device=device)
            with torch.no_grad():
                out = model(input_ids=ids, use_cache=False, return_dict=True)
            logits = out.logits[0]
            start = enc["answer_start"] - 1
            end = start + len(enc["answer_ids"])
            lp = torch.log_softmax(logits[start:end].float(), dim=-1)  # (T, V)
            # Keep only the top-K teacher log-probs + their vocab ids (sparse
            # distillation target). Full-V storage here is what OOM'd the host
            # at paper scale: V≈152k * fp16 * every QA token ~ tens of GB.
            k = min(TEACHER_TOPK, lp.shape[-1])
            topk = torch.topk(lp, k, dim=-1)
            recs.append({
                "answer_ids": enc["answer_ids"],
                "teacher_topk_logprobs": topk.values.to(torch.float16).cpu(),  # (T, K)
                "teacher_topk_idx": topk.indices.to(torch.int32).cpu(),         # (T, K)
                "question": q,
            })
        caches[doc_id] = recs
        report(_PREP_FRAC * (i + 1) / n, None, f"Analyzing documents ({i + 1}/{n})")
    return caches


def train_corpus(corpus_dir, docs, cart_tokens, steps, grad_accum, gen_qs,
                 report=None, should_cancel=None):
    """Train one cartridge per doc; save to corpus_dir/cartridges/. Compact
    re-use of the train.py loop primitives (kept here so train.py stays the
    proven CLI path untouched). `report(progress, eta_seconds, detail)` is an
    optional heartbeat the control plane turns into a live progress bar + ETA.
    `should_cancel()` is polled each step; if it returns True we abort early and
    return {"canceled": True}. Returns timing/size used by the cost view."""
    report = report or (lambda *a, **k: None)
    should_cancel = should_cancel or (lambda: False)
    model, tokenizer = get_model()
    device = next(model.parameters()).device
    t0 = time.time()
    # Total corpus size in model tokens — feeds the prefill/break-even economics.
    corpus_tokens = sum(
        len(tokenizer(d["text"], add_special_tokens=False).input_ids) for d in docs
    )

    cfg = Cfg()
    cfg.cart_tokens = cart_tokens
    cfg.num_docs = len(docs)
    # Resident GPU budget. At tiny/local scale keep every cart resident
    # (budget_b = num_docs) so the full k~U(1,k_max) distractor range is
    # realizable. At corpus scale that doesn't fit — N carts each carry params +
    # Adam state (~0.45 GB at cart_tokens=512/fp32 for an 8B model), so we cap
    # the resident pool well below N. The GPU then holds only `budget_b` carts
    # and the BudgetManager rotates the rest through host RAM (paper §2.2).
    budget_cap = int(os.environ.get("TRAIN_BUDGET_B", "16"))
    cfg.budget_b = min(len(docs), budget_cap)
    cfg.total_steps = steps
    cfg.grad_accum = grad_accum
    cfg.k_min = 1
    # k distractors are drawn from the resident pool, so cap at budget_b - 1.
    cfg.k_max = max(1, min(cfg.k_max, cfg.budget_b - 1, len(docs) - 1))

    def _result(n_carts, canceled):
        return {"n_cartridges": n_carts, "canceled": canceled, "method": "train",
                "train_seconds": round(time.time() - t0, 2), "corpus_tokens": corpus_tokens}

    try:
        report(0.01, None, "Preparing self-study")
        caches = _self_study(model, tokenizer, docs, gen_qs,
                             cfg.gen_answer_max_tokens, report, should_cancel)
        corpus = [{"doc_id": d["doc_id"], "doc_text": d["text"]} for d in docs]
        carts = initialize_cartridges(model, tokenizer, corpus, cart_tokens, torch.float32)

        mgr = BudgetManager(carts, cfg)
        rng = random.Random(1234)
        loop_start = time.time()
        report_every = max(1, steps // 50)  # ~50 heartbeats over the run, regardless of length
        train_span = 0.97 - _PREP_FRAC      # training loop occupies _PREP_FRAC..0.97 of the bar
        for step in range(steps):
            if should_cancel():
                raise _Cancelled()
            mgr.global_step = step
            mgr.maybe_rotate()
            mgr.zero_grads()
            touched = {}
            for _ in range(grad_accum):
                relevant, in_cache, _ = mgr.sample_step(rng)
                recs = caches.get(relevant.cart.doc_id)
                if not recs:
                    continue
                qa = rng.choice(recs)
                prompt_ids = tokenizer(question_prompt(qa["question"]), add_special_tokens=False).input_ids
                full_ids = torch.tensor([prompt_ids + qa["answer_ids"]], device=device)
                out = forward_with_cartridges(model, full_ids, [it.cart for it in in_cache])
                s = len(prompt_ids) - 1
                e = s + len(qa["answer_ids"])
                loss = sparse_kl_loss(
                    out.logits[0, s:e],
                    qa["teacher_topk_idx"].to(device),
                    qa["teacher_topk_logprobs"].to(device),
                ) / grad_accum
                loss.backward()
                for it in in_cache:
                    touched[it.cart.doc_id] = it
            mgr.step_carts(touched.values())
            done = step + 1
            if done % report_every == 0 or done == steps:
                elapsed = time.time() - loop_start
                eta = int(elapsed / done * (steps - done))  # avg step time x remaining steps
                report(_PREP_FRAC + train_span * (done / steps), eta,
                       f"Training cartridges — step {done}/{steps}")
    except _Cancelled:
        return _result(0, True)

    report(0.98, 0, "Saving cartridges")
    cart_dir = Path(corpus_dir) / "cartridges"
    cart_dir.mkdir(parents=True, exist_ok=True)
    for old in cart_dir.glob("*.pt"):
        old.unlink()
    mgr.save_all(cart_dir)
    # persist doc text for retrieval at query time
    (Path(corpus_dir) / "docs.json").write_text(
        json.dumps({d["doc_id"]: d["text"] for d in docs}), encoding="utf-8"
    )
    return _result(len(carts), False)


def _strip_lead(answer: str) -> str:
    """Models sometimes echo the 'A:' / 'Answer:' prompt tail — strip it."""
    for lead in ("A:", "Answer:", "answer:"):
        if answer.lstrip().startswith(lead):
            return answer.lstrip()[len(lead):].lstrip()
    return answer


def _ntok(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def _free_cuda() -> None:
    """Release cached GPU memory after a failed/large op so later work can run."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_tokenizer():
    """Tokenizer WITHOUT the model — CPU-only. For paths that only count/slice tokens
    (skip-pass onboarding, cost accounting) and must not pay ~16GB VRAM for the base."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


def _unload_model() -> None:
    """Drop the cached base model and release its VRAM (it reloads lazily on next use).
    The demo box CO-LOCATES this model, the fused-index encoders AND the vLLM serve
    engine on one L40S; the base model must yield before the index build embeds —
    holding it OOM'd the embedding live (226 MiB free, 2026-07-05)."""
    global _model
    _model = None
    import gc
    gc.collect()
    _free_cuda()


def _answer_with_context(model, tokenizer, carts, question, context="", *,
                         max_new_tokens=96, max_ctx_tokens=3500, want_conf=False):
    """Generate from cart KV (+ optional raw `context` prefix). Returns (answer, mean_logprob).

    context="" -> cartridge alone (the Engram Smart CAG path). context=retrieved-docs -> cart-augmented
    RAG (`cart_rag`, the parity floor). `want_conf` returns the mean token log-prob — a free
    confidence signal (decoded in the same pass) used to route the adaptive cascade."""
    ctx = context
    if context:
        ids = tokenizer(context, add_special_tokens=False).input_ids[:max_ctx_tokens]
        ctx = tokenizer.decode(ids, skip_special_tokens=True)
    prompt = (f"{ctx}\n\nQ: {question}\nA:") if ctx else f"\n\nQ: {question}\nA:"
    out = generate_with_cartridges(model, tokenizer, carts, prompt,
                                   max_new_tokens=max_new_tokens, temperature=0.0,
                                   return_logprob=want_conf)
    text, conf = out if want_conf else (out, None)
    for stop in ("\nQ:", "\n\n", "\n"):
        if stop in text:
            text = text.split(stop, 1)[0]
            break
    return _strip_lead(text.strip()), conf


# Retrieval backend (which cartridges to compose for a query). Default = `dense`:
# semantic embedding retrieval (cosine over a small encoder), which matches meaning
# rather than shared words and tripled paraphrase recall@1 (0.25->0.75) vs the old
# `tfidf` lexical retriever in offline eval. `tfidf` remains available via env for
# comparison/fallback. `dense`/`embedding`/`pgvector` all select the embedding
# retriever; in production the vectors live in pgvector (RDS) for an ANN-indexed,
# multi-tenant store — same selection logic (see C2/C3).
RETRIEVAL_BACKEND = os.environ.get("RETRIEVAL_BACKEND", "dense").lower()


def _make_retriever(docs: dict[str, str]):
    # Semantic/dense retrieval (embedding cosine) matches meaning, not just shared
    # words — it fixes the TF-IDF recall ceiling the accuracy eval exposed. "pgvector"
    # selects the same dense retrieval; in production the vectors live in pgvector (RDS)
    # for a shared, ANN-indexed, multi-tenant store (the selection logic is identical).
    if RETRIEVAL_BACKEND in ("dense", "embedding", "pgvector"):
        return DenseRetriever(docs)
    return TfidfRetriever(docs)


def query_corpus(corpus_dir, question, k):
    model, tokenizer = get_model()
    # load_carts_from() looks in <run_dir>/cartridges, so pass the corpus dir.
    carts = load_carts_from(Path(corpus_dir), device=str(next(model.parameters()).device))
    if not carts:
        raise HTTPException(404, "corpus has no trained cartridges")
    docs_text = json.loads((Path(corpus_dir) / "docs.json").read_text(encoding="utf-8"))
    retriever = _make_retriever({d: docs_text.get(d, "") for d in carts})
    selected = select_carts("retrieval", carts, query=question,
                            retriever=retriever, k=min(k, len(carts)))
    answer = _strip_lead(answer_question(model, tokenizer, selected, question, max_new_tokens=96))
    used = [c.doc_id for c in selected]
    return answer, used


def compare_corpus(corpus_dir, question, k):
    """Answer one question two ways over the same corpus — the product vs the baseline — measuring
    real same-hardware latency + token counts. The control plane turns the token counts into $ (RAG
    at frontier prices, the cartridge path at its multiplexed marginal). Strategies:
      - everyday (Engram Smart CAG): demo-only adaptive cost router for this local HF compare path —
        the SHIPPED product is the vLLM connector serving CAG/QRC cartridges (see VISION.md "what
        ships"). Answer from the cartridge alone (0 raw tokens); escalate to cart + the retrieved
        docs only when the cart is unsure.
      - rag:      the baseline — retrieve k raw docs, stuff them into context every query, generate.
    """
    model, tokenizer = get_model()
    device = str(next(model.parameters()).device)
    carts = load_carts_from(Path(corpus_dir), device=device)
    if not carts:
        raise HTTPException(404, "corpus has no trained cartridges")
    docs_text = json.loads((Path(corpus_dir) / "docs.json").read_text(encoding="utf-8"))
    retriever = _make_retriever({d: docs_text.get(d, "") for d in carts})
    k = max(1, min(k, len(carts)))
    ans_max = 96
    q_tokens = _ntok(tokenizer, question)

    # Warm up once so the first timed strategy isn't charged for lazy CUDA/JIT init.
    try:
        teacher_answer(model, tokenizer, "warm up", "Reply ok.", 4, device)
    except Exception:  # noqa: BLE001
        _free_cuda()

    def _run(gen):
        """Time one strategy's generation; on failure (e.g. CUDA OOM) return a
        short note instead of raising — so one strategy can never take down the
        whole comparison. Frees GPU memory so the remaining strategies still run."""
        t0 = time.perf_counter()
        try:
            ans = _strip_lead(gen())
            return ans, round((time.perf_counter() - t0) * 1000.0, 1), None
        except Exception as exc:  # noqa: BLE001
            _free_cuda()
            name = type(exc).__name__
            return None, None, "out of memory on this GPU" if "OutOfMemory" in name else name

    results: dict[str, dict] = {}

    def _timed(fn):
        """Time a generation that already returns (answer, conf) — no extra trim/strip."""
        t0 = time.perf_counter()
        try:
            return fn(), round((time.perf_counter() - t0) * 1000.0, 1), None
        except Exception as exc:  # noqa: BLE001
            _free_cuda()
            name = type(exc).__name__
            return None, None, "out of memory on this GPU" if "OutOfMemory" in name else name

    # Shared retrieval: the SAME top-k raw docs RAG uses are the docs the cartridge router can
    # escalate to, so the two paths see the same evidence.
    top_ids = retriever.topk(question, k)
    rag_context = "\n\n".join(docs_text.get(d, "") for d in top_ids)
    rag_ctx_tokens = _ntok(tokenizer, rag_context)
    selected = select_carts("retrieval", carts, query=question, retriever=retriever, k=k)
    cart_kv_tokens = sum(c.num_kv_tokens() for c in selected)
    cart_doc_ids = [c.doc_id for c in selected]

    # --- Engram Smart CAG here = a demo-only adaptive cost router for the local HF compare path;
    # the SHIPPED product is the vLLM connector serving CAG/QRC cartridges (see VISION.md "what
    # ships"). Answer from the cartridge alone first (0 raw tokens); escalate to cart + the
    # retrieved docs ONLY if the cart is unsure. The tier it picked + the confidence drive the
    # per-query routing readout in the UI.
    theta = float(os.environ.get("ADAPTIVE_THETA", "-0.7"))
    res, l0_ms, l0_err = _timed(lambda: _answer_with_context(
        model, tokenizer, selected, question, max_new_tokens=ans_max, want_conf=True))
    l0_ans, conf = res if res else (None, None)
    if conf is not None and conf >= theta:          # confident from the cartridge alone
        ev_ans, ev_ms, ev_raw, ev_used, ev_feas, tier = (
            l0_ans, l0_ms, 0, cart_doc_ids, l0_err is None, "cartridge")
    else:                                            # unsure -> escalate to cart + retrieved docs
        res2, l1_ms, l1_err = _timed(lambda: _answer_with_context(
            model, tokenizer, selected, question, context=rag_context, max_new_tokens=ans_max))
        ev_ans, ev_raw, ev_used, ev_feas, tier = (
            (res2[0] if res2 else None), rag_ctx_tokens + q_tokens, top_ids, l1_err is None, "cart+docs")
        ev_ms = (l0_ms or 0) + (l1_ms or 0)          # the router ran L0 then escalated
    results["everyday"] = {
        "answer": ev_ans, "latency_ms": ev_ms,
        "prompt_tokens": cart_kv_tokens + ev_raw, "raw_tokens": ev_raw, "cart_tokens": cart_kv_tokens,
        "gen_tokens": _ntok(tokenizer, ev_ans or ""), "feasible": ev_feas,
        "tier": tier, "confidence": round(conf, 3) if conf is not None else None, "theta": theta,
        "used_docs": ev_used, "note": None,
    }

    # --- RAG (the baseline): top-k raw chunks in context every query, no cartridge ---
    ans, ms, err = _run(lambda: teacher_answer(model, tokenizer, rag_context, question, ans_max, device))
    results["rag"] = {
        "answer": ans, "latency_ms": ms,
        "prompt_tokens": rag_ctx_tokens + q_tokens,
        "gen_tokens": _ntok(tokenizer, ans or ""), "feasible": err is None,
        "used_docs": top_ids, "note": err,
    }

    corpus_tokens = sum(_ntok(tokenizer, t) for t in docs_text.values())
    return {"results": results, "k": k, "corpus_tokens": corpus_tokens}


def _cart_store():
    """The cartridge store CAG carts are written to at onboarding (and the vLLM Inference Service reads
    from). Selected by env: CARTRIDGE_STORE_BACKEND=s3 (+ CARTRIDGE_STORE_BUCKET / _PREFIX) is the
    production durable tier; =local (+ CARTRIDGE_STORE_DIR) is the dev default."""
    from cartridges.cart_store import get_cartridge_store
    if os.environ.get("CARTRIDGE_STORE_BACKEND", "local").lower() == "s3":
        # cache_dir MUST be a host mount, not the container layer: carts are ~0.4 GB each and the
        # default (/tmp inside the container) fills the root EBS at corpus scale (found live —
        # 1k onboarding killed the box at ~210 docs). max_cache_items bounds the mirror.
        return get_cartridge_store(
            "s3", bucket=os.environ["CARTRIDGE_STORE_BUCKET"],
            prefix=os.environ.get("CARTRIDGE_STORE_PREFIX", "cartridges"),
            cache_dir=os.environ.get("CART_CACHE_DIR") or None,
            max_cache_items=int(os.environ.get("CART_CACHE_ITEMS", "64")))
    return get_cartridge_store("local", root=os.environ.get("CARTRIDGE_STORE_DIR", "/data/storage/carts"))


# On-disk KV dtype for onboarded carts — tier defaults per KV_COMPRESSION.md (engram-dynamics-landing repo, internal/) (8B gate
# 2026-07-05): cc_safe for the local/hot tier (3.5x vs bf16 at BETTER-than-fp8 fidelity),
# cc_aggr for the S3 durable tier (4.1x, F1 parity). bf16 remains the lossless escape hatch
# (CART_STORE_DTYPE=bf16); int4 measured worst fidelity-per-byte — don't use it.
CART_STORE_DTYPE = os.environ.get(
    "CART_STORE_DTYPE",
    "cc_aggr" if os.environ.get("CARTRIDGE_STORE_BACKEND", "local").lower() == "s3"
    else "cc_safe").lower()

# --- fused retrieval (the 1k-benchmark's winning retriever, served from the GPU box) ----------
# Built per corpus at onboarding when the control plane asks (RETRIEVAL_BACKEND=fused); embeddings
# persisted under <corpus_dir>/fused_index.pt; queried via POST /retrieve. Encoders load lazily
# and once (embedder + cross-encoder, ~2.5 GB fp16 on the GPU).
_FUSED_ENCODERS = None
_FUSED_CACHE: dict[str, object] = {}   # corpus_dir -> FusedIndex (tiny LRU)
_FUSED_CACHE_MAX = 2


def _fused_encoders():
    global _FUSED_ENCODERS
    if _FUSED_ENCODERS is None:
        from retrieval_fused import load_encoders
        _FUSED_ENCODERS = load_encoders(
            embedder=os.environ.get("RETR_EMBEDDER", "Qwen/Qwen3-Embedding-0.6B"),
            reranker=os.environ.get("RETR_RERANKER", "BAAI/bge-reranker-v2-m3"))
    return _FUSED_ENCODERS


def _index_s3(corpus_dir: str):
    """(bucket, key_prefix) for the corpus's durable index copy, or None when the platform
    isn't on the S3 cart store (local dev keeps everything on disk)."""
    if os.environ.get("CARTRIDGE_STORE_BACKEND", "local").lower() != "s3":
        return None
    bucket = os.environ.get("CARTRIDGE_STORE_BUCKET", "")
    if not bucket:
        return None
    prefix = os.environ.get("CARTRIDGE_STORE_PREFIX", "cartridges").strip("/")
    return bucket, f"{prefix}-indexes/{Path(corpus_dir).name}"


def _index_sync_up(corpus_dir: str) -> None:
    """Push fused_index.pt + docs.json to S3 after a build. The GPU box's EBS is a CACHE:
    both files lived only there, so every box replacement silently killed retrieval until
    a full re-onboard (happened twice, 2026-07-06)."""
    loc = _index_s3(corpus_dir)
    if loc is None:
        return
    bucket, prefix = loc
    import boto3
    s3 = boto3.client("s3")
    for name in ("fused_index.pt", "docs.json"):
        f = Path(corpus_dir) / name
        if f.exists():
            s3.upload_file(str(f), bucket, f"{prefix}/{name}")
    print(f"[index] synced fused index -> s3://{bucket}/{prefix}/", flush=True)


def _index_sync_down(corpus_dir: str) -> bool:
    """Hydrate the index files from S3 onto a fresh box. True if both files landed."""
    loc = _index_s3(corpus_dir)
    if loc is None:
        return False
    bucket, prefix = loc
    import boto3
    from botocore.exceptions import ClientError
    s3 = boto3.client("s3")
    Path(corpus_dir).mkdir(parents=True, exist_ok=True)
    try:
        for name in ("fused_index.pt", "docs.json"):
            s3.download_file(bucket, f"{prefix}/{name}", str(Path(corpus_dir) / name))
    except ClientError:
        return False
    print(f"[index] hydrated fused index <- s3://{bucket}/{prefix}/", flush=True)
    return True


def _fused_index_for(corpus_dir: str):
    """Load (or fetch cached) the corpus's fused index; on a fresh/replaced box it hydrates
    from the S3 durable copy first; 409 only if no copy exists anywhere (corpus was never
    onboarded with index building)."""
    from retrieval_fused import FusedIndex
    idx = _FUSED_CACHE.pop(corpus_dir, None)
    if idx is not None:
        # Re-insert on hit so eviction (oldest-first below) is true LRU, not FIFO — with
        # >MAX active corpora a plain get() kept evicting the busiest index.
        _FUSED_CACHE[corpus_dir] = idx
        return idx
    path = Path(corpus_dir) / "fused_index.pt"
    docs_path = Path(corpus_dir) / "docs.json"
    if not path.exists() or not docs_path.exists():
        if not _index_sync_down(corpus_dir):
            raise HTTPException(409, "no fused index for this corpus — onboard it with "
                                     "RETRIEVAL_BACKEND=fused so the index is built")
    docs = json.loads(docs_path.read_text(encoding="utf-8"))
    embed_fn, rerank_fn, _q = _fused_encoders()
    idx = FusedIndex.load(path, docs, embed_fn=embed_fn, rerank_fn=rerank_fn)
    _FUSED_CACHE[corpus_dir] = idx
    while len(_FUSED_CACHE) > _FUSED_CACHE_MAX:
        _FUSED_CACHE.pop(next(iter(_FUSED_CACHE)))
    return idx


def build_fused_index(corpus_dir: str, docs: list[dict], report=None) -> None:
    """Embed the corpus + persist the fused index (called at onboarding when requested).

    GPU-memory hygiene on failure: the embed step is GPU-heavy and shares the card with the
    serve engine, so on a single-GPU tier it can OOM mid-build. When it throws, the partial
    FusedIndex (its dense matrix on GPU) plus embed()'s intermediate activations must be freed,
    or the NEXT build attempt starts with less headroom and OOMs too — this is the compounding
    leak that made three attempts accumulate ~10.5 GB during the 2026-07-22 deploy. Two things
    keep the memory alive that plain scope-exit does not: (1) the propagating exception's
    traceback pins EVERY frame's locals (incl. the partial idx and embed()'s activation tensors)
    for as long as any caller holds the error — and the job runner stores it as the failure
    reason; (2) torch's caching allocator keeps freed blocks reserved until empty_cache(). And a
    module-level strong ref beats both: save() copies the matrix to CPU but leaves idx.mat on the
    GPU, so if the index were already published to _FUSED_CACHE, nulling the local ref would free
    nothing. So the durable sync runs BEFORE we publish to the serve cache (a failed S3 upload
    never leaves a non-durable index cached — and never leaks its GPU matrix), and on any failure
    we evict the cache entry, drop our own ref, clear the traceback frames (unpins the deeper
    activations) BEFORE gc/empty_cache, then re-raise the original error unchanged."""
    from retrieval_fused import FusedIndex
    embed_fn, rerank_fn, q_prompt = _fused_encoders()
    if report:
        report(0.99, None, "Building retrieval index")
    idx = None
    try:
        idx = FusedIndex({d["doc_id"]: d["text"] for d in docs if d.get("text", "").strip()},
                         embed_fn=embed_fn, rerank_fn=rerank_fn, q_prompt=q_prompt)
        idx.embed()
        idx.save(Path(corpus_dir) / "fused_index.pt")
        _index_sync_up(corpus_dir)         # durable copy FIRST: box EBS is only a cache
        _FUSED_CACHE[corpus_dir] = idx     # publish to the serve cache only once durable
    except BaseException as e:
        # Release GPU memory the failed build reserved. Evict any cache entry FIRST — a
        # module-level strong ref (idx.mat stays on GPU after save()'s .cpu() copy) would keep the
        # dense matrix resident across gc/empty_cache. Then null our ref and clear_frames BEFORE
        # gc/empty_cache, else the traceback-pinned tensors survive collection and empty_cache
        # can't return their blocks. clear_frames silently skips this still-executing frame (its
        # idx is already None) and clears the deeper ones.
        _FUSED_CACHE.pop(corpus_dir, None)
        idx = None
        if e.__traceback__ is not None:
            traceback.clear_frames(e.__traceback__)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise


def onboard_cag_corpus(corpus_dir, docs, report=None, build_index=False, should_cancel=None):
    """Production onboarding for the vLLM serve path: build a CAG cart per document (one frozen forward
    pass — the document's full KV, NO gradient descent) and write it to the cartridge store by doc_id,
    where the Inference Service serves it. 'Read once, infer many.' Returns the same shape as train()
    so the control plane stores n_cartridges / seconds / corpus_tokens unchanged.
    `build_index=True` additionally embeds the corpus into the fused retrieval index (the control
    plane sets it when RETRIEVAL_BACKEND=fused).
    `should_cancel()` is polled at batch boundaries (same cooperative contract as train_corpus);
    on cancel we return {"canceled": True} early — carts already in the durable store stay, and
    the idempotent exists() split makes the next run resume where this one stopped."""
    # Model loads LAZILY on the first doc that actually needs building: an all-reused
    # re-run (idempotent path below) never touches the GPU until the index build.
    tokenizer = get_tokenizer()
    model = None
    store = _cart_store()
    max_doc_tok = int(os.environ.get("CAG_MAX_DOC_TOK", "4096"))
    # Idempotent by doc_id: a cart already in the durable store is reused, so re-running
    # onboarding (a new corpus over the same docs, or a job that died late — e.g. in the
    # index build) costs an exists() check instead of forward+encode+upload per doc.
    # doc_id is the content key by convention (the loader slugs stable ids from titles);
    # FORCE_REONBOARD=1 rebuilds everything (use after changing model or store dtype).
    force = os.environ.get("FORCE_REONBOARD", "0") == "1"
    should_cancel = should_cancel or (lambda: False)
    t0 = time.perf_counter()
    cart_build_s = 0.0  # GPU wall-clock ACTUALLY building carts (excludes idempotent reuse + index build)
    valid = [d for d in docs if d.get("text", "").strip()]
    corpus_tokens = sum(min(len(d["text"]) // 4 + 1, max_doc_tok) for d in valid)  # ~est (chars/4)
    # Idempotent split: a cart already in the durable store is reused (an exists() check, no GPU).
    to_build = [d for d in valid if force or not store.exists(d["doc_id"])]
    n_skipped = len(valid) - len(to_build)
    n = n_skipped
    # BATCHED onboarding — the frozen forward is the throughput bottleneck. On a multi-GPU
    # device_map=auto box the forward is a naive PIPELINE (one GPU active at a time), so a
    # batch-of-1 forward wastes N-1 GPUs and is latency-bound (~10 s/doc measured at 30B/4×L40S).
    # Batch B docs per forward to amortize that pipeline cost; cag_carts_batch stays token-exact
    # (right-pad + explicit positions). Sort by length so each batch pads to a similar max; a token
    # budget caps per-batch VRAM. Tune with CAG_ONBOARD_BATCH / CAG_ONBOARD_BATCH_TOKENS.
    from cartridges.serve.serve_carts import cag_carts_batch
    to_build.sort(key=lambda d: len(d["text"]))
    tok_budget = int(os.environ.get("CAG_ONBOARD_BATCH_TOKENS", "12288"))
    max_batch = int(os.environ.get("CAG_ONBOARD_BATCH", "16"))

    def _flush(group):
        nonlocal model, n, cart_build_s
        if not group:
            return
        _fs = time.perf_counter()
        if model is None:
            model, _ = get_model()
        for doc_id, cart in cag_carts_batch(
                model, tokenizer, [(d["doc_id"], d["text"]) for d in group],
                max_ctx_tokens=max_doc_tok, model_ref=_onboard_model_ref()):
            store.put(doc_id, cart, store_dtype=CART_STORE_DTYPE)   # -> durable tier
            n += 1
        _free_cuda()
        cart_build_s += time.perf_counter() - _fs   # count only real forward+encode+upload work
        if report:
            report(0.02 + 0.96 * min(n, len(valid)) / max(len(valid), 1),
                   detail=f"Onboarded {n} cartridge(s)"
                          + (f" ({n_skipped} reused)" if n_skipped else ""))

    def _canceled_result():
        return {"n_cartridges": n, "canceled": True, "method": "cag",
                "train_seconds": round(time.perf_counter() - t0, 1),
                "cart_seconds": round(cart_build_s, 1), "n_built": n - n_skipped,
                "corpus_tokens": corpus_tokens}

    group, group_tok = [], 0
    for d in to_build:
        if should_cancel():
            return _canceled_result()
        est = min(len(d["text"]) // 4 + 1, max_doc_tok)
        if group and (group_tok + est > tok_budget or len(group) >= max_batch):
            _flush(group)
            group, group_tok = [], 0
        group.append(d)
        group_tok += est
    _flush(group)
    if should_cancel():   # last chance before the (long) docs.json + index-build tail
        return _canceled_result()
    # Persist doc texts: the fused index rebuilds its lexical side from these at load, and any
    # retrieval backend needs them co-located with the index. (The dir may not exist on this box —
    # the control plane's storage mirror lives on its own filesystem.)
    Path(corpus_dir).mkdir(parents=True, exist_ok=True)
    (Path(corpus_dir) / "docs.json").write_text(
        json.dumps({d["doc_id"]: d["text"] for d in docs if d.get("text", "").strip()}),
        encoding="utf-8")
    if build_index:
        # The base model is dead weight during embedding — yield its VRAM to the
        # encoders (no-op when the run was all-reused and it never loaded).
        model = None
        _unload_model()
        build_fused_index(corpus_dir, docs, report=report)
        _free_cuda()
    return {"n_cartridges": n, "canceled": False, "method": "cag",
            "train_seconds": round(time.perf_counter() - t0, 1),
            "cart_seconds": round(cart_build_s, 1), "n_built": len(to_build),
            "corpus_tokens": corpus_tokens}


def _make_reporter(progress_url, token):
    """Build a non-blocking, failure-proof training heartbeat poster, paired with
    a cancel Event. Heartbeats are best-effort: a slow or down control plane must
    never stall or fail GPU training, so each POST runs in a daemon thread with a
    short timeout and any error is swallowed. The control plane's heartbeat
    *response* may carry {"cancel": true} (the only inbound channel to the
    worker); when it does we set the Event, and the training loop polls
    `cancel_event.is_set` to abort. On AWS this is the same callback URL."""
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


# ---------------------------------------------------------------- HTTP API
# ML-plane shared-token auth (DEFAULT OFF). When ML_AUTH_TOKEN is set and non-empty, every route
# EXCEPT /health requires `Authorization: Bearer <token>` (constant-time compared); unset/empty
# reproduces today's behavior exactly (no auth — trust is SG-only). The backend attaches this same
# token on every ML-plane request (platform/backend/app/ml_client.py). Mirrors the identical gate on
# the vLLM Inference Service (platform/ml_service/vllm_inference.py).
ML_AUTH_TOKEN = os.environ.get("ML_AUTH_TOKEN", "")
_AUTH_OPEN_PATHS = frozenset({"/health", "/healthz", "/ready", "/readiness", "/readyz"})


def _bearer_ok(auth_header: str | None, token: str) -> bool:
    """True when `auth_header` is exactly 'Bearer <token>'. Constant-time (hmac.compare_digest) so a
    wrong token can't be discovered by timing; a missing/malformed header is a plain False."""
    if not auth_header:
        return False
    scheme, _, presented = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not presented:
        return False
    return hmac.compare_digest(presented, token)


app = FastAPI(title="Cartridge ML Service")


@app.middleware("http")
async def _ml_auth(request, call_next):
    """Shared-token gate (DEFAULT OFF). ML_AUTH_TOKEN unset/empty -> today's behavior (open). Set ->
    every route except the liveness/readiness paths needs a correct Bearer token; a missing/wrong one
    gets 401 before the handler runs. Read live from the env (not the import-time value) so a
    deploy/test that sets the env after import still takes effect."""
    from fastapi.responses import JSONResponse
    token = os.environ.get("ML_AUTH_TOKEN", "")
    if token and request.url.path not in _AUTH_OPEN_PATHS:
        if not _bearer_ok(request.headers.get("authorization"), token):
            return JSONResponse({"detail": "missing or invalid ML auth token"}, status_code=401)
    return await call_next(request)


class TrainReq(BaseModel):
    corpus_dir: str
    docs: list[dict]
    cart_tokens: int = 64
    steps: int = 60
    grad_accum: int = 4
    gen_qs: int = 6
    # Onboarding method: "train" = per-doc gradient descent (parity reference); "encoder" =
    # amortized encoder, ~2 frozen forwards/doc, no per-doc training (the cheap production path).
    method: str = "train"
    encoder_ckpt: str | None = None  # encoder.pt path (else ENCODER_CKPT env) when method="encoder"
    # Optional progress callback (control plane -> live progress bar). Omit to disable.
    job_id: str | None = None
    progress_url: str | None = None
    progress_token: str | None = None


class QueryReq(BaseModel):
    corpus_dir: str
    question: str
    k: int = 3


class CompareReq(BaseModel):
    corpus_dir: str
    question: str
    k: int = 3


class OnboardCagReq(BaseModel):
    corpus_dir: str
    docs: list[dict]                 # [{doc_id, text}]
    build_index: bool = False        # also build the fused retrieval index (RETRIEVAL_BACKEND=fused)
    job_id: str | None = None
    progress_url: str | None = None
    progress_token: str | None = None


# Rerank candidate pool: the measured lever (50 keeps GPU cost ~100ms). Serving default reads
# RETR_POOL so a box can tune it without a client change; unchanged at 50 unless the env is set.
# Empty/unset both fall back to 50 (compose files often export the key blank).
_RETR_POOL_DEFAULT = int(os.environ.get("RETR_POOL", "").strip() or "50")


class RetrieveReq(BaseModel):
    corpus_dir: str
    question: str
    k: int = 3
    pool: int = _RETR_POOL_DEFAULT   # rerank candidates (the measured lever; 50 keeps GPU cost ~100ms)


class OffboardReq(BaseModel):
    doc_ids: list[str]               # cart ids (filename slugs) to delete from the durable store


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME, "device": DEVICE}


@app.post("/train")
def train_ep(req: TrainReq):
    if not req.docs:
        raise HTTPException(400, "no documents")
    corpus_dir = _safe_corpus_dir(req.corpus_dir)
    report, cancel_event = _make_reporter(req.progress_url, req.progress_token)
    with _lock:
        if req.method == "encoder":
            # Amortized-encoder onboarding (forward-pass materialization, no per-doc training).
            result = onboard_with_encoder(corpus_dir, req.docs, req.encoder_ckpt, report=report)
        else:
            result = train_corpus(corpus_dir, req.docs, req.cart_tokens,
                                  req.steps, req.grad_accum, req.gen_qs,
                                  report=report, should_cancel=cancel_event.is_set)
    # {n_cartridges, canceled, method, train_seconds, corpus_tokens} — control plane stores these.
    return result


@app.post("/query")
def query_ep(req: QueryReq):
    corpus_dir = _safe_corpus_dir(req.corpus_dir)
    with _lock:
        answer, used = query_corpus(corpus_dir, req.question, req.k)
    return {"answer": answer, "used_docs": used}


@app.post("/compare")
def compare_ep(req: CompareReq):
    corpus_dir = _safe_corpus_dir(req.corpus_dir)
    with _lock:
        return compare_corpus(corpus_dir, req.question, req.k)


@app.post("/onboard_cag")
def onboard_cag_ep(req: OnboardCagReq):
    """Build CAG carts for a corpus (the vLLM serve path's onboarding: one forward pass/doc, no
    training) and persist them to the cartridge store the Inference Service serves from."""
    if not req.docs:
        raise HTTPException(400, "no documents")
    corpus_dir = _safe_corpus_dir(req.corpus_dir)
    report, cancel_event = _make_reporter(req.progress_url, req.progress_token)
    with _lock:
        return onboard_cag_corpus(corpus_dir, req.docs, report=report,
                                  build_index=req.build_index,
                                  should_cancel=cancel_event.is_set)


@app.post("/retrieve")
def retrieve_ep(req: RetrieveReq):
    """Fused retrieval over an onboarded corpus: ranked doc_ids for the question. The control
    plane (RETRIEVAL_BACKEND=fused) calls this instead of its zero-dep BM25 — same doc_ids feed
    the resident-KV serve path and the RAG baseline."""
    corpus_dir = _safe_corpus_dir(req.corpus_dir)
    # Narrowed lock (default): retrieval serializes only against other retrievals, not against a
    # long onboarding/train hold. RETRIEVE_EXCLUSIVE=1 falls back to the global _lock on small GPUs
    # where concurrent-with-onboarding embedding is a VRAM risk (see _retr_lock comment).
    with (_lock if RETRIEVE_EXCLUSIVE else _retr_lock):
        idx = _fused_index_for(corpus_dir)
        return {"doc_ids": idx.retrieve(req.question, k=req.k, pool=req.pool)}


@app.post("/offboard")
def offboard_ep(req: OffboardReq):
    """Delete carts by id from the DURABLE cartridge store — the S3 object + its local mirror (or, on
    the local backend, the on-disk `{id}.pt`). This is HALF of the data-deletion path: it removes the
    blob at rest but NOT any warm KV the serving engine is holding. The control plane pairs this with
    the Inference Service's POST /invalidate to purge those caches; the two together are the complete
    'deleting a memory removes the document from serving' guarantee (see DATA_LIFECYCLE.md, engram-dynamics-landing repo).

    Each id is validated first (validate_cart_id: no path separators / '..' / leading dot) so a caller
    can never coerce a delete outside the store root — a bad id fails the WHOLE request 400 rather than
    partially deleting. delete() -> True means a blob was removed (`deleted`); False means it wasn't
    there (`missing`) — the caller treats missing as already-gone, so re-running is idempotent."""
    from cartridges.cart_store import validate_cart_id
    for did in req.doc_ids:
        try:
            validate_cart_id(did)
        except ValueError:
            raise HTTPException(400, f"invalid cart id {did!r}") from None
    store = _cart_store()
    deleted, missing = [], []
    for did in req.doc_ids:
        (deleted if store.delete(did) else missing).append(did)
    return {"deleted": deleted, "missing": missing}


@app.get("/carts")
def carts_ep():
    """List every cart id currently in the durable store. The control plane's GC reconciliation sweep
    (POST /internal/gc/carts) diffs this against the doc slugs it still references to find orphans —
    carts left behind when an offboard didn't reach the store (ML plane was down during a delete)."""
    return {"cart_ids": _cart_store().list_ids()}
