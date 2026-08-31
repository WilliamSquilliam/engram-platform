"""Cost-comparison model for the workspace 'Costs' view and the demo page —
**apples-to-apples (same open model)**.

Engram Smart CAG (cartridge) and RAG are compared on the SAME self-hosted open model
(Qwen3-30B-A3B class), because that's the honest baseline: parity is measured
same-model, and production serves an open model. The cost lever is "read once" —
RAG re-prefills the retrieved documents through the model on every query; the
cartridge's KV is preloaded, so it prefills only the question. Decode is ~equal
for both, so the difference is the skipped prefill.

RAG (same open model) is the only realistic baseline shown — full-corpus / frontier
long-context prefill isn't a production strategy, so it's intentionally not modeled.
Full write-up: COST_COMPARISON.md (engram-dynamics-landing repo).

Self-contained (no torch / repo deps); modeled estimates, env-overridable.
"""
import os

# ---- same-model serving (cartridge + open-RAG run this) ----------------------
SERVE_HOURLY = float(os.environ.get("SERVE_GPU_HOURLY", "2.24"))  # 1x L40S g6e.2xlarge on-demand
PREFILL_TPS = float(os.environ.get("PREFILL_TPS", "25000"))      # 30B-A3B FP8 prefill tok/s (soft)
DECODE_TPS = float(os.environ.get("DECODE_TPS", "1500"))         # ... decode tok/s aggregate (soft)
UTIL = float(os.environ.get("SERVE_UTIL", "0.5"))               # multiplexed utilisation
PGVECTOR_MO = 5.0       # pgvector in existing RDS (both sides, self-hosted)

Q_IN, Q_OUT = 150, 300  # question / answer tokens
RAG_CHUNKS = 6_000      # retrieved context RAG prefills per query

# GPU $/hr to turn measured training wall-clock into a training $ figure.
GPU_HOURLY_ONDEMAND = float(os.environ.get("GPU_HOURLY_ONDEMAND", "2.24"))
GPU_HOURLY_SPOT = float(os.environ.get("GPU_HOURLY_SPOT", "0.65"))

# The REAL on-demand $/hr of the box actually serving (set per tier by the deploy: g6e.xlarge ~1.86,
# g6e.12xlarge ~10.49). Measured $/query uses THIS, not a documentation constant.
INFERENCE_GPU_HOURLY = float(os.environ.get("INFERENCE_GPU_HOURLY", SERVE_HOURLY))


# Standard answer length for cost comparison: both sides priced at the SAME decode volume, so
# per-question verbosity (a model choice) can't masquerade as a serving-cost difference.
STD_ANSWER_TOKENS = int(os.environ.get("STD_ANSWER_TOKENS", "150"))


def price_normalized(ttft_ms: float | None, decode_tps: float | None,
                     hourly: float | None = None, util: float | None = None) -> float | None:
    """$/query from MEASURED components at a standard answer length: measured TTFT (where the
    serving difference lives — prefill skipped vs re-done) + STD_ANSWER_TOKENS at the side's
    measured decode rate. Robust to verbosity and honest: every input is measured."""
    if not ttft_ms or not decode_tps:
        return None
    hourly = INFERENCE_GPU_HOURLY if hourly is None else hourly
    util = UTIL if util is None else util
    gpu_s = ttft_ms / 1000.0 + STD_ANSWER_TOKENS / decode_tps
    return gpu_s / max(util, 1e-9) * (hourly / 3600.0)


def price_from_latency(latency_ms: float, hourly: float | None = None, util: float | None = None) -> float:
    """$/query straight from MEASURED wall-clock GPU time on the serving box x its real $/hr, divided by
    the multiplex utilisation assumption — the most-measured cost number we can show. Cart and RAG are
    clocked on the SAME box, so their $ ratio is exactly the measured latency ratio."""
    hourly = INFERENCE_GPU_HOURLY if hourly is None else hourly
    util = UTIL if util is None else util
    return (latency_ms / 1000.0) / max(util, 1e-9) * (hourly / 3600.0)


def _serve(prefill_tok: float, gen_tok: float = Q_OUT) -> float:
    """$/query to serve `prefill_tok` of new context + decode on the shared open-model GPU."""
    gpu_seconds = prefill_tok / PREFILL_TPS + gen_tok / DECODE_TPS
    return gpu_seconds / UTIL * (SERVE_HOURLY / 3600.0)


# ---------------------------------------------------------------- strategies
def cartridge_cost() -> float:
    """Engram Smart CAG: KV preloaded -> prefill only the question (read-once), then decode.
    Flat in corpus size; same open model as the RAG baseline."""
    return _serve(Q_IN)


def rag_cost(n_per_mo: float, ctx: int = RAG_CHUNKS) -> float:
    """open-RAG (same model): prefill the retrieved context every query + decode,
    plus the amortized pgvector store. Apples-to-apples with cartridge_cost()."""
    return _serve(ctx + Q_IN) + PGVECTOR_MO / max(n_per_mo, 1e-9)


def compare(corpus_tokens: int, queries_per_month: int) -> dict:
    n = max(1, queries_per_month)
    cart = cartridge_cost()
    rag = rag_cost(n)

    # RAG (same open model) is the only realistic head-to-head baseline; full-corpus / frontier
    # prefill is intentionally not shown (it isn't a production strategy).
    strategies = [
        {
            "key": "cartridge", "label": "Cartridge (this platform)",
            "per_query": round(cart, 6), "per_month": round(cart * n, 2),
            "feasible": True, "quality": "open-model answer, full-corpus fidelity",
            "note": "read-once KV cartridges; no per-query document prefill",
        },
        {
            "key": "rag", "label": "RAG (open model, self-hosted — same model)",
            "per_query": round(rag, 6), "per_month": round(rag * n, 2),
            "feasible": True, "quality": "same open model, lossy chunk recall",
            "note": "re-prefills ~6k retrieved tokens through the model every query",
        },
    ]

    savings = {
        # honest, same-model: cartridge vs open-RAG (the read-once per-query prefill skip)
        "vs_rag_x": round(rag / cart, 1),
        "vs_rag_pct": round((1 - cart / rag) * 100, 1),
    }
    return {
        "inputs": {"corpus_tokens": corpus_tokens, "queries_per_month": queries_per_month},
        "strategies": strategies,
        "savings": savings,
        "basis": "same-model (open 30B) for cartridge & RAG — the realistic apples-to-apples baseline",
    }


# --------------------------------------------------------------------------- #
# Per-strategy pricing from MEASURED token counts (used by the side-by-side
# compare). Cartridge & RAG are the SAME open model (apples-to-apples). Mirrors
# the model above so numbers tie out.
# --------------------------------------------------------------------------- #
def price_rag(prompt_tokens: int, gen_tokens: int, n_per_mo: int) -> float:
    """$/query for open-RAG on the same model: prefill the retrieved prompt + decode,
    plus the amortized pgvector store. Uses the query's real token counts."""
    return _serve(prompt_tokens, gen_tokens) + PGVECTOR_MO / max(n_per_mo, 1e-9)


def price_everyday() -> float:
    """$/query for the Engram Smart CAG cartridge path (read-once serving, flat in corpus size)."""
    return cartridge_cost()


# ---- onboarding estimate (review step, BEFORE any GPU run) -------------------
# Coarse per-document constants for the wizard's "review" step — a pre-run sizing estimate, not a
# measured number (measured timing/cost come from metrics above once a run completes). One obvious
# place so a tweak updates every estimate. Env-overridable like the rest of this module.
ONBOARD_SECONDS_PER_DOC = float(os.environ.get("ONBOARD_SECONDS_PER_DOC", "8.0"))   # est. GPU s / doc
ONBOARD_CART_GB_PER_DOC = float(os.environ.get("ONBOARD_CART_GB_PER_DOC", "0.05"))  # est. cart storage / doc


def onboard_estimate(n_documents: int) -> dict:
    """Coarse pre-run sizing for the review step: estimated onboarding wall-clock + GPU $ from the
    per-doc constants above x the on-demand GPU rate. Deliberately rough — the real figures land on
    the corpus after the run (see /corpora/{id}/economics)."""
    est_seconds = n_documents * ONBOARD_SECONDS_PER_DOC
    est_cost = training_cost(est_seconds, GPU_HOURLY_ONDEMAND)
    return {
        "est_seconds": round(est_seconds, 1),
        "est_cost_ondemand": round(est_cost, 4),
        "est_cart_gb": round(n_documents * ONBOARD_CART_GB_PER_DOC, 3),
        "gpu_hourly_ondemand": GPU_HOURLY_ONDEMAND,
        "seconds_per_doc": ONBOARD_SECONDS_PER_DOC,
    }


def training_cost(train_seconds: float | None, gpu_hourly: float) -> float:
    """Training $ = measured GPU wall-clock x GPU $/hr."""
    return (train_seconds or 0.0) / 3600.0 * gpu_hourly


def breakeven_queries(train_cost: float, alt_per_q: float | None, cart_per_q: float) -> float | None:
    """Queries until the one-time training cost is repaid by the per-query saving vs an
    alternative. None if the alternative isn't cheaper to beat or doesn't save anything.

    NOTE: vs the same-model open-RAG the per-query saving is the read-once prefill skip, so this
    break-even is large. See COST_COMPARISON.md (engram-dynamics-landing repo) — the cartridge's edge vs open-RAG is latency /
    large-context / quality, with a positive cost case for large-context x high-reuse x stable corpora."""
    if alt_per_q is None:
        return None
    save = alt_per_q - cart_per_q
    if save <= 0:
        return None
    return train_cost / save
