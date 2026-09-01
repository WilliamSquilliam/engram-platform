"""Ring buffer of MEASURED per-query metrics from the live serving path, so the demo / cost views show
real numbers collected at run time on THIS deployment — not modeled documentation constants. The buffer
is a hot, process-local cache (summaries stay cheap and read the LATEST record per side); every record
is ALSO appended to the `measurements` table so the numbers — and the /metrics/savings lifetime totals —
survive a restart or span a fleet.

Each record is one head-to-head: the cart serve path and the RAG baseline, both clocked on the same
vLLM engine. `summary()` reduces recent records to the latest per side + measured $/query (from measured
latency x the box's real $/hr). On first use the buffer warms from the last N persisted rows, so a fresh
process reflects history instead of showing {measured: False} until the next live query lands."""
from __future__ import annotations

import logging
import os
import threading
from collections import deque

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_BUFFER_MAX = int(os.environ.get("METRICS_BUFFER", "200"))
# The buffer keeps a short history for debugging, but summary() reads only the LATEST
# record per side — the cost/summary views reflect the question just asked, not a
# median over the session (the stream path appends cart and rag as adjacent records).
_RECORDS: deque[dict] = deque(maxlen=_BUFFER_MAX)
_WARMED = False  # lazy one-shot: warm the buffer from the DB on the first record()/summary()

# Labels for "measured on this deployment (<model> / <instance>)" — set per tier by the deploy.
MODEL_LABEL = os.environ.get("INFERENCE_MODEL_LABEL", os.environ.get("CARTRIDGES_MODEL", ""))
INSTANCE_LABEL = os.environ.get("INFERENCE_INSTANCE_LABEL", "")


def _num(m: dict, field: str):
    """Numeric fields only (the metrics dict can carry None / non-numeric flags like `tier`)."""
    v = m.get(field)
    return v if isinstance(v, (int, float)) else None


def _price(m: dict) -> float | None:
    """$/query for one measured side, length-normalized (measured TTFT + a standard answer at the
    measured decode rate); falls back to raw-latency pricing when ttft/decode aren't available.
    Same basis summary() and the compare view use, computed once here so the persisted row and the
    lifetime aggregates never re-derive pricing."""
    from . import metrics
    lat = _num(m, "latency_ms")
    return (metrics.price_normalized(_num(m, "ttft_ms"), _num(m, "decode_tps"))
            or (metrics.price_from_latency(lat) if lat else None))


def _persist(cart: dict | None, rag: dict | None, tenant_id: str | None) -> None:
    """Best-effort append of the just-recorded sides to the measurements table via a short-lived
    session. A DB failure must NEVER break the serve path: swallow everything, log once at warning.

    tenant_id (when the caller is a corpus-scoped serve path) is stamped on every row so per-tenant
    billing can attribute the query; None leaves the row NULL (deployment-level / demo)."""
    try:
        from .db import SessionLocal
        from .models import Measurement

        rows = []
        for side, m in (("cart", cart), ("rag", rag)):
            if not m:
                continue
            rows.append(Measurement(
                side=side,
                tenant_id=tenant_id,
                latency_ms=_num(m, "latency_ms"),
                ttft_ms=_num(m, "ttft_ms"),
                prompt_tokens=_num(m, "prompt_tokens"),
                resident_kv_tokens=_num(m, "resident_kv_tokens"),
                gen_tokens=_num(m, "gen_tokens"),
                decode_tps=_num(m, "decode_tps"),
                confidence=_num(m, "confidence"),
                cost_per_query=_price(m),
                model_label=MODEL_LABEL or None,
                instance_label=INSTANCE_LABEL or None,
            ))
        if not rows:
            return
        session = SessionLocal()
        try:
            session.add_all(rows)
            session.commit()
        finally:
            session.close()
    except Exception:  # noqa: BLE001 — persistence is best-effort; the serve path must survive it
        logger.warning("measurements: failed to persist a measured record (continuing)", exc_info=True)


def _warm_from_db() -> None:
    """Lazily fill the in-memory buffer from the last N persisted rows so summaries survive a restart.
    Rows are grouped back into {cart, rag} records by their recorded order. Best-effort: on any DB
    failure the buffer just starts empty (behaves like a fresh process before Alembic ran)."""
    global _WARMED
    if _WARMED:
        return
    _WARMED = True  # set first: a failing warm must not retry on every call
    try:
        from .db import SessionLocal
        from .models import Measurement

        session = SessionLocal()
        try:
            # Last N rows, oldest-first, so appending preserves chronological order in the deque.
            rows = (session.query(Measurement)
                    .order_by(Measurement.created_at.desc(), Measurement.id.desc())
                    .limit(_BUFFER_MAX).all())
        finally:
            session.close()
        for r in reversed(rows):
            side = {"cart": {}, "rag": {}}
            m = {"latency_ms": r.latency_ms, "ttft_ms": r.ttft_ms,
                 "prompt_tokens": r.prompt_tokens, "resident_kv_tokens": r.resident_kv_tokens,
                 "gen_tokens": r.gen_tokens, "decode_tps": r.decode_tps,
                 "confidence": r.confidence, "measured": True}
            side[r.side if r.side in side else "cart"] = m
            _RECORDS.append(side)
    except Exception:  # noqa: BLE001 — a cold DB (pre-migration) just means an empty warm
        logger.warning("measurements: failed to warm buffer from DB (starting empty)", exc_info=True)


def record(cart: dict | None, rag: dict | None, *, tenant_id: str | None = None) -> None:
    """Append one measured head-to-head (each side is the {latency_ms, prompt_tokens, ...} metrics dict
    the Inference Service returned). Missing sides are tolerated. Updates the hot in-memory buffer AND
    durably persists the row(s); a DB failure never propagates to the caller.

    tenant_id (keyword-only, default None) attributes the persisted rows to the owning tenant for
    per-tenant billing. Corpus-scoped serve paths (chat / mcp / compare) pass their corpus's tenant;
    non-corpus callers omit it and the rows stay NULL (deployment-level, fleet-totals-only)."""
    with _LOCK:
        _warm_from_db()
        _RECORDS.append({"cart": cart or {}, "rag": rag or {}})
    _persist(cart, rag, tenant_id)  # outside the lock: the serve path shouldn't block on DB I/O


def summary() -> dict:
    """LAST measured query's metrics + measured $/query (latest record per side — the stream path
    records cart and rag as adjacent records for one question). {measured: False} until the first
    live query lands (or a warmed row exists), so callers can fall back to the modeled scenario and
    label it honestly."""
    from . import metrics
    with _LOCK:
        _warm_from_db()
        recs = list(_RECORDS)
    base = {"measured": False, "n": len(recs), "model": MODEL_LABEL, "instance": INSTANCE_LABEL}
    if not recs:
        return base

    def last(side: str) -> dict:
        for r in reversed(recs):
            if r.get(side):
                return r[side]
        return {}

    def num(d: dict, field: str):
        v = d.get(field)
        return v if isinstance(v, (int, float)) else None

    cart, rag = last("cart"), last("rag")
    cart_lat, rag_lat = num(cart, "latency_ms"), num(rag, "latency_ms")
    cart_prompt, rag_prompt = num(cart, "prompt_tokens"), num(rag, "prompt_tokens")
    cart_resident = num(cart, "resident_kv_tokens")
    # $/query LENGTH-NORMALIZED (measured TTFT + a standard-length answer at the measured decode
    # rate) so verbosity differences between the two answers can't pose as serving cost; falls
    # back to raw-latency pricing when ttft/decode aren't available.
    cart_q = (metrics.price_normalized(num(cart, "ttft_ms"), num(cart, "decode_tps"))
              or (metrics.price_from_latency(cart_lat) if cart_lat else None))
    rag_q = (metrics.price_normalized(num(rag, "ttft_ms"), num(rag, "decode_tps"))
             or (metrics.price_from_latency(rag_lat) if rag_lat else None))

    def _x(num, den):
        return round(num / den, 1) if num and den else None

    return {
        **base,
        "measured": bool(cart_lat or rag_lat),
        "cart": {"latency_ms": cart_lat, "prompt_tokens": cart_prompt,
                 "resident_kv_tokens": cart_resident,
                 "cost_per_query": round(cart_q, 6) if cart_q else None},
        "rag": {"latency_ms": rag_lat, "prompt_tokens": rag_prompt,
                "cost_per_query": round(rag_q, 6) if rag_q else None},
        "savings": {
            "faster_than_rag_x": _x(rag_lat, cart_lat),
            "fewer_prefill_tokens_x": _x(rag_prompt, cart_prompt),
            "cheaper_than_rag_x": _x(rag_q, cart_q),
        },
    }
