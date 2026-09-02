"""Cost-comparison metrics. Authenticated: these aggregate deployment-wide query
volume and unit economics (lifetime counts, avg $/query, monthly breakdown) —
operational intel an invite-only beta must not publish to anonymous callers
(2026-09 security sweep). The marketing site carries its own static figures."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from .. import measurements, metrics
from ..deps import get_current_user, get_db
from ..models import Measurement, User

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/cost-comparison")
def cost_comparison(
    corpus_tokens: int = Query(1_000_000, ge=1_000, le=100_000_000),
    queries_per_month: int = Query(100_000, ge=1, le=100_000_000),
    _: User = Depends(get_current_user),
):
    """Cartridge (this platform) vs RAG on the same open model. `measured` = the real per-query latency
    / prefill / $ collected at run time on THIS deployment (empty until the first live query); `modeled`
    = the scenario projection for the corpus-size / volume sliders. The UI shows measured when present
    and labels the modeled projection as such."""
    return {**metrics.compare(corpus_tokens, queries_per_month), "measured": measurements.summary()}


def _month_bucket(dialect: str):
    """Portable 'YYYY-MM' bucket over Measurement.created_at. SQLite has strftime; Postgres has
    to_char — branch on the bound dialect so the monthly breakdown works in dev/tests AND prod."""
    if dialect == "postgresql":
        return func.to_char(Measurement.created_at, "YYYY-MM")
    # sqlite (dev/demo/tests) and anything else with strftime semantics
    return func.strftime("%Y-%m", Measurement.created_at)


@router.get("/savings")
def savings(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    """Deployment-level LIFETIME aggregates from the persisted measurements (authenticated — see
    module docstring): per-side totals (count, avg latency, avg $/query), the estimated
    cumulative savings of the cart path vs RAG (per-query cost delta x the number served), and a
    per-month breakdown. Empty/zeroed until the first live query has been recorded."""
    # Per-side rollup: count + averages. func.avg/func.count are portable across sqlite & postgres.
    rows = (db.query(
                Measurement.side,
                func.count(Measurement.id),
                func.avg(Measurement.latency_ms),
                func.avg(Measurement.cost_per_query),
            )
            .group_by(Measurement.side)
            .all())

    def _r(v, nd):
        return round(v, nd) if v is not None else None

    sides: dict[str, dict] = {}
    for side, n, avg_lat, avg_cost in rows:
        sides[side] = {
            "count": int(n or 0),
            "avg_latency_ms": _r(avg_lat, 1),
            "avg_cost_per_query": _r(avg_cost, 6),
        }
    cart = sides.get("cart", {"count": 0, "avg_latency_ms": None, "avg_cost_per_query": None})
    rag = sides.get("rag", {"count": 0, "avg_latency_ms": None, "avg_cost_per_query": None})

    # Estimated cumulative savings: the per-query cost delta (RAG - cart) applied over the number of
    # cart queries actually served. Only meaningful once both sides have a measured average cost.
    cart_q, rag_q = cart["avg_cost_per_query"], rag["avg_cost_per_query"]
    per_query_delta = (rag_q - cart_q) if (cart_q is not None and rag_q is not None) else None
    cumulative = (round(per_query_delta * cart["count"], 4)
                  if per_query_delta is not None else None)

    # Monthly breakdown: one bucket per calendar month, per side, with counts + avg cost.
    bucket = _month_bucket(db.bind.dialect.name)
    monthly_rows = (db.query(
                        bucket.label("month"),
                        Measurement.side,
                        func.count(Measurement.id),
                        func.avg(Measurement.cost_per_query),
                    )
                    .group_by("month", Measurement.side)
                    .order_by("month")
                    .all())
    months: dict[str, dict] = {}
    for month, side, n, avg_cost in monthly_rows:
        m = months.setdefault(month, {"month": month,
                                      "cart": {"count": 0, "avg_cost_per_query": None},
                                      "rag": {"count": 0, "avg_cost_per_query": None}})
        if side in ("cart", "rag"):
            m[side] = {"count": int(n or 0), "avg_cost_per_query": _r(avg_cost, 6)}
    monthly = [months[k] for k in sorted(months)]

    return {
        "measured_on": {"model": measurements.MODEL_LABEL, "instance": measurements.INSTANCE_LABEL},
        "totals": {"cart": cart, "rag": rag},
        "savings": {
            "per_query_cost_delta": _r(per_query_delta, 6),
            "cumulative_savings": cumulative,
            "queries_served": cart["count"] + rag["count"],
        },
        "monthly": monthly,
    }
