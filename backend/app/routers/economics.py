"""Per-corpus training economics for the workspace 'Costs' view: what the last
training run cost (measured GPU wall-clock x GPU $/hr, on-demand + spot) and the
break-even — how many queries until that one-time cost is repaid by the per-query
saving vs RAG (the only realistic baseline). Per-query costs use the same model as /demo.
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from .. import measurements, metrics
from ..deps import get_current_user, get_db
from ..models import User
from .corpora import get_owned_corpus  # reuse tenant-scoped ownership check

router = APIRouter(tags=["economics"])


def _round_q(x: float | None) -> int | None:
    return round(x) if x is not None else None


@router.get("/corpora/{corpus_id}/economics")
def economics(
    corpus_id: str,
    queries_per_month: int = Query(100_000, ge=1, le=100_000_000),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    corpus = get_owned_corpus(db, user, corpus_id)
    n = queries_per_month
    corpus_tokens = corpus.corpus_tokens or 0
    n_carts = corpus.n_cartridges or 0
    train_s = corpus.train_seconds

    tc_on = metrics.training_cost(train_s, metrics.GPU_HOURLY_ONDEMAND)
    tc_spot = metrics.training_cost(train_s, metrics.GPU_HOURLY_SPOT)

    # Prefer per-query costs MEASURED on the live serving path; fall back to the model before any run.
    meas = measurements.summary()
    if meas.get("measured") and meas["cart"].get("cost_per_query") and meas["rag"].get("cost_per_query"):
        cart_q, rag_q, measured = meas["cart"]["cost_per_query"], meas["rag"]["cost_per_query"], True
    else:
        cart_q, rag_q, measured = metrics.price_everyday(), metrics.rag_cost(n), False

    return {
        "trained": train_s is not None,
        "n_cartridges": n_carts,
        "corpus_tokens": corpus_tokens,
        "train_seconds": train_s,
        "gpu_hourly_ondemand": metrics.GPU_HOURLY_ONDEMAND,
        "gpu_hourly_spot": metrics.GPU_HOURLY_SPOT,
        "train_cost_ondemand": round(tc_on, 4),
        "train_cost_spot": round(tc_spot, 4),
        "cost_per_cart_ondemand": round(tc_on / n_carts, 5) if n_carts else None,
        "queries_per_month": n,
        # RAG is the only realistic baseline (full-corpus / frontier prefill dropped from the UI).
        "per_query": {
            "everyday": round(cart_q, 6),
            "rag": round(rag_q, 6),
        },
        "per_query_measured": measured,
        "breakeven_vs_rag": _round_q(metrics.breakeven_queries(tc_on, rag_q, cart_q)),
    }
