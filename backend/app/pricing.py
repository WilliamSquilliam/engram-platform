"""Platform BILLING pricing — the single source of truth for turning usage aggregates
(queries served, storage held, documents onboarded) into an estimated $/period, for the
tenant billing shell (E10) and the platform-admin cost-per-tenant view (E11).

Distinct level from metrics.py: that module prices ONE query at the GPU-second level (the
demo/cost-comparison view). This module prices a WHOLE ACCOUNT at the plan level — the list-price
rate card a customer is billed on. It REUSES metrics.py rather than re-deriving GPU numbers: the
per-onboarded-doc figure is anchored to the same measured GPU $/doc the onboarding estimate uses
(metrics.onboard_estimate), so a tweak to the GPU rate flows through to billing automatically.

All rates env-overridable (BILL_* ), so ONE place changes the rate card. Numbers are list-price
placeholders for the beta (Stripe is deferred) — honest, in one table, nothing scattered."""
import os

from . import metrics

# --- rate card (list price) --------------------------------------------------
# $/1,000 queries served. Covers the amortized serve GPU + platform overhead; a round
# beta list price, not a per-query GPU cost (that lives in metrics.price_normalized).
PRICE_PER_1K_QUERIES = float(os.environ.get("BILL_PER_1K_QUERIES", "2.00"))

# $/GB-month of stored corpus data (raw documents + cartridges held resident).
PRICE_PER_GB_MONTH = float(os.environ.get("BILL_PER_GB_MONTH", "0.50"))

# $/onboarded document. Anchored to the measured onboarding GPU cost (metrics.onboard_estimate
# for one doc) x a margin multiplier, so it never drifts from the real GPU number underneath.
ONBOARD_MARGIN = float(os.environ.get("BILL_ONBOARD_MARGIN", "3.0"))


def price_per_onboarded_doc() -> float:
    """$/onboarded doc = the measured GPU cost to onboard one document (metrics, on-demand)
    x the billing margin. Single derivation so the GPU rate is the only knob underneath."""
    gpu_cost_per_doc = metrics.onboard_estimate(1)["est_cost_ondemand"]
    return gpu_cost_per_doc * ONBOARD_MARGIN


def estimate_cost_usd(
    queries: int = 0,
    storage_gb: float = 0.0,
    documents: int = 0,
) -> float:
    """Estimated $/period from usage aggregates against the rate card above. One function both
    dashboards call so the tenant bill and the platform cost-per-tenant number are computed the
    SAME way. Rounded to cents."""
    q_cost = (max(queries, 0) / 1000.0) * PRICE_PER_1K_QUERIES
    s_cost = max(storage_gb, 0.0) * PRICE_PER_GB_MONTH
    d_cost = max(documents, 0) * price_per_onboarded_doc()
    return round(q_cost + s_cost + d_cost, 2)


def rate_card() -> dict:
    """The rate card the billing shell surfaces so the plan's pricing is visible in one place."""
    return {
        "per_1k_queries_usd": round(PRICE_PER_1K_QUERIES, 4),
        "per_gb_month_usd": round(PRICE_PER_GB_MONTH, 4),
        "per_onboarded_doc_usd": round(price_per_onboarded_doc(), 4),
        "currency": "usd",
    }


# --- plan definitions (beta) -------------------------------------------------
# Per-plan soft limits the billing shell shows. Beta is generous/uncapped-in-practice; real
# enforcement + Stripe metering land later. Kept here so plan limits and rates live together.
PLAN_LIMITS: dict[str, dict] = {
    "beta": {"queries": None, "storage_gb": None, "documents": None, "seats": None},
    "starter": {"queries": 50_000, "storage_gb": 25, "documents": 1_000, "seats": 5},
    "growth": {"queries": 500_000, "storage_gb": 250, "documents": 20_000, "seats": 25},
}


def plan_limits(plan: str) -> dict:
    """Limits for a plan; unknown plans fall back to the beta (uncapped) row."""
    return PLAN_LIMITS.get(plan, PLAN_LIMITS["beta"])
