"""Platform BILLING pricing — the single source of truth for turning usage aggregates into an
estimated $/period, for the tenant billing shell (E10) and the platform-admin cost-per-tenant
view (E11).

TWO METERS (pricing decision 2026-09-02):
  - memory:     $/onboarded-document/month  (BILL_PER_DOC_MONTH, default 0.03) — the recurring
                charge for keeping a document resident and served.
  - inference:  $/1,000 served queries       (BILL_PER_1K_QUERIES, default 1.30).

Onboarding is FREE: a one-time per-doc onboarding charge would tax the very action we want
customers to take, and the onboarded document is the moat — so the value is captured by the
RECURRING memory meter (a resident doc earns every month) rather than a one-time fee. Recurring
revenue beats one-time revenue, and free onboarding removes the friction on growing the document
base. price_per_onboarded_doc() is kept (returns 0.0) only so existing callers/compat don't break.

Storage is NOT billed: it's informational only (shown for context), so the old $/GB-month meter and
the metrics-anchored per-onboarded-doc margin derivation are gone — memory is priced per DOCUMENT,
which is the unit customers reason about, not per GB.

Distinct level from metrics.py: that module prices ONE query at the GPU-second level (the
demo/cost-comparison view). This module prices a WHOLE ACCOUNT at the plan level — the list-price
rate card a customer is billed on.

Billing is DISABLED during the beta (config.BILLING_ENABLED, default false): this rate card is what
the billing shell surfaces so pricing is visible in one place, but no charges are made and Stripe is
dark-launched (see routers/billing.py). All rates env-overridable (BILL_*), so ONE place changes the
rate card.
"""
import os

# --- rate card (list price) --------------------------------------------------
# $/1,000 queries served (the INFERENCE meter). A round beta list price, not a per-query GPU cost
# (that lives in metrics.price_normalized).
PRICE_PER_1K_QUERIES = float(os.environ.get("BILL_PER_1K_QUERIES", "1.30"))

# $/onboarded-document/month (the MEMORY meter): the recurring charge for keeping one document
# resident and served for a month. This is the moat — a resident doc earns every month.
PRICE_PER_DOC_MONTH = float(os.environ.get("BILL_PER_DOC_MONTH", "0.03"))


def price_per_onboarded_doc() -> float:
    """Onboarding is FREE — always 0.0. Kept only for backward compat with callers that referenced a
    per-onboarded-doc price; the value a document earns is now the recurring memory meter above, not a
    one-time onboarding fee."""
    return 0.0


def estimate_cost_usd(
    queries: int = 0,
    storage_gb: float = 0.0,
    documents: int = 0,
) -> float:
    """Estimated $/period from usage aggregates against the two-meter rate card. Signature is UNCHANGED
    (both dashboards call estimate_cost_usd(queries, storage_gb, documents)); the pricing is now:
      inference = queries / 1000 x per-1k-queries
      memory    = documents x per-doc-month   (recurring — one month priced here)
    storage_gb is accepted for signature compatibility but is INFORMATIONAL ONLY — storage is not
    billed. One function both dashboards call so the tenant bill and the platform cost-per-tenant
    number are computed the SAME way. Rounded to cents."""
    q_cost = (max(queries, 0) / 1000.0) * PRICE_PER_1K_QUERIES
    d_cost = max(documents, 0) * PRICE_PER_DOC_MONTH
    return round(q_cost + d_cost, 2)


def rate_card() -> dict:
    """The rate card the billing shell surfaces so the plan's pricing is visible in one place. Carries
    the new per_doc_month_usd key (the MEMORY meter); per_onboarded_doc_usd stays for compat and is
    always 0.0 (onboarding is free)."""
    return {
        "per_1k_queries_usd": round(PRICE_PER_1K_QUERIES, 4),
        "per_doc_month_usd": round(PRICE_PER_DOC_MONTH, 4),
        "per_onboarded_doc_usd": 0.0,  # onboarding is free
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
