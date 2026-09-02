"""Stripe billing — DARK-LAUNCHED: fully wired, DISABLED by config.BILLING_ENABLED.

Two meters (see pricing.py): memory ($/onboarded-doc/month) and inference ($/1k queries). The
internal usage tables (usage.py / measurements) REMAIN the source of truth for what a tenant used;
Stripe is only the RATING layer that turns those numbers into an invoice. So this router never derives
a charge itself — it pushes meter events from the usage tables and lets Stripe rate them.

Everything degrades safely when disabled: GET /billing/status is always a 200 with enabled:false;
POST /billing/portal 503s; the webhook 503s; the usage reporter no-ops. The stripe SDK is imported
LAZILY inside the enabled branches so the dependency is never touched (and tests never hit the network)
while billing is off.
"""
import logging
import secrets

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from .. import config, pricing, usage
from ..deps import get_db, require_tenant_admin
from ..models import Tenant, User
from ..schemas import BillingPortalResp, BillingStatusResp

logger = logging.getLogger(__name__)
router = APIRouter(tags=["billing"])


def _init_stripe():
    """Lazily import + configure the official Stripe SDK. Kept out of module import so the dependency
    is only touched when billing is actually enabled (and never in the disabled/test path)."""
    import stripe  # official SDK — never hand-roll Stripe HTTP

    stripe.api_key = config.STRIPE_SECRET_KEY
    return stripe


@router.get("/billing/status", response_model=BillingStatusResp)
def billing_status(
    admin: User = Depends(require_tenant_admin), db: Session = Depends(get_db)
):
    """Billing status for the caller's workspace — ALWAYS 200, safe when billing is disabled. Surfaces
    the flag, the rate card (so pricing is visible in one place), and whether the manage-billing portal
    is available for this tenant."""
    tenant = db.get(Tenant, admin.tenant_id)
    enabled = config.BILLING_ENABLED
    # Portal is offered whenever billing is enabled (the portal call lazily creates the customer if the
    # tenant doesn't have one yet, so a null stripe_customer_id doesn't block it).
    portal_available = bool(enabled and tenant and tenant.stripe_customer_id) or enabled
    return BillingStatusResp(
        enabled=enabled,
        rate_card=pricing.rate_card(),
        portal_available=portal_available,
    )


@router.post("/billing/portal", response_model=BillingPortalResp)
def billing_portal(
    admin: User = Depends(require_tenant_admin), db: Session = Depends(get_db)
):
    """Open the Stripe billing portal for the caller's workspace. 503 while billing is disabled (the
    beta). When enabled: lazily create the Stripe customer for the tenant if it has none yet (stamped
    with tenant_id metadata + persisted), then create a billing-portal session and return its URL."""
    if not config.BILLING_ENABLED:
        raise HTTPException(503, "Billing is not enabled during the beta.")
    stripe = _init_stripe()
    tenant = db.get(Tenant, admin.tenant_id)
    if tenant is None:
        raise HTTPException(404, "Workspace not found")
    if not tenant.stripe_customer_id:
        customer = stripe.Customer.create(
            name=tenant.name,
            metadata={"tenant_id": tenant.id},
        )
        tenant.stripe_customer_id = customer.id
        db.commit()
    session = stripe.billing_portal.Session.create(
        customer=tenant.stripe_customer_id,
        return_url=config.FRONTEND_URL + "/admin",
    )
    return BillingPortalResp(url=session.url)


@router.post("/billing/webhook")
async def billing_webhook(
    request: Request,
    stripe_signature: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    """Stripe -> control-plane webhook. NO auth dependency: Stripe calls it, and authenticity is proven
    by the signature (verified with stripe.Webhook.construct_event against STRIPE_WEBHOOK_SECRET), not a
    user JWT. 503 while billing is disabled; 400 on a bad/absent signature. Dark-launch SKELETON: it
    logs invoice.paid / invoice.payment_failed with the tenant mapping and acks everything else 200 —
    no business logic yet."""
    if not config.BILLING_ENABLED:
        raise HTTPException(503, "Billing is not enabled during the beta.")
    stripe = _init_stripe()
    payload = await request.body()
    try:
        event = stripe.Webhook.construct_event(
            payload, stripe_signature or "", config.STRIPE_WEBHOOK_SECRET
        )
    except Exception as exc:  # noqa: BLE001 — any verification failure is a 400 (bad signature)
        raise HTTPException(400, "Invalid Stripe signature") from exc

    etype = event.get("type")
    if etype in ("invoice.paid", "invoice.payment_failed"):
        obj = (event.get("data") or {}).get("object") or {}
        customer_id = obj.get("customer")
        tenant = (
            db.query(Tenant).filter(Tenant.stripe_customer_id == customer_id).first()
            if customer_id else None
        )
        logger.info(
            "Stripe webhook %s for customer %s (tenant %s)",
            etype, customer_id, tenant.id if tenant else "unknown",
        )
    # Everything else is acknowledged 200 with no action (skeleton).
    return {"received": True}


def _report_usage(db: Session) -> dict:
    """Push meter events to Stripe for every tenant that has a stripe_customer_id. Two meters:
      - memory:    the tenant's CURRENT onboarded-doc count (one event per tenant per call; Stripe's
                   meter aggregates over the period).
      - inference: the queries served SINCE the last report — delta = tenant_query_count - the persisted
                   high-water mark, clamped >= 0, and the mark is advanced ONLY after a successful send
                   (so a failed push retries the same delta next time, never loses or double-counts).
    The usage tables stay the source of truth; Stripe only rates these numbers."""
    stripe = _init_stripe()
    reported = 0
    tenants = db.query(Tenant).filter(Tenant.stripe_customer_id.isnot(None)).all()
    for t in tenants:
        docs = usage.tenant_document_count(db, t.id)
        total_queries = usage.tenant_query_count(db, t.id)
        delta = max(0, total_queries - (t.billing_reported_queries or 0))
        # Memory meter: current resident-doc count. Inference meter: only the new queries since last mark.
        stripe.billing.MeterEvent.create(
            event_name=config.STRIPE_PRICE_MEMORY_ID,
            payload={"stripe_customer_id": t.stripe_customer_id, "value": str(docs)},
        )
        if delta:
            stripe.billing.MeterEvent.create(
                event_name=config.STRIPE_PRICE_INFERENCE_ID,
                payload={"stripe_customer_id": t.stripe_customer_id, "value": str(delta)},
            )
        # Advance the high-water mark only after the sends above succeeded.
        t.billing_reported_queries = total_queries
        reported += 1
    db.commit()
    return {"reported": True, "tenants": reported}


@router.post("/internal/billing/report-usage")
def report_usage(
    x_internal_token: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    """Internal usage-reporter (scheduled/operator-invoked; NOT user-facing) — guarded by the shared
    internal token, constant-time compared, exactly like routers/jobs.py's internal routes.

    When billing is disabled -> 200 {"reported": false, "reason": "billing_disabled"} (a clean no-op so
    a scheduler can call it harmlessly during the beta). When enabled: fail CLOSED if INTERNAL_API_TOKEN
    is unset (503, mirrors gc_carts) and 401 on a wrong token, then push the meter events."""
    if not config.BILLING_ENABLED:
        return {"reported": False, "reason": "billing_disabled"}
    if not config.INTERNAL_API_TOKEN:
        raise HTTPException(503, "usage reporting requires INTERNAL_API_TOKEN")
    if not secrets.compare_digest(config.INTERNAL_API_TOKEN, x_internal_token or ""):
        raise HTTPException(401, "invalid internal token")
    return _report_usage(db)
