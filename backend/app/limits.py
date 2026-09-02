"""Beta usage limits — invisible until hit, graceful when hit.

Two caps, generous by design: documents per workspace (lifetime count) and served queries per calendar
month. Each cap resolves to an EFFECTIVE value: the tenant's per-workspace override when a platform_admin
set one (the "contact us to raise it" lever, PATCH /platform-admin/tenants/{id}/limits), otherwise the
global config default. 0 (or None) means UNLIMITED, so the cap is trivially disabled per-tenant or fleet-wide.

When a cap IS hit the check raises HTTP 429 with a structured, friendly body — plain words, no jargon,
so the UI can show it directly. Enforced BEFORE the work (upload / query) so nothing is half-committed.
"""
from fastapi import HTTPException

from . import config
from .models import Tenant

# Friendly copy shown to the user verbatim. Beta limits are generous and easy to raise, so the message
# says exactly that and points at the way to lift it — no dead end.
_DOC_MESSAGE = (
    "You have reached the beta limit for documents in your workspace. Beta limits are generous by "
    "design and easy to raise. Reach out and we will bump it right away."
)
_QUERY_MESSAGE = (
    "You have reached the beta limit for queries this month. Beta limits are generous by design and "
    "easy to raise. Reach out and we will bump it right away."
)


def _effective(override: int | None, default: int) -> int:
    """Resolve a cap: the tenant override wins when set (not None), else the config default. 0 stays 0
    (unlimited) either way — callers treat 0 as 'no cap'."""
    return override if override is not None else default


def effective_max_docs(tenant: Tenant) -> int:
    """Documents cap for this workspace: max_docs_override if set, else BETA_MAX_DOCS_PER_TENANT. 0 =
    unlimited."""
    return _effective(tenant.max_docs_override, config.BETA_MAX_DOCS_PER_TENANT)


def effective_max_queries(tenant: Tenant) -> int:
    """Monthly query cap for this workspace: max_queries_override if set, else
    BETA_MAX_QUERIES_PER_MONTH. 0 = unlimited."""
    return _effective(tenant.max_queries_override, config.BETA_MAX_QUERIES_PER_MONTH)


def check_document_limit(tenant: Tenant, current_docs: int, incoming: int = 0) -> None:
    """Raise 429 if this workspace's document count would exceed its cap once `incoming` more are added.
    Enforced at the upload route BEFORE saving anything, so a request that would cross the cap is rejected
    whole (never a partial upload). No-op when the cap is 0 (unlimited)."""
    cap = effective_max_docs(tenant)
    if cap and current_docs + incoming > cap:
        raise HTTPException(
            429,
            detail={"error": "beta_limit", "limit": "documents", "message": _DOC_MESSAGE},
        )


def check_query_limit(tenant: Tenant, queries_this_month: int) -> None:
    """Raise 429 if this workspace has already reached its monthly query cap. Enforced at chat dispatch
    BEFORE the engine is called (on both the stream and non-stream entries). No-op when the cap is 0
    (unlimited)."""
    cap = effective_max_queries(tenant)
    if cap and queries_this_month >= cap:
        raise HTTPException(
            429,
            detail={"error": "beta_limit", "limit": "queries", "message": _QUERY_MESSAGE},
        )
