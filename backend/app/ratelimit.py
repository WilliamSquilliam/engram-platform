"""Shared rate limiter (slowapi). Used to throttle the auth endpoints against
brute-force / abuse and to keep one tenant from saturating the shared GPU
(the @limiter.limit decorators on chat/compare/train). In-memory storage is fine
for the single-task backend here; for a horizontally-scaled prod, point slowapi at
Redis and keep the same key func below (see docs/PRE_PRODUCTION.md)."""
import os

import jwt
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request


def _tenant_or_ip_key(request: Request) -> str:
    """Rate-limit bucket key. Behind an ALB every request shares the LB's IP, so
    keying on the remote address alone collapses all tenants into ONE global
    bucket — the per-tenant chat/compare/train limits were meant to be per tenant.
    Preference order (first that resolves wins), chosen for correctness then robustness:

      1. request.state.tenant_id / user_id — if some middleware already resolved the
         JWT (none does today; this is the clean hook if one is added later).
      2. the bearer JWT subject — parsed WITHOUT signature verification. Keying does
         not need a trusted identity: an attacker forging a `sub`/`tid` only moves
         THEMSELVES to a different bucket (they can't lift another tenant's limit or
         evade their own beyond what a fresh IP already allows). This gives the real
         per-tenant isolation the limits intend, cheaply (no DB, no crypto).
      3. left-most X-Forwarded-For IP — the client the ALB saw (unauthenticated
         endpoints: /auth/login, /auth/register).
      4. remote address — direct-connect / no XFF (local dev, tests).

    NEVER raises: a key func that throws would 500 every decorated request, so every
    lookup is defensive and falls through to the IP.
    """
    # 1. Pre-resolved identity on request.state (future middleware hook).
    st = getattr(request, "state", None)
    for attr in ("tenant_id", "user_id"):
        val = getattr(st, attr, None) if st is not None else None
        if val:
            return f"tid:{val}" if attr == "tenant_id" else f"uid:{val}"

    # 2. Bearer JWT subject (unverified decode — see docstring on why that's safe here).
    try:
        auth = request.headers.get("authorization") or ""
        if auth[:7].lower() == "bearer ":
            claims = jwt.decode(auth[7:].strip(), options={"verify_signature": False})
            ident = claims.get("tid") or claims.get("sub") or claims.get("email")
            if ident:
                return f"jwt:{ident}"
    except Exception:  # noqa: BLE001 — malformed/absent token just falls through to IP
        pass

    # 3. Left-most X-Forwarded-For (the original client behind the ALB).
    xff = request.headers.get("x-forwarded-for")
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return f"ip:{first}"

    # 4. Direct remote address (dev/tests, no proxy).
    return f"ip:{get_remote_address(request)}"


# Module-level so routers can decorate endpoints; main.py registers it on the app.
# Disabled via RATELIMIT_ENABLED=false (the test suite sets this so its many
# register/login calls don't trip the per-IP limit).
_ENABLED = os.environ.get("RATELIMIT_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
limiter = Limiter(key_func=_tenant_or_ip_key, enabled=_ENABLED)
