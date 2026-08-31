"""Control-plane API entrypoint. Stateless FastAPI app → scales out on ECS/EKS."""
import logging

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy import text
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import config
from .db import engine, init_db
from .logging_config import setup_logging
from .ratelimit import limiter
from .routers import (
    audit, auth, chat, compare, connectors, corpora, economics, jobs, metrics, model_tiers,
    onboarding,
)

setup_logging()
config.validate()  # fail fast on insecure prod config
logger = logging.getLogger(__name__)

app = FastAPI(title="Cartridge KV Platform — Control Plane")

# Auth-endpoint rate limiting (slowapi): the limiter is attached here; individual
# routes opt in with @limiter.limit (see routers/auth.py).
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Host-header allow-list in prod (disabled when ALLOWED_HOSTS is unset or "*").
if config.IS_PROD and config.ALLOWED_HOSTS and "*" not in config.ALLOWED_HOSTS:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=config.ALLOWED_HOSTS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Baseline hardening headers on every response. HSTS only in prod (behind TLS)."""
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    if config.IS_PROD:
        resp.headers.setdefault(
            "Strict-Transport-Security", "max-age=63072000; includeSubDomains"
        )
    return resp

# Signed session cookie carrying OAuth state/nonce across the Google round-trip
# (Authlib requirement). same_site="lax" lets the cookie ride the redirect back
# from Google; https_only is enforced in production (behind TLS).
app.add_middleware(
    SessionMiddleware,
    secret_key=config.SESSION_SECRET,
    same_site="lax",
    https_only=config.IS_PROD,
)

init_db()


def _bootstrap_admin() -> None:
    """Seed a single operator account when open registration is disabled (prod).
    Idempotent: no-op if the user already exists or the env vars are unset."""
    if not (config.BOOTSTRAP_ADMIN_EMAIL and config.BOOTSTRAP_ADMIN_PASSWORD):
        return
    from .db import SessionLocal
    from .models import Tenant, User
    from .security import hash_password

    db = SessionLocal()
    try:
        if db.query(User).filter(User.email == config.BOOTSTRAP_ADMIN_EMAIL).first():
            return
        tenant = Tenant(name="Admin workspace")
        db.add(tenant)
        db.flush()
        db.add(User(
            tenant_id=tenant.id,
            email=config.BOOTSTRAP_ADMIN_EMAIL,
            hashed_password=hash_password(config.BOOTSTRAP_ADMIN_PASSWORD),
        ))
        db.commit()
        logger.info("Bootstrapped admin user %s", config.BOOTSTRAP_ADMIN_EMAIL)
    finally:
        db.close()


_bootstrap_admin()

app.include_router(auth.router)
app.include_router(corpora.router)
app.include_router(jobs.router)
app.include_router(chat.router)
app.include_router(compare.router)
app.include_router(economics.router)
app.include_router(metrics.router)
app.include_router(audit.router)
app.include_router(model_tiers.router)
app.include_router(connectors.router)
app.include_router(onboarding.router)


@app.get("/health")
def health():
    """Liveness: the process is up."""
    return {"ok": True}


@app.get("/ready")
def ready():
    """Readiness: dependencies (DB, ML service) are reachable. 503 if not, so an
    orchestrator (ECS/K8s) holds traffic until the app can actually serve."""
    checks: dict[str, str] = {}
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["db"] = "ok"
    except Exception:  # noqa: BLE001
        checks["db"] = "fail"
    try:
        r = httpx.get(f"{config.ML_SERVICE_URL}/health", timeout=2.0)
        checks["ml_service"] = "ok" if r.status_code == 200 else "fail"
    except Exception:  # noqa: BLE001
        checks["ml_service"] = "fail"
    # The vLLM Inference Service is the serve path the whole product rides on — readiness
    # without it meant the app reported ready while every chat/compare would 502.
    if config.INFERENCE_BACKEND == "vllm":
        try:
            r = httpx.get(f"{config.INFERENCE_SERVICE_URL}/health", timeout=2.0)
            body = r.json() if r.status_code == 200 else {}
            checks["inference"] = "ok" if body.get("engine_ready") else "warming"
        except Exception:  # noqa: BLE001
            checks["inference"] = "fail"
    ok = all(v in ("ok", "warming") for v in checks.values())  # warming = up, first build in progress
    return JSONResponse(checks, status_code=200 if ok else 503)
