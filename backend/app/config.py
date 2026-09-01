"""Backend configuration. Everything is env-overridable so the same image runs
locally (SQLite + filesystem) and on AWS (Postgres + S3) by changing env only."""
import os
from pathlib import Path

PLATFORM_ROOT = Path(__file__).resolve().parents[2]

# Environment management: load the platform env file BEFORE reading any setting.
# Precedence (lowest -> highest): .env.local (committed, laptop-runnable) -> .env
# (gitignored, your local overrides) -> real process env (docker-compose / ECS task
# definition). load_dotenv never overrides an already-set var, so AWS env always
# wins. Net: local = .env.local, AWS = real env / .env.aws. See README "Environments".
try:
    from dotenv import load_dotenv
    # Layered, override=False so the first value set wins: real process env >
    # .env (gitignored, your secrets/overrides) > .env.local (committed defaults).
    load_dotenv(PLATFORM_ROOT / ".env")
    load_dotenv(PLATFORM_ROOT / ".env.local")
except ImportError:
    pass

DATA_DIR = Path(os.environ.get("PLATFORM_DATA_DIR", PLATFORM_ROOT / ".data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR = Path(os.environ.get("PLATFORM_STORAGE_DIR", DATA_DIR / "storage"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# SQLite locally; set DATABASE_URL=postgresql+psycopg://... on AWS (RDS/Aurora).
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{(DATA_DIR / 'platform.db').as_posix()}")

# The torch-bearing ML service (train + onboarding + HF inference).
ML_SERVICE_URL = os.environ.get("ML_SERVICE_URL", "http://localhost:8001")

# Inference path for chat. "hf" (default) = the ml_service /query above (HF DynamicCache; retrieves
# internally; the validated local path). "vllm" = the separate vLLM Inference Service
# (platform/ml_service/vllm_inference.py) that serves resident-KV carts by doc_id — the control plane
# retrieves the doc_ids (RETRIEVAL_BACKEND) and hands them over. Keep "hf" locally; "vllm" in the
# GPU-served deployment.
INFERENCE_BACKEND = os.environ.get("INFERENCE_BACKEND", "hf").lower()
INFERENCE_SERVICE_URL = os.environ.get("INFERENCE_SERVICE_URL", "http://localhost:8002")

# ML-plane shared-token auth (DEFAULT "" = off). When set, ml_client attaches
# `Authorization: Bearer <ML_AUTH_TOKEN>` to EVERY request to the ML service (:8001) and the vLLM
# Inference Service (:8002); those services enforce the same token (see their _ml_auth middleware).
# Empty = today's behavior (no header, no auth). Both planes share one token — it's an internal,
# same-VPC trust boundary, not per-caller identity.
ML_AUTH_TOKEN = os.environ.get("ML_AUTH_TOKEN", "")
INFERENCE_TOPK = int(os.environ.get("INFERENCE_TOPK", "3"))
INFERENCE_MAX_TOKENS = int(os.environ.get("INFERENCE_MAX_TOKENS", "96"))
# Adaptive router: on the vLLM cart side, if the cart-alone answer's mean-token-logprob
# confidence drops below ADAPTIVE_THETA, escalate to the RAG backup (full retrieved context on
# the same engine) and flag it in the UI. "" / unset disables escalation (pure single-cart CAG).
ADAPTIVE_THETA = os.environ.get("ADAPTIVE_THETA", "")
# Control-plane retrieval backend: "bm25" (pure-python, zero-dep, the default) or "pgvector" (the
# architecture target, swapped behind retrieval.retrieve()).
RETRIEVAL_BACKEND = os.environ.get("RETRIEVAL_BACKEND", "bm25").lower()

# Where the ML worker posts training-progress heartbeats. Locally that's this same
# process; on AWS it's the control-plane's internal service DNS/ALB, reachable from
# the worker subnet. INTERNAL_API_TOKEN (empty = no check locally) guards that
# callback so only the worker can report progress.
BACKEND_INTERNAL_URL = os.environ.get("BACKEND_INTERNAL_URL", "http://localhost:8000")
INTERNAL_API_TOKEN = os.environ.get("INTERNAL_API_TOKEN", "")

JWT_SECRET = os.environ.get("JWT_SECRET", "dev-secret-change-me")
JWT_ALG = "HS256"
JWT_EXPIRE_MIN = int(os.environ.get("JWT_EXPIRE_MIN", "1440"))
# "Remember me on this device": the long-lived session minted only when login asks for it
# (remember_me form field). Default 30 days; the normal session stays JWT_EXPIRE_MIN.
JWT_REMEMBER_EXPIRE_MIN = int(os.environ.get("JWT_REMEMBER_EXPIRE_MIN", "43200"))

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")

# --- Google OIDC sign-in (optional) ---------------------------------------
# Set GOOGLE_CLIENT_ID + GOOGLE_CLIENT_SECRET (from a Google Cloud OAuth 2.0
# "Web application" client) to enable the "Continue with Google" button. When
# either is blank the button is hidden and only email/password sign-in is shown.
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.environ.get(
    "GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback"
)
GOOGLE_ENABLED = bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)

# --- source connectors (E2): OAuth creds gate a connector's availability ------
# The connector registry (connectors/registry.py) mirrors serving.py: filesystem is always
# available; google_drive and sharepoint render as "coming soon" (available=False) until their
# OAuth app credentials are configured here. Setting both id+secret for a provider flips it on with
# NO code change. OAuth is NOT implemented yet — these only gate the /connectors availability flag.
# Google Drive reuses the Google OAuth client (GOOGLE_CLIENT_ID/SECRET) shared with Google sign-in.
GDRIVE_CLIENT_ID = os.environ.get("GDRIVE_CLIENT_ID", GOOGLE_CLIENT_ID)
GDRIVE_CLIENT_SECRET = os.environ.get("GDRIVE_CLIENT_SECRET", GOOGLE_CLIENT_SECRET)
GDRIVE_ENABLED = bool(GDRIVE_CLIENT_ID and GDRIVE_CLIENT_SECRET)
# SharePoint / Microsoft Graph: an Entra ID (Azure AD) app registration.
SHAREPOINT_CLIENT_ID = os.environ.get("SHAREPOINT_CLIENT_ID", "")
SHAREPOINT_CLIENT_SECRET = os.environ.get("SHAREPOINT_CLIENT_SECRET", "")
SHAREPOINT_TENANT_ID = os.environ.get("SHAREPOINT_TENANT_ID", "")
SHAREPOINT_ENABLED = bool(
    SHAREPOINT_CLIENT_ID and SHAREPOINT_CLIENT_SECRET and SHAREPOINT_TENANT_ID
)

# --- auth backend: local JWT (default) or OIDC/JWKS (Keycloak/Cognito/...) -----
# local: our /auth/register+login mint HS256 JWTs verified with JWT_SECRET.
# oidc:  the IdP (e.g. Keycloak) issues RS256 JWTs; we verify them against its JWKS
#        and find-or-create the tenant+user from the token's email claim. The rest
#        of the app only depends on get_current_user, so the swap stays contained.
AUTH_BACKEND = os.environ.get("AUTH_BACKEND", "local").lower()
OIDC_ISSUER = os.environ.get("OIDC_ISSUER", "")  # e.g. https://kc.example.com/realms/cartridge
OIDC_AUDIENCE = os.environ.get("OIDC_AUDIENCE", "")  # expected `aud` claim (optional)
OIDC_JWKS_URL = os.environ.get("OIDC_JWKS_URL", "")  # defaults to {issuer}/protocol/openid-connect/certs
if AUTH_BACKEND == "oidc" and not OIDC_JWKS_URL and OIDC_ISSUER:
    OIDC_JWKS_URL = OIDC_ISSUER.rstrip("/") + "/protocol/openid-connect/certs"

# Where the browser is sent back after Google sign-in; the SPA reads the issued
# JWT from the URL fragment there.
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:3000")

# Signs the short-lived session cookie carrying OAuth state/nonce across the
# Google round-trip (Authlib requirement). Defaults to JWT_SECRET.
SESSION_SECRET = os.environ.get("SESSION_SECRET", JWT_SECRET)

# Training knobs for the local small-model run (override to scale up).
TRAIN_CART_TOKENS = int(os.environ.get("TRAIN_CART_TOKENS", "64"))
TRAIN_STEPS = int(os.environ.get("TRAIN_STEPS", "60"))
TRAIN_GRAD_ACCUM = int(os.environ.get("TRAIN_GRAD_ACCUM", "8"))
TRAIN_GEN_QS = int(os.environ.get("TRAIN_GEN_QS", "6"))

# Onboarding method: "train" = per-doc gradient descent (parity reference, the default that
# always works); "encoder" = amortized encoder (~54× cheaper, ~2 frozen forwards/doc) — needs an
# encoder.pt trained for the active base model, pointed to by ENCODER_CKPT (see cartridges.encoder).
ONBOARD_METHOD = os.environ.get("ONBOARD_METHOD", "train").lower()
ENCODER_CKPT = os.environ.get("ENCODER_CKPT", "")

# --- environment + production safety -------------------------------------
# ENV=production makes the app refuse to boot with insecure dev defaults and
# enables secure-cookie / strict-CORS behaviour.
ENV = os.environ.get("ENV", "development").lower()
IS_PROD = ENV in ("production", "prod")


def _flag(name: str, default: bool) -> bool:
    return os.environ.get(name, "true" if default else "false").strip().lower() in (
        "1", "true", "yes", "on",
    )


# STARTTLS for SMTP (on by default; ports 587/25). Defined here where _flag exists.
SMTP_STARTTLS = _flag("SMTP_STARTTLS", default=True)


# Open self-service registration. ON for local/dev; OFF in prod by default so a
# reachable API can't be used to spin up tenants — seed the single operator via
# BOOTSTRAP_ADMIN_* instead. Override with ALLOW_REGISTRATION if you really want it.
ALLOW_REGISTRATION = _flag("ALLOW_REGISTRATION", default=not IS_PROD)

# One-shot operator seed (used when open registration is disabled). If both are set
# and the user doesn't exist yet, it's created at startup. Inject via Secrets Manager.
BOOTSTRAP_ADMIN_EMAIL = os.environ.get("BOOTSTRAP_ADMIN_EMAIL", "")
BOOTSTRAP_ADMIN_PASSWORD = os.environ.get("BOOTSTRAP_ADMIN_PASSWORD", "")

# The founder / cross-tenant superuser. The bootstrap admin (above) is seeded as
# platform_admin automatically; set PLATFORM_ADMIN_EMAIL to also promote an
# already-existing user (e.g. one created via Google sign-in) to platform_admin at
# startup. Empty = only the bootstrap admin is a platform_admin.
PLATFORM_ADMIN_EMAIL = os.environ.get("PLATFORM_ADMIN_EMAIL", "")

# --- transactional email gating (E1) --------------------------------------
# Same placeholder pattern as serving/connectors: email is OFF by default so every
# flow (invites, password reset, access-request approval) is fully usable and
# testable with NO email provider configured. Backends:
#   none (default): don't send; the token/reset/invite LINK is returned in the API
#                   response body and logged, so the flow is completable by hand.
#   ses:            send via boto3 SES (already a dependency).
#   smtp:           send via stdlib smtplib.
# When ses/smtp is selected the link is sent by email and NOT leaked in the response.
EMAIL_BACKEND = os.environ.get("EMAIL_BACKEND", "none").lower()
EMAIL_FROM = os.environ.get("EMAIL_FROM", "no-reply@engram.local")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
# SMTP settings (only read when EMAIL_BACKEND=smtp).
SMTP_HOST = os.environ.get("SMTP_HOST", "localhost")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
# SMTP_STARTTLS is a bool flag; set after _flag() is defined below.

# How long invite / password-reset tokens stay valid (short expiries — security must).
INVITE_EXPIRE_HOURS = int(os.environ.get("INVITE_EXPIRE_HOURS", "168"))  # 7 days
PASSWORD_RESET_EXPIRE_HOURS = int(os.environ.get("PASSWORD_RESET_EXPIRE_HOURS", "1"))

# Upload limits (anti-DoS): reject any single document over MAX_UPLOAD_MB or a
# request whose documents sum past MAX_REQUEST_MB. Files are read into memory.
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "25"))
MAX_REQUEST_MB = int(os.environ.get("MAX_REQUEST_MB", "200"))

# Host header allow-list (Starlette TrustedHostMiddleware). Comma-separated; the
# middleware is only installed when this is set and does not contain "*". Behind the
# private SSM tunnel / internal ALB the Host is "localhost:<port>" and the ALB
# health check uses the task IP, so this is left as "*" (disabled) for the private
# deploy — set it to your real domain for a public production launch.
ALLOWED_HOSTS = [h.strip() for h in os.environ.get("ALLOWED_HOSTS", "").split(",") if h.strip()]

_DEV_SECRETS = {"dev-secret-change-me", "dev-session-secret-change-me", ""}


def validate() -> None:
    """Fail fast in production rather than boot with insecure defaults."""
    if not IS_PROD:
        return
    problems = []
    if JWT_SECRET in _DEV_SECRETS or len(JWT_SECRET) < 32:
        problems.append("JWT_SECRET must be a strong (>=32 char) secret")
    if SESSION_SECRET in _DEV_SECRETS or len(SESSION_SECRET) < 32:
        problems.append("SESSION_SECRET must be a strong (>=32 char) secret")
    if DATABASE_URL.startswith("sqlite"):
        problems.append("DATABASE_URL must point at Postgres in production")
    if any("localhost" in o for o in CORS_ORIGINS):
        problems.append("CORS_ORIGINS must not include localhost in production")
    if "*" in CORS_ORIGINS:
        # allow_credentials=True + wildcard origin is an XSRF/credential-leak setup.
        problems.append("CORS_ORIGINS must list explicit origins (no '*') in production")
    if not INTERNAL_API_TOKEN or len(INTERNAL_API_TOKEN) < 32:
        # The worker -> control-plane progress callback is unauthenticated without it.
        problems.append("INTERNAL_API_TOKEN must be a strong (>=32 char) secret in production")
    if EMAIL_BACKEND == "none":
        # With EMAIL_BACKEND=none the invite/reset/approval LINK is returned in the API response
        # (dev convenience). In prod that leaks account-takeover tokens to any caller — a real
        # provider (ses/smtp) must be configured so links are emailed, not echoed.
        problems.append("EMAIL_BACKEND must not be 'none' in production (invite/reset links would leak)")
    if not ML_AUTH_TOKEN or len(ML_AUTH_TOKEN) < 32:
        # The control plane <-> ML/vLLM planes share this bearer token; unset = the ML endpoints
        # (train, onboard, inference) are reachable on the VPC with no auth.
        problems.append("ML_AUTH_TOKEN must be a strong (>=32 char) secret in production")
    if problems:
        raise RuntimeError("Invalid production config:\n  - " + "\n  - ".join(problems))
