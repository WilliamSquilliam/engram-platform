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
# Server-side answer-length CEILING (a guardrail, not the expected stopping mechanism — the model
# stops at EOS well before it on short answers). 96 -> 256 (2026-09-03, with the founder): 96 was
# benchmark-sized and truncated real answers mid-deliberation; the QRC 256-token diagnostic measured
# k=3 accuracy 13/15 -> 14/15 from this alone. Clients may request LESS per call (ChatReq.max_tokens,
# clamped here); worst case ~10s decode/request on the current box, bounded GPU time per query.
INFERENCE_MAX_TOKENS = int(os.environ.get("INFERENCE_MAX_TOKENS", "256"))
# Adaptive router: on the vLLM cart side, if the cart-alone answer's mean-token-logprob
# confidence drops below ADAPTIVE_THETA, escalate to the RAG backup (full retrieved context on
# the same engine) and flag it in the UI. "" / unset disables escalation (pure single-cart CAG).
ADAPTIVE_THETA = os.environ.get("ADAPTIVE_THETA", "")
# --- QRC serving (k>1 accuracy fix; PLAN decisions 2026-09-03) --------------------------------
# Composing k>1 solo-encoded carts interferes REGARDLESS of compression, so k>1 serving keeps the
# TOP-1 doc as the resident cart and routes the other docs' answer-bearing chunks (app/chunking.py,
# byte-identical core to what the bench measures).
#   QRC_MODE=resident (DEFAULT — promoted with the founder 2026-09-03 after the official run:
#                      accuracy 13/15 == hybrid == RAG, zero span-load errors, and the sweep anchors
#                      were measured in this mode: 3.53 qps @conc128, 2.2x the retired multi-cart)
#                      -> top-1 full cart + docs 2..k loaded as KV SPANS (wheel >= 0.7.0 required).
#   QRC_MODE=hybrid   -> same evidence but docs 2..k's chunks re-prefilled as text context (no wheel
#                        0.7.0 dependency — the fallback if a serving box runs an older wheel).
#   QRC_MODE=off      -> legacy multi-cart serve (every retrieved doc_id handed over, no context).
QRC_MODE = os.environ.get("QRC_MODE", "resident").lower()
# Chunk-description sidecar generation at onboard (one cart-resident generation per doc yields a short
# line per chunk, folded into the chunk's INDEX text as routing metadata — never served). Only runs on
# the vllm backend when QRC_MODE=hybrid (see routers/jobs.py). DEFAULT OFF (decided with the founder,
# 2026-09-03): measured at 16.7 s/doc — 6x the 2.72 s/doc onboarding headline — while the validated
# 13/15 routing accuracy came mostly from raw chunk text (4 of 13 sidecar docs had failed to parse and
# routing didn't suffer). Flip on per-env once the generation is cheaper (bigger chunks -> fewer lines,
# or piggyback on the existing doc-description call).
QRC_CHUNK_DESC = os.environ.get("QRC_CHUNK_DESC", "off").lower()
# Token budget for the routed context PER doc. The canonical default lives in chunking.CHUNK_BUDGET_TOKENS
# (one source of truth shared with the bench); this is only an optional env override for tuning without
# touching the shared core.
from . import chunking as _chunking  # noqa: E402 — after os import; pure module, no app/torch imports
QRC_BUDGET_TOKENS = int(os.environ.get("QRC_BUDGET_TOKENS", str(_chunking.CHUNK_BUDGET_TOKENS)))

# Control-plane retrieval backend: "hybrid" (the DEFAULT — industry-standard bm25s lexical + fastembed
# dense fused by RRF, in-process, torch-free) or "fused" (the GPU-box BM25+dense+rerank pipeline).
# "bm25" is kept as a legacy alias -> hybrid (the hand-rolled pure-python scorer it named is replaced;
# hybrid runs lexical-only when the dense stage is off, which reproduces bm25-only behavior). "pgvector"
# stays a documented seam. All swap behind retrieval.retrieve() — the rest of the app only sees retrieve().
RETRIEVAL_BACKEND = os.environ.get("RETRIEVAL_BACKEND", "hybrid").lower()
# --- Dynamic top-k: keep a variable number of retrieved docs by RELEVANCE ratio (dense cosine when
# available, else raw bm25 — never the rank-derived RRF scores), fused order preserved, capped at the
# requested k, always keeping >= 1. "on" (DEFAULT — promoted with the founder 2026-09-03: ratio 0.85
# measured 14/15 at mean k 2.4 on the official suite; cutting weak runner-ups never hurt and is
# strictly cheaper). "off" = flat top-k. Lower ratio admits more docs; higher keeps only close ones.
RETRIEVAL_DYNAMIC_K = os.environ.get("RETRIEVAL_DYNAMIC_K", "on").lower()
RETRIEVAL_DYNK_RATIO = float(os.environ.get("RETRIEVAL_DYNK_RATIO", "0.85"))
# Dense stage of the hybrid retriever: "on" (default) builds the fastembed embedder lazily; "off"
# runs hybrid lexical-only (bm25s), skipping any model download. Tests set "off" so they never fetch
# a model. Hybrid ALSO degrades to lexical-only automatically if fastembed import/model-load fails.
RETRIEVAL_DENSE = os.environ.get("RETRIEVAL_DENSE", "on").lower()
# Dense embedding model for the hybrid retriever's dense stage (fastembed / ONNX, torch-free).
RETRIEVAL_DENSE_MODEL = os.environ.get("RETRIEVAL_DENSE_MODEL", "BAAI/bge-small-en-v1.5")
# fastembed downloads its ONNX model on first use; keep it under DATA_DIR so it's cached across
# process restarts instead of re-downloaded (one dir shared by every worker on the box).
FASTEMBED_CACHE_DIR = Path(os.environ.get("FASTEMBED_CACHE_DIR", DATA_DIR / "fastembed"))

# --- Platform-admin GPU controls: Lambda Cloud serving box (E12) ------------------------------
# The GPU serving box is a Lambda Cloud instance. Lambda has NO stop state: "stop" == terminate
# (billing $0), "start" == launch a fresh box. A persistent filesystem (LAMBDA_FS_NAME) holds the
# model weights + the self-provision bundle, and a cloud-init user_data script provisions the fresh
# box unattended, so terminate/relaunch is the intended flow. GPU_CONTROL_ENABLED gates the whole
# feature: with no LAMBDA_API_KEY the controls report offline (still 200) and start/stop 503.
LAMBDA_API_KEY = os.environ.get("LAMBDA_API_KEY", "")
LAMBDA_API_BASE = os.environ.get("LAMBDA_API_BASE", "https://cloud.lambda.ai/api/v1")
LAMBDA_INSTANCE_NAME = os.environ.get("LAMBDA_INSTANCE_NAME", "engram-serving")
LAMBDA_FS_NAME = os.environ.get("LAMBDA_FS_NAME", "engram-fs")
# Preferred instance-type name substring (e.g. "b200"); LAMBDA_TYPE_FALLBACK is the exact type name
# used when nothing matches the filter. The launch region must ALSO host LAMBDA_FS_NAME (the weights +
# self-provision bundle live there — launching elsewhere gives an unprovisioned box).
LAMBDA_TYPE_FILTER = os.environ.get("LAMBDA_TYPE_FILTER", "b200")
LAMBDA_TYPE_FALLBACK = os.environ.get("LAMBDA_TYPE_FALLBACK", "2x_h100_sxm")
LAMBDA_SSH_KEY_NAME = os.environ.get("LAMBDA_SSH_KEY_NAME", "engram-lambda")

# Cloudflare DNS: point the serve/onboard hostnames at the fresh box's IP after a launch. Best-effort
# and optional — with no token/zone the status read reports dns_pointed=null and skips reconcile.
# Accept the legacy CLOUDFLARE_API_KEY name as a fallback so an existing env doesn't need renaming.
CLOUDFLARE_API_TOKEN = os.environ.get(
    "CLOUDFLARE_API_TOKEN", os.environ.get("CLOUDFLARE_API_KEY", "")
)
CLOUDFLARE_ZONE_ID = os.environ.get("CLOUDFLARE_ZONE_ID", "")

# Master switch for the GPU controls, derived from whether an API key is configured.
GPU_CONTROL_ENABLED = bool(LAMBDA_API_KEY)


# --- LLM document descriptions at onboarding (Feature 1) --------------------------------------
# DESCRIBE_MAX_TOKENS caps each one-sentence description generation. The DEFAULT-ON flag
# DOC_DESCRIPTIONS_ENABLED needs _flag() (defined below), so it's set further down. The per-doc time
# estimate constant lives in metrics.py (single source of truth for the review-step sizing).
DESCRIBE_MAX_TOKENS = int(os.environ.get("DESCRIBE_MAX_TOKENS", "60"))

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

# --- source connectors (E2): OAuth creds + an encryption key gate availability ------
# The connector registry (connectors/registry.py) mirrors serving.py: filesystem is always
# available; google_drive and sharepoint render as "coming soon" (available=False) until their
# OAuth app credentials AND the token-encryption key (CONNECTOR_ENC_KEY, below) are configured.
# Setting a provider's id+secret + the enc key flips it on with NO code change.
# Google Drive reuses the Google OAuth client (GOOGLE_CLIENT_ID/SECRET) shared with Google sign-in.
GDRIVE_CLIENT_ID = os.environ.get("GDRIVE_CLIENT_ID", GOOGLE_CLIENT_ID)
GDRIVE_CLIENT_SECRET = os.environ.get("GDRIVE_CLIENT_SECRET", GOOGLE_CLIENT_SECRET)
# SharePoint / Microsoft Graph: an Entra ID (Azure AD) app registration. The app must be
# multi-tenant so any customer M365 org can consent — the OAuth authority is
# "organizations" by default; SHAREPOINT_TENANT_ID optionally pins a single tenant for testing.
SHAREPOINT_CLIENT_ID = os.environ.get("SHAREPOINT_CLIENT_ID", "")
SHAREPOINT_CLIENT_SECRET = os.environ.get("SHAREPOINT_CLIENT_SECRET", "")
SHAREPOINT_TENANT_ID = os.environ.get("SHAREPOINT_TENANT_ID", "")

# Fernet key (urlsafe base64, 32 bytes) that encrypts OAuth refresh/access tokens AT REST in the
# connector_connections table. Generate with:
#   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# When UNSET a connector reports unavailable exactly like missing OAuth creds — tokens are NEVER
# stored in plaintext. Rotating this key invalidates existing connections (decrypt then 503s asking
# the user to reconnect); the connect flow is idempotent so reconnecting simply re-encrypts.
CONNECTOR_ENC_KEY = os.environ.get("CONNECTOR_ENC_KEY", "")

# Availability = the registry's IMPLEMENTED flag AND creds AND the enc key. Creds alone (or the key
# alone) must never flip a connector on — a connection with no way to decrypt its tokens is useless.
GDRIVE_ENABLED = bool(GDRIVE_CLIENT_ID and GDRIVE_CLIENT_SECRET and CONNECTOR_ENC_KEY)
# TENANT_ID is an OPTIONAL single-tenant override, NOT required for availability (the app is
# multi-tenant against "organizations"), so it's intentionally left out of the gate.
SHAREPOINT_ENABLED = bool(
    SHAREPOINT_CLIENT_ID and SHAREPOINT_CLIENT_SECRET and CONNECTOR_ENC_KEY
)

# Public base URL of THIS control-plane API — used to build each connector's OAuth redirect URI
# ({API_BASE_URL}/connectors/{provider}/callback), which the operator registers with the provider.
# Defaults to the local dev API; set to the real https API origin in production.
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

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

# LLM doc descriptions at onboarding (Feature 1), DEFAULT ON. After a wizard onboard the serving model
# is asked for a one-sentence description of each doc, stored on the Document row (best-effort — a
# failure never fails onboarding). Defined here (not at the top block) because _flag exists from here.
DOC_DESCRIPTIONS_ENABLED = _flag("DOC_DESCRIPTIONS_ENABLED", default=True)


# Open self-service registration. ON for local/dev; OFF in prod by default so a
# reachable API can't be used to spin up tenants — seed the single operator via
# BOOTSTRAP_ADMIN_* instead. Override with ALLOW_REGISTRATION if you really want it.
ALLOW_REGISTRATION = _flag("ALLOW_REGISTRATION", default=not IS_PROD)

# --- Stripe billing (DARK-LAUNCHED) + beta limits -------------------------------------------
# Billing is fully wired but DISABLED by default: with BILLING_ENABLED off the billing router is
# inert (status is safe, portal 503s, the webhook + usage reporter no-op), so the code ships without
# charging anyone. Flip BILLING_ENABLED=true only once the four Stripe values below are set (validate()
# enforces that pairing in prod so a half-configured billing fails the boot, not silently misbills).
# Two meters (see pricing.py): STRIPE_PRICE_MEMORY_ID = $/onboarded-doc/month, STRIPE_PRICE_INFERENCE_ID
# = $/1k queries. Internal usage tables stay the source of truth; Stripe is only the rating layer.
BILLING_ENABLED = _flag("BILLING_ENABLED", default=False)
# Accept STRIPE_API_KEY as an alias (what the operator's .env uses) so the secret key
# resolves under either name — same tolerance as CLOUDFLARE_API_TOKEN/_API_KEY.
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", os.environ.get("STRIPE_API_KEY", ""))
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_MEMORY_ID = os.environ.get("STRIPE_PRICE_MEMORY_ID", "")
STRIPE_PRICE_INFERENCE_ID = os.environ.get("STRIPE_PRICE_INFERENCE_ID", "")

# Beta limits — invisible until hit, generous by design (app/limits.py enforces; a platform_admin
# raises them per-tenant via max_docs_override / max_queries_override). 0 means UNLIMITED. Docs are a
# lifetime count per workspace; queries are counted per calendar month (UTC).
BETA_MAX_DOCS_PER_TENANT = int(os.environ.get("BETA_MAX_DOCS_PER_TENANT", "5000"))
BETA_MAX_QUERIES_PER_MONTH = int(os.environ.get("BETA_MAX_QUERIES_PER_MONTH", "100000"))

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

# OCR page ceiling for scanned/image-only PDFs. Parsing runs INSIDE the upload request and OCR
# is ~1-2s/page on CPU, so a big scan would tie up the request for minutes — cap it and fail
# cleanly (asking the user to split the file) rather than block. Only ever hit when a PDF's text
# layer is (near-)empty and we fall back to rendering + OCR (see parsing._extract_pdf).
OCR_MAX_PAGES = int(os.environ.get("OCR_MAX_PAGES", "40"))

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
    if BILLING_ENABLED and not all(
        [STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET, STRIPE_PRICE_MEMORY_ID, STRIPE_PRICE_INFERENCE_ID]
    ):
        # A half-configured billing (flag on, some Stripe value missing) would create customers /
        # portal sessions but silently fail to meter, or reject every webhook — misbilling. Fail the
        # boot instead so billing is all-or-nothing.
        problems.append(
            "BILLING_ENABLED requires STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET, "
            "STRIPE_PRICE_MEMORY_ID and STRIPE_PRICE_INFERENCE_ID to all be set"
        )
    if problems:
        raise RuntimeError("Invalid production config:\n  - " + "\n  - ".join(problems))
