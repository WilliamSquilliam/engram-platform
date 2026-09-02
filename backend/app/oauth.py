"""Authlib OAuth registry — Google sign-in + the source-connector OAuth clients.

Three clients, each registered ONLY when its credentials are present so this import is safe on
installs without creds (the buttons/connectors stay hidden):
  - google            : OIDC sign-in (openid email profile) — the login flow (routers/auth.py).
  - connector_google  : Drive read-only ingestion (drive.readonly, offline access for a refresh
                        token). Uses the SAME Google OAuth client as sign-in unless GDRIVE_* overrides.
  - connector_sharepoint : Microsoft Graph read-only ingestion via the multi-tenant "organizations"
                        authority, so ANY customer M365 org can consent.

Production replaces direct Google OIDC sign-in with Keycloak as the federated broker — the rest of the
app only depends on the JWT create_access_token issues, so that swap is contained to this module + the
/auth/google routes. The connector clients are independent of the sign-in backend.
"""
from . import config

try:
    from authlib.integrations.starlette_client import OAuth

    oauth = OAuth()
except ImportError:  # authlib not installed yet — password login still works.
    oauth = None


# Microsoft Graph (SharePoint) endpoints. The authority segment is "organizations" (multi-tenant work/
# school accounts) unless SHAREPOINT_TENANT_ID pins a single tenant for testing. Built as functions so a
# test that monkeypatches config.SHAREPOINT_TENANT_ID gets the right authority on the next register.
def _ms_authority() -> str:
    tenant = config.SHAREPOINT_TENANT_ID or "organizations"
    return f"https://login.microsoftonline.com/{tenant}"


def ms_authorize_url() -> str:
    return f"{_ms_authority()}/oauth2/v2.0/authorize"


def ms_token_url() -> str:
    return f"{_ms_authority()}/oauth2/v2.0/token"


# Provider registry: OAuth-client name + the scopes + the extra authorize params each needs. The
# connectors router reads this so authorize/callback/refresh are provider-agnostic (one code path,
# table-driven).
PROVIDERS = {
    "google_drive": {
        "client": "connector_google",
        # readonly is enough to browse + download; access_type=offline + prompt=consent make Google
        # return a refresh token EVERY time (without prompt=consent a re-consent omits it).
        "scope": "https://www.googleapis.com/auth/drive.readonly",
        "authorize_params": {"access_type": "offline", "prompt": "consent"},
    },
    "sharepoint": {
        "client": "connector_sharepoint",
        # offline_access -> refresh token; Files/Sites read for browse+download; User.Read for the
        # account label (/me).
        "scope": "offline_access Files.Read.All Sites.Read.All User.Read",
        "authorize_params": {},
    },
}


def _registered(name: str) -> bool:
    """Whether an OAuth client is already registered (public API; avoids touching internals)."""
    return oauth is not None and oauth.create_client(name) is not None


def register_all() -> None:
    """(Re)register every configured OAuth client. Idempotent: a client already registered is skipped,
    so a test that sets creds via monkeypatch can call this again to register a client mid-suite."""
    if oauth is None:
        return

    if config.GOOGLE_ENABLED and not _registered("google"):
        oauth.register(
            name="google",
            client_id=config.GOOGLE_CLIENT_ID,
            client_secret=config.GOOGLE_CLIENT_SECRET,
            # OIDC discovery doc — Authlib pulls endpoints, JWKS, and validates id_token.
            server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"},
        )

    # Google Drive connector: same OIDC discovery doc as sign-in (so token endpoint + refresh come for
    # free), but its own client id/secret (GDRIVE_* default to GOOGLE_*) and the Drive scope.
    if config.GDRIVE_ENABLED and not _registered("connector_google"):
        oauth.register(
            name="connector_google",
            client_id=config.GDRIVE_CLIENT_ID,
            client_secret=config.GDRIVE_CLIENT_SECRET,
            server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
            client_kwargs={"scope": PROVIDERS["google_drive"]["scope"]},
        )

    # SharePoint / Microsoft Graph: explicit authorize/token endpoints (the multi-tenant authority),
    # no OIDC discovery (Graph's common metadata doc isn't needed for this flow).
    if config.SHAREPOINT_ENABLED and not _registered("connector_sharepoint"):
        oauth.register(
            name="connector_sharepoint",
            client_id=config.SHAREPOINT_CLIENT_ID,
            client_secret=config.SHAREPOINT_CLIENT_SECRET,
            access_token_url=ms_token_url(),
            authorize_url=ms_authorize_url(),
            client_kwargs={"scope": PROVIDERS["sharepoint"]["scope"]},
        )


register_all()
