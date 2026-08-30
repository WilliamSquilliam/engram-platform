"""Authlib OAuth registry. Google is registered only when credentials are present,
so this import is safe (and the button stays hidden) on installs without creds.
Production replaces direct Google OIDC with Keycloak as the federated broker — the
rest of the app only depends on the JWT `create_access_token` issues, so that swap
is contained to this module + the /auth/google routes."""
from .config import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_ENABLED

try:
    from authlib.integrations.starlette_client import OAuth

    oauth = OAuth()
except ImportError:  # authlib not installed yet — password login still works.
    oauth = None

if GOOGLE_ENABLED and oauth is not None:
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        # OIDC discovery doc — Authlib pulls endpoints, JWKS, and validates id_token.
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
