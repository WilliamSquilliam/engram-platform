"""Source connectors: the 'connect a source' menu + the OAuth connect flow + browse.

Two surfaces:
  - GET /connectors                     : the config-driven catalog (filesystem always available;
                                          google_drive/sharepoint 'coming soon' until creds + the
                                          token-encryption key are configured). Registry-driven so the
                                          menu never hardcodes availability.
  - the connect + browse flow           : /{provider}/authorize -> the provider consent URL (the SPA
                                          redirects itself), /{provider}/callback -> exchange the code,
                                          store an ENCRYPTED connection, redirect back to the setup page;
                                          /connections + /connections/{id}/browse drive the folder picker.

The actual IMPORT of a folder's documents lives in routers/corpora.py (it writes through the same
save path uploads use). Tokens are NEVER logged; state/CSRF rides Authlib's signed-session state
(the same pattern the Google login flow uses).

User-visible strings say "document base", never "corpus/corpora".
"""
import datetime
import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from .. import config
from ..connectors import connectors as list_connectors
from ..connectors import crypto, providers
from ..deps import get_current_user, get_db
from ..models import ConnectorConnection, User
from ..oauth import PROVIDERS, oauth, register_all
from ..schemas import BrowseResp, ConnectionResp, ConnectorAuthorizeResp
from .corpora import get_owned_corpus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/connectors", tags=["connectors"])


@router.get("")
def list_source_connectors(user: User = Depends(get_current_user)) -> dict:
    """The ingestion sources shown in the 'connect a source' menu. `available` connectors are
    selectable now; the rest render as 'coming soon' until their OAuth app credentials + the token
    encryption key are configured. No connector's secrets are exposed — only the availability flag
    derived from them."""
    return {
        "connectors": [
            {
                "id": c.id,
                "label": c.label,
                "available": c.available,
                "description": c.description,
            }
            for c in list_connectors()
        ]
    }


# --- OAuth connect flow -----------------------------------------------------------------------

def _redirect_uri(provider: str) -> str:
    """The provider redirect URI the operator registers — {API}/connectors/{provider}/callback. Fixed
    per provider (must EXACTLY match what's registered), built from config.API_BASE_URL."""
    return f"{config.API_BASE_URL}/connectors/{provider}/callback"


def _provider_client(provider: str):
    """Resolve the Authlib client for a provider, or 404 if the provider is unknown / not configured
    (creds or the enc key missing). register_all() picks up creds set after import (tests)."""
    register_all()
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise HTTPException(404, "Unknown source")
    # Availability is creds AND the enc key (config.*_ENABLED already fold in CONNECTOR_ENC_KEY).
    enabled = {"google_drive": config.GDRIVE_ENABLED, "sharepoint": config.SHAREPOINT_ENABLED}
    if not enabled.get(provider) or oauth is None:
        raise HTTPException(404, "This source is not configured")
    client = oauth.create_client(spec["client"])
    if client is None:
        raise HTTPException(404, "This source is not configured")
    return client, spec


@router.get("/{provider}/authorize", response_model=ConnectorAuthorizeResp)
async def authorize(
    provider: str,
    corpus_id: str,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Step 1 of connect: return the provider consent URL for the SPA to redirect ITSELF to (we don't
    302 an XHR). The user must own the document base being connected. corpus_id + the user id are
    carried in Authlib's signed-session state so the callback can't be tampered to target another base
    or user."""
    get_owned_corpus(db, user, corpus_id)  # 404s if not this tenant's base
    client, spec = _provider_client(provider)
    redirect_uri = _redirect_uri(provider)
    # create_authorization_url returns {"url", "state", ...}; save_authorize_data persists it (plus our
    # corpus_id/user_id) in the signed session keyed by state — tamper-proof + single-use (Authlib
    # validates + clears it on callback).
    rv = await client.create_authorization_url(redirect_uri, **spec["authorize_params"])
    await client.save_authorize_data(
        request, redirect_uri=redirect_uri, corpus_id=corpus_id, user_id=user.id, **rv
    )
    return ConnectorAuthorizeResp(url=rv["url"])


def _recover_state(request: Request, spec: dict) -> tuple[str | None, str | None]:
    """Read the corpus_id + user_id we stashed in Authlib's signed-session state during authorize,
    BEFORE authorize_access_token clears it. The framed data lives under _state_{client}_{state} in the
    session (tamper-proof: the session cookie is signed). A forged state not in the session yields
    nothing and the callback bails."""
    state = request.query_params.get("state", "")
    session_key = f"_state_{spec['client']}_{state}"
    framed = (request.session.get(session_key) or {}).get("data") or {}
    return framed.get("corpus_id"), framed.get("user_id")


def _fail_redirect(provider: str) -> RedirectResponse:
    return RedirectResponse(f"{config.FRONTEND_URL}/document-base?connector_error={provider}")


@router.get("/{provider}/callback")
async def callback(provider: str, request: Request, db: Session = Depends(get_db)):
    """Step 2 of connect: the provider redirects here with a code. Exchange it, fetch the account
    label, UPSERT the connection for (tenant, provider, account) with the tokens ENCRYPTED at rest,
    then redirect back to the base's setup page. Any failure redirects with ?connector_error=<provider>
    (never a raw traceback to the browser)."""
    try:
        client, spec = _provider_client(provider)
    except HTTPException:
        return _fail_redirect(provider)

    # Recover the state data (corpus_id + user_id) BEFORE authorize_access_token clears it.
    corpus_id, user_id = _recover_state(request, spec)

    try:
        token = await client.authorize_access_token(request)  # validates state, exchanges the code
    except Exception:  # noqa: BLE001 — any OAuth failure -> back to the setup page with an error flag
        logger.warning("connector callback token exchange failed for %s", provider)
        return _fail_redirect(provider)

    refresh_token = token.get("refresh_token")
    access_token = token.get("access_token")
    if not refresh_token or not access_token or not corpus_id or not user_id:
        # No refresh token = we can't import later (Google without prompt=consent, or a missing scope).
        logger.warning("connector callback for %s missing refresh token / state", provider)
        return _fail_redirect(provider)

    # Resolve the user (the callback has no Authorization header — identity comes from the signed state).
    user = db.get(User, user_id)
    if user is None:
        return _fail_redirect(provider)

    try:
        if provider == "google_drive":
            account_label = providers.drive_account_email(access_token)
        else:
            account_label = providers.graph_account_label(access_token)
    except Exception:  # noqa: BLE001 — label fetch failed (network/permission); still a failed connect
        logger.warning("connector callback for %s could not fetch account label", provider)
        return _fail_redirect(provider)

    _upsert_connection(db, user, provider, account_label, refresh_token, access_token, token)
    return RedirectResponse(
        f"{config.FRONTEND_URL}/document-base/{corpus_id}/setup?connected={provider}"
    )


def _upsert_connection(db: Session, user: User, provider: str, account_label: str,
                       refresh_token: str, access_token: str, token: dict) -> ConnectorConnection:
    """Create or update the (tenant, provider, account) connection with ENCRYPTED tokens. Reconnecting
    the same account re-keys the existing row instead of piling up duplicates."""
    expires_at = None
    expires_in = token.get("expires_in")
    if expires_in:
        expires_at = (datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
                      + datetime.timedelta(seconds=int(expires_in)))
    conn = (
        db.query(ConnectorConnection)
        .filter(
            ConnectorConnection.tenant_id == user.tenant_id,
            ConnectorConnection.provider == provider,
            ConnectorConnection.account_label == account_label,
        )
        .first()
    )
    if conn is None:
        conn = ConnectorConnection(
            tenant_id=user.tenant_id, provider=provider, account_label=account_label,
            created_by=user.id,
        )
        db.add(conn)
    conn.enc_refresh_token = crypto.encrypt(refresh_token)
    conn.enc_access_token = crypto.encrypt(access_token)
    conn.token_expires_at = expires_at
    db.commit()
    db.refresh(conn)
    return conn


# --- connections + browse ---------------------------------------------------------------------

def get_owned_connection(db: Session, user: User, connection_id: str) -> ConnectorConnection:
    """Fetch a connection scoped to the caller's tenant. A connection of ANOTHER tenant (or a bad id)
    is 404 — cross-tenant use must never reveal a connection exists."""
    conn = db.get(ConnectorConnection, connection_id)
    if conn is None or conn.tenant_id != user.tenant_id:
        raise HTTPException(404, "Connection not found")
    return conn


@router.get("/connections", response_model=list[ConnectionResp])
def list_connections(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """This tenant's source connections (no tokens exposed — only the account label)."""
    rows = (
        db.query(ConnectorConnection)
        .filter(ConnectorConnection.tenant_id == user.tenant_id)
        .order_by(ConnectorConnection.created_at.desc())
        .all()
    )
    return [
        ConnectionResp(id=c.id, provider=c.provider, account_label=c.account_label,
                       created_at=c.created_at)
        for c in rows
    ]


@router.get("/connections/{connection_id}/browse", response_model=BrowseResp)
def browse(
    connection_id: str,
    folder_id: str = "",
    site_id: str = "",
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """One level of the connected source's tree: subfolders to drill into + a count of importable files
    here. Ids are opaque provider refs the client passes straight back (see providers module)."""
    conn = get_owned_connection(db, user, connection_id)
    token = providers.access_token(db, conn)  # refreshes + re-encrypts if the cached token expired
    try:
        if conn.provider == "google_drive":
            result = providers.drive_browse(token, folder_id)
        else:
            result = providers.graph_browse(token, folder_id, site_id or None)
    except httpx.HTTPStatusError as exc:
        # A 401 mid-browse means the cached token was rejected — refresh once and retry.
        if exc.response.status_code == 401:
            token = providers.access_token(db, conn, force_refresh=True)
            if conn.provider == "google_drive":
                result = providers.drive_browse(token, folder_id)
            else:
                result = providers.graph_browse(token, folder_id, site_id or None)
        else:
            raise HTTPException(502, "The source could not be read right now.") from exc
    return BrowseResp(**result)
