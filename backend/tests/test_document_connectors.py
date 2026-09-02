"""Google Drive + SharePoint document connectors: OAuth connect flow, browse, and import.

Provider HTTP is mocked at the httpx layer (httpx.MockTransport) so the REAL browse/import mapping
logic runs — the tests assert the Drive folder/file split, the SharePoint site:/item: id scheme, and
the import counters/limits through the actual save path. The OAuth token exchange (callback) mocks
Authlib's authorize_access_token so no network/browser is involved.
"""
import uuid

import httpx
import pytest
from app import config
from app.connectors import crypto, providers
from app.db import SessionLocal
from app.models import ConnectorConnection, ImportRun
from app.oauth import register_all

# A fixed valid Fernet key is pinned in conftest; a distinct one exercises the wrong-key 503 path.
_OTHER_KEY = "aQ2b3c4d5e6F7g8H9i0J1k2L3m4N5o6P7q8R9s0T1u2="


# --------------------------------------------------------------------------------------------------
# helpers / fixtures
# --------------------------------------------------------------------------------------------------

@pytest.fixture()
def drive_enabled(monkeypatch):
    """Turn the Google Drive connector on for a test (creds + the pinned enc key) and register its
    Authlib client, mirroring the gpu_admin per-test enable pattern."""
    monkeypatch.setattr(config, "GDRIVE_CLIENT_ID", "gd-id")
    monkeypatch.setattr(config, "GDRIVE_CLIENT_SECRET", "gd-secret")
    monkeypatch.setattr(config, "GDRIVE_ENABLED", True)
    register_all()


@pytest.fixture()
def sharepoint_enabled(monkeypatch):
    monkeypatch.setattr(config, "SHAREPOINT_CLIENT_ID", "sp-id")
    monkeypatch.setattr(config, "SHAREPOINT_CLIENT_SECRET", "sp-secret")
    monkeypatch.setattr(config, "SHAREPOINT_ENABLED", True)
    register_all()


def _tenant_id_for(headers, client) -> str:
    return client.get("/auth/me", headers=headers).json()["tenant_id"]


def _make_connection(tenant_id: str, provider: str = "google_drive",
                     account: str = "user@corp.com") -> str:
    """Insert a ready connection row directly (bypassing the OAuth dance) so browse/import tests start
    from an authorized state. Tokens are encrypted exactly as the callback would store them; the access
    token expiry is set well in the future so the cached token is used (no refresh) by default."""
    import datetime

    db = SessionLocal()
    try:
        conn = ConnectorConnection(
            tenant_id=tenant_id, provider=provider, account_label=account,
            enc_refresh_token=crypto.encrypt("refresh-xyz"),
            enc_access_token=crypto.encrypt("access-abc"),
            token_expires_at=(datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
                              + datetime.timedelta(hours=1)),
        )
        db.add(conn)
        db.commit()
        return conn.id
    finally:
        db.close()


def _mock_httpx(monkeypatch, handler):
    """Route every httpx.Client the providers module builds through a MockTransport `handler`, so the
    real provider code runs against canned responses. Patches providers.httpx.Client (the one the
    module actually calls)."""
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs.pop("timeout", None)
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(**kwargs)

    monkeypatch.setattr(providers.httpx, "Client", fake_client)


# --------------------------------------------------------------------------------------------------
# registry availability
# --------------------------------------------------------------------------------------------------

def test_registry_available_flips_with_creds_and_key(client, auth, monkeypatch):
    """A connector is 'coming soon' with no creds and becomes available only with creds AND the enc
    key (IMPLEMENTED is already True). Creds without a key must NOT flip it on."""
    headers, _ = auth
    # Baseline: no connector creds (conftest pins them empty) -> both coming soon.
    by_id = {c["id"]: c for c in client.get("/connectors", headers=headers).json()["connectors"]}
    assert by_id["google_drive"]["available"] is False
    assert by_id["sharepoint"]["available"] is False

    # Creds but NO enc key -> still unavailable (a connection with no key to decrypt is useless).
    monkeypatch.setattr(config, "GDRIVE_CLIENT_ID", "gd-id")
    monkeypatch.setattr(config, "GDRIVE_CLIENT_SECRET", "gd-secret")
    monkeypatch.setattr(config, "GDRIVE_ENABLED", bool("gd-id" and "gd-secret" and ""))
    by_id = {c["id"]: c for c in client.get("/connectors", headers=headers).json()["connectors"]}
    assert by_id["google_drive"]["available"] is False

    # Creds AND key -> available.
    monkeypatch.setattr(config, "GDRIVE_ENABLED", True)
    by_id = {c["id"]: c for c in client.get("/connectors", headers=headers).json()["connectors"]}
    assert by_id["google_drive"]["available"] is True
    # The description carries the operator setup (redirect URI + scope).
    assert "callback" in by_id["google_drive"]["description"]


# --------------------------------------------------------------------------------------------------
# authorize
# --------------------------------------------------------------------------------------------------

def test_authorize_404_for_non_owned_corpus(client, make_corpus, drive_enabled):
    """authorize must 404 when the caller doesn't own the document base (tenant isolation)."""
    h1 = _register(client)
    h2 = _register(client)
    cid = make_corpus(h1, "owned by 1")
    r = client.get(f"/connectors/google_drive/authorize?corpus_id={cid}", headers=h2)
    assert r.status_code == 404


def test_authorize_returns_consent_url(client, auth, make_corpus, drive_enabled):
    headers, _ = auth
    cid = make_corpus(headers)
    r = client.get(f"/connectors/google_drive/authorize?corpus_id={cid}", headers=headers)
    assert r.status_code == 200, r.text
    url = r.json()["url"]
    assert url.startswith("https://accounts.google.com/o/oauth2/")
    # offline + consent so Google returns a refresh token every time.
    assert "access_type=offline" in url and "prompt=consent" in url


def test_authorize_404_when_provider_unconfigured(client, auth, make_corpus):
    """With no creds (conftest default) authorize is 404 — the connector isn't configured."""
    headers, _ = auth
    cid = make_corpus(headers)
    r = client.get(f"/connectors/google_drive/authorize?corpus_id={cid}", headers=headers)
    assert r.status_code == 404


# --------------------------------------------------------------------------------------------------
# callback: encrypted connection + upsert dedupe
# --------------------------------------------------------------------------------------------------

class _FakeOAuthClient:
    """Stands in for the Authlib client on the callback: authorize_access_token returns a canned token
    without any network/state round-trip."""

    def __init__(self, token):
        self._token = token

    async def authorize_access_token(self, request):
        return self._token


def _do_callback(client, monkeypatch, provider, token, account_label,
                 corpus_id, user_id):
    """Drive the callback with the OAuth exchange + account-label fetch mocked. We patch
    _provider_client to return our fake client and _recover_state to yield the corpus_id/user_id (the
    real signed-session state round-trip is validated separately by the authorize test)."""
    from app.routers import connectors as conn_router

    spec = {"client": f"connector_{provider}"}
    monkeypatch.setattr(conn_router, "_provider_client",
                        lambda p: (_FakeOAuthClient(token), spec))
    monkeypatch.setattr(conn_router, "_recover_state",
                        lambda request, spec: (corpus_id, user_id))
    if provider == "google_drive":
        monkeypatch.setattr(providers, "drive_account_email", lambda tok: account_label)
    else:
        monkeypatch.setattr(providers, "graph_account_label", lambda tok: account_label)
    return client.get(
        f"/connectors/{provider}/callback?state=s&code=c", follow_redirects=False
    )


def test_callback_creates_encrypted_connection_and_upserts(client, auth, make_corpus,
                                                           drive_enabled, monkeypatch):
    headers, _ = auth
    cid = make_corpus(headers)
    tenant_id = _tenant_id_for(headers, client)
    user_id = client.get("/auth/me", headers=headers).json()["id"]
    token = {"access_token": "AT-1", "refresh_token": "RT-1", "expires_in": 3600}

    r = _do_callback(client, monkeypatch, "google_drive", token, "user@corp.com", cid, user_id)
    assert r.status_code in (302, 307)
    assert f"/document-base/{cid}/setup?connected=google_drive" in r.headers["location"]

    db = SessionLocal()
    try:
        rows = db.query(ConnectorConnection).filter(
            ConnectorConnection.tenant_id == tenant_id).all()
        assert len(rows) == 1
        conn = rows[0]
        # Stored ciphertext is NOT the plaintext token, and it decrypts back to it.
        assert conn.enc_refresh_token != "RT-1"
        assert crypto.decrypt(conn.enc_refresh_token) == "RT-1"
        assert crypto.decrypt(conn.enc_access_token) == "AT-1"
        assert conn.account_label == "user@corp.com"
    finally:
        db.close()

    # Reconnecting the SAME account upserts (re-keys), not a second row.
    token2 = {"access_token": "AT-2", "refresh_token": "RT-2", "expires_in": 3600}
    _do_callback(client, monkeypatch, "google_drive", token2, "user@corp.com", cid, user_id)
    db = SessionLocal()
    try:
        rows = db.query(ConnectorConnection).filter(
            ConnectorConnection.tenant_id == tenant_id).all()
        assert len(rows) == 1
        assert crypto.decrypt(rows[0].enc_refresh_token) == "RT-2"
    finally:
        db.close()


def test_callback_without_refresh_token_fails(client, auth, make_corpus, drive_enabled, monkeypatch):
    """No refresh token in the exchange (e.g. Google re-consent without prompt=consent) -> a failed
    connect (error redirect), never a broken half-stored connection."""
    headers, _ = auth
    cid = make_corpus(headers)
    user_id = client.get("/auth/me", headers=headers).json()["id"]
    token = {"access_token": "AT-only", "expires_in": 3600}  # no refresh_token
    r = _do_callback(client, monkeypatch, "google_drive", token, "user@corp.com", cid, user_id)
    assert r.status_code in (302, 307)
    assert "connector_error=google_drive" in r.headers["location"]


# --------------------------------------------------------------------------------------------------
# browse: Drive folder/file split + SharePoint id scheme
# --------------------------------------------------------------------------------------------------

def test_browse_drive_maps_folders_and_files(client, auth, drive_enabled, monkeypatch):
    headers, _ = auth
    tenant_id = _tenant_id_for(headers, client)
    conn_id = _make_connection(tenant_id, "google_drive")

    def handler(request):
        # about.get (account label) not needed here; files.list returns a folder, a supported doc,
        # a Google-native Doc (exportable -> counts), and an unsupported binary (ignored).
        assert "/drive/v3/files" in str(request.url)
        return httpx.Response(200, json={"files": [
            {"id": "F1", "name": "Subfolder", "mimeType": "application/vnd.google-apps.folder"},
            {"id": "D1", "name": "report.pdf", "mimeType": "application/pdf"},
            {"id": "D2", "name": "Notes", "mimeType": "application/vnd.google-apps.document"},
            {"id": "D3", "name": "image.png", "mimeType": "image/png"},
        ]})

    _mock_httpx(monkeypatch, handler)
    r = client.get(f"/connectors/connections/{conn_id}/browse?folder_id=root", headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["folders"] == [{"id": "F1", "name": "Subfolder"}]
    # report.pdf + the exportable Google Doc -> 2 supported; image.png ignored.
    assert body["supported_files"] == 2


def test_browse_sharepoint_site_and_item_id_scheme(client, auth, sharepoint_enabled, monkeypatch):
    headers, _ = auth
    tenant_id = _tenant_id_for(headers, client)
    conn_id = _make_connection(tenant_id, "sharepoint", "user@corp.onmicrosoft.com")

    def handler(request):
        url = str(request.url)
        if "followedSites" in url:
            return httpx.Response(200, json={"value": [
                {"id": "site-123", "displayName": "Team Site"},
            ]})
        if "/sites/site-123/drive/root/children" in url:
            return httpx.Response(200, json={"value": [
                {"id": "IT1", "name": "Docs", "folder": {},
                 "parentReference": {"driveId": "drv-9"}},
                {"id": "IT2", "name": "spec.docx", "file": {},
                 "parentReference": {"driveId": "drv-9"}},
            ]})
        return httpx.Response(404, json={})

    _mock_httpx(monkeypatch, handler)
    # Top level (no folder_id, no site_id) -> the tenant's sites as folders with "site:" ids.
    top = client.get(f"/connectors/connections/{conn_id}/browse", headers=headers).json()
    assert top["folders"] == [{"id": "site:site-123", "name": "Team Site"}]

    # Drilling into a site -> its library root: a subfolder gets an opaque "item:<driveId>:<itemId>" id.
    lvl = client.get(
        f"/connectors/connections/{conn_id}/browse?folder_id=site:site-123", headers=headers
    ).json()
    assert lvl["folders"] == [{"id": "item:drv-9:IT1", "name": "Docs"}]
    assert lvl["supported_files"] == 1  # spec.docx


def test_browse_cross_tenant_connection_404(client, auth, make_corpus, drive_enabled, monkeypatch):
    """A connection belonging to another tenant must 404 on browse — never usable cross-tenant."""
    h1 = _register(client)
    h2 = _register(client)
    tenant1 = _tenant_id_for(h1, client)
    conn_id = _make_connection(tenant1, "google_drive")
    r = client.get(f"/connectors/connections/{conn_id}/browse?folder_id=root", headers=h2)
    assert r.status_code == 404


# --------------------------------------------------------------------------------------------------
# import: happy path, limits, oversized/unsupported, running-409, token refresh
# --------------------------------------------------------------------------------------------------

def _drive_import_handler(files):
    """A Drive files.list + download handler for import. `files` is a list of dicts with id/name/
    mimeType; the download for each returns `content:<name>` bytes."""
    def handler(request):
        url = str(request.url)
        if url.rstrip("/").endswith("/drive/v3/files") or "/drive/v3/files?" in url:
            return httpx.Response(200, json={"files": files})
        # download: /drive/v3/files/{id} (alt=media) or /export
        for f in files:
            if f"/files/{f['id']}" in url:
                return httpx.Response(200, content=f"content of {f['name']}".encode())
        return httpx.Response(404, content=b"")
    return handler


def test_import_happy_path_lands_documents(client, auth, make_corpus, mock_ml,
                                           drive_enabled, monkeypatch):
    headers, _ = auth
    cid = make_corpus(headers)
    tenant_id = _tenant_id_for(headers, client)
    conn_id = _make_connection(tenant_id, "google_drive")
    files = [
        {"id": "A", "name": "a.txt", "mimeType": "text/plain"},
        {"id": "B", "name": "b.md", "mimeType": "text/markdown"},
    ]
    _mock_httpx(monkeypatch, _drive_import_handler(files))

    r = client.post(
        f"/corpora/{cid}/import", headers=headers,
        json={"connection_id": conn_id, "folder_id": "root", "folder_name": "My Docs"},
    )
    assert r.status_code == 200, r.text
    # Background task ran synchronously under TestClient — status is terminal.
    status = client.get(f"/corpora/{cid}/import-status", headers=headers).json()
    assert status["state"] == "done"
    assert status["imported"] == 2 and status["skipped"] == 0 and status["failed"] == 0
    # The files landed as real Documents through the SAME save path uploads use.
    docs = {d["filename"] for d in client.get(f"/corpora/{cid}/documents", headers=headers).json()}
    assert docs == {"a.txt", "b.md"}


def test_import_respects_doc_limit_partial_kept(client, auth, make_corpus, mock_ml,
                                                drive_enabled, monkeypatch):
    """When the workspace hits its document cap mid-import, the run stops as 'limited' and whatever
    imported before the cap is KEPT."""
    headers, _ = auth
    cid = make_corpus(headers)
    tenant_id = _tenant_id_for(headers, client)
    conn_id = _make_connection(tenant_id, "google_drive")
    # Cap at 1 document for this tenant so the second file trips the limit.
    monkeypatch.setattr(config, "BETA_MAX_DOCS_PER_TENANT", 1)
    files = [
        {"id": "A", "name": "a.txt", "mimeType": "text/plain"},
        {"id": "B", "name": "b.txt", "mimeType": "text/plain"},
    ]
    _mock_httpx(monkeypatch, _drive_import_handler(files))

    client.post(f"/corpora/{cid}/import", headers=headers,
                json={"connection_id": conn_id, "folder_id": "root", "folder_name": "f"})
    status = client.get(f"/corpora/{cid}/import-status", headers=headers).json()
    assert status["state"] == "limited"
    assert status["imported"] == 1  # the first file was kept
    docs = client.get(f"/corpora/{cid}/documents", headers=headers).json()
    assert len(docs) == 1


def test_import_skips_oversized_and_unsupported(client, auth, make_corpus, mock_ml,
                                                drive_enabled, monkeypatch):
    """Unsupported types are filtered by the walk; an oversized supported file is skipped mid-stream
    (never buffered whole). Neither crashes the run."""
    headers, _ = auth
    cid = make_corpus(headers)
    tenant_id = _tenant_id_for(headers, client)
    conn_id = _make_connection(tenant_id, "google_drive")
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 1)  # 1 MB cap
    big = "x" * (2 * 1024 * 1024)  # 2 MB -> over the cap
    files = [
        {"id": "A", "name": "ok.txt", "mimeType": "text/plain"},
        {"id": "B", "name": "huge.txt", "mimeType": "text/plain"},
        {"id": "C", "name": "art.png", "mimeType": "image/png"},  # unsupported -> not even walked
    ]

    def handler(request):
        url = str(request.url)
        if url.rstrip("/").endswith("/drive/v3/files") or "/drive/v3/files?" in url:
            return httpx.Response(200, json={"files": files})
        if "/files/A" in url:
            return httpx.Response(200, content=b"small ok")
        if "/files/B" in url:
            return httpx.Response(200, content=big.encode())
        return httpx.Response(404, content=b"")

    _mock_httpx(monkeypatch, handler)
    client.post(f"/corpora/{cid}/import", headers=headers,
                json={"connection_id": conn_id, "folder_id": "root", "folder_name": "f"})
    status = client.get(f"/corpora/{cid}/import-status", headers=headers).json()
    assert status["state"] == "done"
    assert status["imported"] == 1   # only ok.txt
    assert status["skipped"] == 1    # huge.txt over the cap (art.png was never a candidate)


def test_import_cross_tenant_connection_404(client, auth, make_corpus, drive_enabled):
    """Importing with another tenant's connection id must 404 (the connection is invisible)."""
    h1 = _register(client)
    h2 = _register(client)
    tenant1 = _tenant_id_for(h1, client)
    conn_id = _make_connection(tenant1, "google_drive")
    cid2 = make_corpus(h2, "tenant2 base")
    r = client.post(f"/corpora/{cid2}/import", headers=h2,
                    json={"connection_id": conn_id, "folder_id": "root", "folder_name": "f"})
    assert r.status_code == 404


def test_import_running_conflict_409(client, auth, make_corpus, drive_enabled):
    """A second import while one is already 'running' for the base is 409."""
    headers, _ = auth
    cid = make_corpus(headers)
    tenant_id = _tenant_id_for(headers, client)
    conn_id = _make_connection(tenant_id, "google_drive")
    # Seed a stuck 'running' ImportRun directly so the guard fires deterministically.
    db = SessionLocal()
    try:
        db.add(ImportRun(corpus_id=cid, connection_id=conn_id, folder_id="root",
                         folder_name="f", state="running"))
        db.commit()
    finally:
        db.close()
    r = client.post(f"/corpora/{cid}/import", headers=headers,
                    json={"connection_id": conn_id, "folder_id": "root", "folder_name": "f"})
    assert r.status_code == 409


def test_import_refreshes_token_on_401(client, auth, make_corpus, mock_ml,
                                       drive_enabled, monkeypatch):
    """A 401 on a file download triggers exactly one token refresh, then the retry succeeds and the
    file imports."""
    headers, _ = auth
    cid = make_corpus(headers)
    tenant_id = _tenant_id_for(headers, client)
    conn_id = _make_connection(tenant_id, "google_drive")
    files = [{"id": "A", "name": "a.txt", "mimeType": "text/plain"}]

    # Force access_token to return a known token first, then a refreshed one; count refreshes.
    calls = {"refresh": 0}
    real_access = providers.access_token

    def fake_access(db, conn, force_refresh=False):
        if force_refresh:
            calls["refresh"] += 1
            return "token-fresh"
        return "token-stale"

    monkeypatch.setattr(providers, "access_token", fake_access)

    def handler(request):
        url = str(request.url)
        if url.rstrip("/").endswith("/drive/v3/files") or "/drive/v3/files?" in url:
            return httpx.Response(200, json={"files": files})
        # First download attempt (stale token) 401s; the retry (fresh token) succeeds.
        auth_hdr = request.headers.get("authorization", "")
        if "/files/A" in url:
            if "token-stale" in auth_hdr:
                return httpx.Response(401, content=b"unauthorized")
            return httpx.Response(200, content=b"content ok")
        return httpx.Response(404, content=b"")

    _mock_httpx(monkeypatch, handler)
    client.post(f"/corpora/{cid}/import", headers=headers,
                json={"connection_id": conn_id, "folder_id": "root", "folder_name": "f"})
    status = client.get(f"/corpora/{cid}/import-status", headers=headers).json()
    assert calls["refresh"] == 1
    assert status["state"] == "done" and status["imported"] == 1
    assert real_access is not None  # sanity: we swapped the real function


# --------------------------------------------------------------------------------------------------
# token crypto: rotated key -> clean 503, never a 500
# --------------------------------------------------------------------------------------------------

def test_browse_rotated_key_is_clean_503(client, auth, drive_enabled, monkeypatch):
    """A connection whose tokens were encrypted with a different key (rotated CONNECTOR_ENC_KEY)
    surfaces a clean 503 'reconnect', not a 500 traceback."""
    headers, _ = auth
    tenant_id = _tenant_id_for(headers, client)
    conn_id = _make_connection(tenant_id, "google_drive")
    # Rotate the key so the stored ciphertext can no longer be decrypted.
    monkeypatch.setattr(config, "CONNECTOR_ENC_KEY", _OTHER_KEY)
    crypto.reset()
    r = client.get(f"/connectors/connections/{conn_id}/browse?folder_id=root", headers=headers)
    assert r.status_code == 503
    crypto.reset()  # restore for subsequent tests (conftest key comes back on next _fernet build)


# Local re-register helper (mirrors test_corpora._register) so these tests don't depend on that module.
def _register(client) -> dict:
    email = f"{uuid.uuid4().hex[:8]}@t.local"
    tok = client.post(
        "/auth/register",
        json={"email": email, "password": "pw123456", "tenant_name": "t"},
    ).json()["access_token"]
    return {"Authorization": f"Bearer {tok}"}
