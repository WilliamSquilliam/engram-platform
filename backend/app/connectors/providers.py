"""Provider API clients for the Google Drive + SharePoint (Microsoft Graph) connectors.

Plain REST over httpx against the documented Drive v3 and Graph v1.0 endpoints — no
google-api-python-client / msal (keeps deps light; the surface we use is small and stable). Every
provider call goes through a short timeout and NEVER logs a token or file bytes.

Two capabilities, per provider, plus token lifecycle:
  - access_token(db, conn) -> a currently-valid OAuth access token, refreshing (and re-encrypting the
    stored token) via the connection's refresh token when the cached one is missing/expired.
  - browse(conn, token, folder_id, site_id) -> {folders, supported_files, path_hint} — one level of the
    tree, folders the user can drill into and a count of importable files at this level.
  - walk(conn, token, folder_id) -> yields (rel_path, download_callable) for every supported file under
    the folder RECURSIVELY (paginated, capped) — the import worker's file stream.
  - download(...) streams a file into a size-capped buffer, rejecting anything over MAX_UPLOAD_MB
    mid-stream so a huge file can't exhaust memory.

The opaque-id scheme the browse/import APIs pass around (frontend treats these as blobs):
  Google Drive: a folder_id is a Drive file id, or "root" for My Drive's top.
  SharePoint:   no ids            -> the tenant's sites, each returned as a folder with id "site:<id>"
                "site:<siteId>"   -> that site's default document library root
                "item:<driveId>:<itemId>" -> a folder inside a library
"""
from __future__ import annotations

import datetime
import logging
from collections.abc import Callable, Iterator

import httpx
from fastapi import HTTPException
from sqlalchemy.orm import Session

from ..models import ConnectorConnection
from ..parsing import SUPPORTED_EXTS
from . import crypto

logger = logging.getLogger(__name__)

# Every provider HTTP call is bounded (anti-hang; security requirement <=30s). Downloads stream so this
# is per-chunk read, not whole-file.
_HTTP_TIMEOUT = httpx.Timeout(30.0)
# Recursive walk safety cap: never CONSIDER more than this many files in one import run, so a giant tree
# can't run unbounded. Counted across the whole recursive walk (folders + files inspected).
_MAX_FILES_PER_RUN = 2000
# Streamed-download chunk size.
_CHUNK = 64 * 1024

# Google Docs editors export to an Office format we can already parse. Anything not in this map that is
# a google-apps.* type (Forms, Sites, ...) has no useful text export and is skipped.
_GOOGLE_EXPORT = {
    "application/vnd.google-apps.document": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"),
}
_GOOGLE_FOLDER = "application/vnd.google-apps.folder"


def _ext(name: str) -> str:
    return ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""


def _supported(name: str) -> bool:
    return _ext(name) in SUPPORTED_EXTS


def _now() -> datetime.datetime:
    # Naive UTC to match the DateTime columns (see models._now).
    return datetime.datetime.now(datetime.UTC).replace(tzinfo=None)


class ImportSizeExceeded(Exception):
    """A file exceeded MAX_UPLOAD_MB while streaming — raised mid-download so we stop reading instead of
    buffering unbounded. The import worker turns it into a `skipped`, never a crash."""


# --------------------------------------------------------------------------------------------------
# Token lifecycle (shared): refresh via Authlib using the stored refresh token, re-encrypt the result.
# --------------------------------------------------------------------------------------------------

def _token_endpoint_and_client(conn: ConnectorConnection) -> tuple[str, str, str]:
    """(token_url, client_id, client_secret) for the connection's provider. A plain OAuth2
    refresh_token grant is a documented POST to the token endpoint — we do it directly (httpx) rather
    than through Authlib's ASYNC Starlette client, because refresh runs in a SYNC background thread.
    Raises 503 if the provider isn't configured."""
    from .. import config
    from ..oauth import ms_token_url

    if conn.provider == "google_drive" and config.GDRIVE_ENABLED:
        # Google's OAuth2 token endpoint is stable; no need to fetch the discovery doc for a refresh.
        return ("https://oauth2.googleapis.com/token",
                config.GDRIVE_CLIENT_ID, config.GDRIVE_CLIENT_SECRET)
    if conn.provider == "sharepoint" and config.SHAREPOINT_ENABLED:
        return (ms_token_url(), config.SHAREPOINT_CLIENT_ID, config.SHAREPOINT_CLIENT_SECRET)
    raise HTTPException(503, "This source is not configured.")


def _refresh_access_token(db: Session, conn: ConnectorConnection) -> str:
    """Mint a fresh access token from the connection's refresh token via a direct refresh_token grant,
    persist the new access token (encrypted) + expiry, and return the plaintext access token. Raises
    503 if the provider isn't configured, 401 if the refresh itself is rejected (revoked/expired
    grant)."""
    token_url, client_id, client_secret = _token_endpoint_and_client(conn)
    refresh_token = crypto.decrypt(conn.enc_refresh_token)  # may raise a clean 503 on a rotated key
    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as c:
            r = c.post(token_url, data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
            })
            r.raise_for_status()
            new_token = r.json()
    except Exception as exc:  # noqa: BLE001 — a rejected refresh is an auth failure, not a 500
        logger.warning("token refresh failed for connection %s (%s)", conn.id, conn.provider)
        raise HTTPException(401, "The connection to this source has expired — please reconnect.") from exc

    access = new_token.get("access_token")
    if not access:
        raise HTTPException(401, "The connection to this source has expired — please reconnect.")
    # Google may or may not return a rotated refresh token; MS with offline_access does. Keep the new
    # one when present so the stored grant stays fresh.
    if new_token.get("refresh_token"):
        conn.enc_refresh_token = crypto.encrypt(new_token["refresh_token"])
    conn.enc_access_token = crypto.encrypt(access)
    expires_in = new_token.get("expires_in")
    conn.token_expires_at = (
        _now() + datetime.timedelta(seconds=int(expires_in)) if expires_in else None
    )
    db.commit()
    return access


def access_token(db: Session, conn: ConnectorConnection, force_refresh: bool = False) -> str:
    """A currently-valid access token for this connection. Uses the cached (encrypted) access token
    when it exists and isn't near expiry; otherwise refreshes. `force_refresh` is the 401-retry path
    (the cached token was rejected mid-run)."""
    if not force_refresh and conn.enc_access_token and conn.token_expires_at:
        # 60s skew so we don't hand out a token about to expire on the next call.
        if conn.token_expires_at - _now() > datetime.timedelta(seconds=60):
            return crypto.decrypt(conn.enc_access_token)
    return _refresh_access_token(db, conn)


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# --------------------------------------------------------------------------------------------------
# Google Drive
# --------------------------------------------------------------------------------------------------

_DRIVE_API = "https://www.googleapis.com/drive/v3"


def drive_account_email(token: str) -> str:
    """The signed-in Drive account's email — the connection's account_label (about.get)."""
    with httpx.Client(timeout=_HTTP_TIMEOUT) as c:
        r = c.get(f"{_DRIVE_API}/about", params={"fields": "user"}, headers=_auth_headers(token))
        r.raise_for_status()
        return (r.json().get("user") or {}).get("emailAddress") or "Google Drive"


def _drive_list(client: httpx.Client, token: str, folder_id: str, page_token: str | None) -> dict:
    """One page of files.list for the children of `folder_id` (or 'root'). trashed=false; we ask for
    the fields browse+import need and page via nextPageToken."""
    q = f"'{folder_id or 'root'}' in parents and trashed=false"
    params = {
        "q": q,
        "fields": "nextPageToken, files(id, name, mimeType)",
        "pageSize": 200,
        # Drive files can live in Shared Drives too; include them so a corporate Drive works.
        "includeItemsFromAllDrives": "true",
        "supportsAllDrives": "true",
    }
    if page_token:
        params["pageToken"] = page_token
    r = client.get(f"{_DRIVE_API}/files", params=params, headers=_auth_headers(token))
    r.raise_for_status()
    return r.json()


def drive_browse(token: str, folder_id: str) -> dict:
    """One level of Drive: subfolders (to drill into) + a count of importable files here."""
    folders: list[dict] = []
    supported = 0
    with httpx.Client(timeout=_HTTP_TIMEOUT) as c:
        page_token = None
        while True:
            data = _drive_list(c, token, folder_id, page_token)
            for f in data.get("files", []):
                if f.get("mimeType") == _GOOGLE_FOLDER:
                    folders.append({"id": f["id"], "name": f.get("name", "")})
                elif f.get("mimeType") in _GOOGLE_EXPORT or _supported(f.get("name", "")):
                    supported += 1
            page_token = data.get("nextPageToken")
            if not page_token:
                break
    return {"folders": folders, "supported_files": supported, "path_hint": "My Drive"}


def _drive_download_one(token: str, file_id: str, mime_type: str, max_bytes: int) -> bytes:
    """Download (or export) a single Drive file into a size-capped buffer. Google-native docs are
    exported to Office; everything else uses alt=media. Raises ImportSizeExceeded past max_bytes."""
    if mime_type in _GOOGLE_EXPORT:
        export_mime, _ext_name = _GOOGLE_EXPORT[mime_type]
        url = f"{_DRIVE_API}/files/{file_id}/export"
        params = {"mimeType": export_mime}
    else:
        url = f"{_DRIVE_API}/files/{file_id}"
        params = {"alt": "media", "supportsAllDrives": "true"}
    return _stream_to_buffer(url, params, token, max_bytes)


# A download callable takes a (possibly refreshed) access token and returns the file bytes, so the
# import worker can hand it a fresh token on a mid-run 401 without re-walking the tree.
Downloader = Callable[[str], bytes]


def drive_walk(token: str, folder_id: str, max_bytes: int) -> Iterator[tuple[str, Downloader]]:
    """Recursively yield (relative_path, download_callable) for every supported file under `folder_id`.
    Folders are walked breadth-first; the total number of items CONSIDERED is capped at
    _MAX_FILES_PER_RUN so a giant tree stops cleanly. Google-native docs get an exported extension so
    the parser recognizes them (e.g. a Doc becomes '<name>.docx'). The download callable takes the
    access token as an argument so a 401-refresh can retry with a fresh one."""
    considered = 0
    # queue of (drive_folder_id, relative_prefix)
    stack: list[tuple[str, str]] = [(folder_id or "root", "")]
    with httpx.Client(timeout=_HTTP_TIMEOUT) as c:
        while stack:
            fid, prefix = stack.pop()
            page_token = None
            while True:
                data = _drive_list(c, token, fid, page_token)
                for f in data.get("files", []):
                    considered += 1
                    if considered > _MAX_FILES_PER_RUN:
                        return
                    name = f.get("name", "")
                    mime = f.get("mimeType")
                    if mime == _GOOGLE_FOLDER:
                        stack.append((f["id"], f"{prefix}{name}/"))
                        continue
                    if mime in _GOOGLE_EXPORT:
                        rel = f"{prefix}{name}{_GOOGLE_EXPORT[mime][1]}"
                    elif _supported(name):
                        rel = f"{prefix}{name}"
                    else:
                        continue  # unsupported native/binary type -> the worker never sees it
                    fid_, mime_ = f["id"], mime

                    def _dl(tok, _id=fid_, _mime=mime_):
                        return _drive_download_one(tok, _id, _mime, max_bytes)

                    yield rel, _dl
                page_token = data.get("nextPageToken")
                if not page_token:
                    break


# --------------------------------------------------------------------------------------------------
# SharePoint / Microsoft Graph
# --------------------------------------------------------------------------------------------------

_GRAPH_API = "https://graph.microsoft.com/v1.0"


def graph_account_label(token: str) -> str:
    """The signed-in account's userPrincipalName — the connection's account_label (/me)."""
    with httpx.Client(timeout=_HTTP_TIMEOUT) as c:
        r = c.get(f"{_GRAPH_API}/me", headers=_auth_headers(token))
        r.raise_for_status()
        body = r.json()
        return body.get("userPrincipalName") or body.get("displayName") or "SharePoint"


def _graph_sites(client: httpx.Client, token: str) -> list[dict]:
    """The tenant's sites to show at the top level: followed sites first, falling back to a wildcard
    site search. Capped at ~20 so the menu stays sane. Ids are prefixed 'site:' so the browse/import
    id scheme is opaque + self-describing."""
    sites: list[dict] = []
    seen: set[str] = set()
    for url in (f"{_GRAPH_API}/me/followedSites", f"{_GRAPH_API}/sites?search=*"):
        try:
            r = client.get(url, headers=_auth_headers(token))
            r.raise_for_status()
        except httpx.HTTPStatusError:
            continue  # followedSites can 404 on some tenants — fall through to the search
        for s in r.json().get("value", []):
            sid = s.get("id")
            if not sid or sid in seen:
                continue
            seen.add(sid)
            name = s.get("displayName") or s.get("name") or s.get("webUrl", "site")
            sites.append({"id": f"site:{sid}", "name": name})
            if len(sites) >= 20:
                return sites
        if sites:
            break  # followedSites returned something — don't also dump the wildcard search
    return sites


def _graph_children_url(folder_id: str, site_id: str | None) -> str:
    """Resolve an opaque id (+ optional site_id) to the Graph children endpoint.
      site:<id>                 -> that site's default drive root children
      item:<driveId>:<itemId>   -> that folder's children
      (site_id given, no folder) -> the site's default drive root children."""
    if folder_id.startswith("site:"):
        sid = folder_id[len("site:"):]
        return f"{_GRAPH_API}/sites/{sid}/drive/root/children"
    if folder_id.startswith("item:"):
        _, drive_id, item_id = folder_id.split(":", 2)
        return f"{_GRAPH_API}/drives/{drive_id}/items/{item_id}/children"
    if site_id:
        return f"{_GRAPH_API}/sites/{site_id}/drive/root/children"
    raise HTTPException(400, "Unrecognized folder reference.")


def _graph_folder_id(child: dict) -> str:
    """Opaque id for a folder child so the frontend can pass it straight back: item:<driveId>:<itemId>.
    driveId comes from parentReference (Graph always includes it on drive items)."""
    drive_id = (child.get("parentReference") or {}).get("driveId", "")
    return f"item:{drive_id}:{child['id']}"


def graph_browse(token: str, folder_id: str, site_id: str | None) -> dict:
    """One level of SharePoint. With no folder_id and no site_id we list the tenant's SITES as the
    top-level 'folders'; otherwise we list a drive folder's children (subfolders + a supported-file
    count)."""
    with httpx.Client(timeout=_HTTP_TIMEOUT) as c:
        if not folder_id and not site_id:
            return {"folders": _graph_sites(c, token), "supported_files": 0,
                    "path_hint": "SharePoint sites"}
        folders: list[dict] = []
        supported = 0
        url = _graph_children_url(folder_id, site_id)
        while url:
            r = c.get(url, headers=_auth_headers(token))
            r.raise_for_status()
            body = r.json()
            for child in body.get("value", []):
                if "folder" in child:
                    folders.append({"id": _graph_folder_id(child), "name": child.get("name", "")})
                elif "file" in child and _supported(child.get("name", "")):
                    supported += 1
            url = body.get("@odata.nextLink")
        return {"folders": folders, "supported_files": supported, "path_hint": "SharePoint library"}


def _graph_download_one(token: str, drive_id: str, item_id: str, max_bytes: int) -> bytes:
    url = f"{_GRAPH_API}/drives/{drive_id}/items/{item_id}/content"
    return _stream_to_buffer(url, None, token, max_bytes)


def graph_walk(token: str, folder_id: str, site_id: str | None,
               max_bytes: int) -> Iterator[tuple[str, Downloader]]:
    """Recursively yield (relative_path, download_callable) for every supported file under the folder.
    Resolves the starting endpoint from the opaque id, then walks nested folders via each child's own
    drive/item ids. Capped at _MAX_FILES_PER_RUN considered items. The download callable takes the
    access token as an argument so a 401-refresh can retry with a fresh one."""
    considered = 0
    start = _graph_children_url(folder_id, site_id)
    stack: list[tuple[str, str]] = [(start, "")]
    with httpx.Client(timeout=_HTTP_TIMEOUT) as c:
        while stack:
            url, prefix = stack.pop()
            while url:
                r = c.get(url, headers=_auth_headers(token))
                r.raise_for_status()
                body = r.json()
                for child in body.get("value", []):
                    considered += 1
                    if considered > _MAX_FILES_PER_RUN:
                        return
                    name = child.get("name", "")
                    ref = child.get("parentReference") or {}
                    drive_id = ref.get("driveId", "")
                    if "folder" in child:
                        sub = f"{_GRAPH_API}/drives/{drive_id}/items/{child['id']}/children"
                        stack.append((sub, f"{prefix}{name}/"))
                        continue
                    if "file" in child and _supported(name):
                        item_id = child["id"]
                        rel = f"{prefix}{name}"

                        def _dl(tok, _d=drive_id, _i=item_id):
                            return _graph_download_one(tok, _d, _i, max_bytes)

                        yield rel, _dl
                url = body.get("@odata.nextLink")


# --------------------------------------------------------------------------------------------------
# Shared size-capped streaming download
# --------------------------------------------------------------------------------------------------

def _stream_to_buffer(url: str, params: dict | None, token: str, max_bytes: int) -> bytes:
    """Stream a provider download into memory, aborting (ImportSizeExceeded) the moment the running
    total passes max_bytes — so an over-limit file is skipped without ever buffering the whole thing.
    Never logs the bytes."""
    buf = bytearray()
    with httpx.Client(timeout=_HTTP_TIMEOUT, follow_redirects=True) as c:
        with c.stream("GET", url, params=params, headers=_auth_headers(token)) as r:
            r.raise_for_status()
            for chunk in r.iter_bytes(_CHUNK):
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    raise ImportSizeExceeded()
    return bytes(buf)
