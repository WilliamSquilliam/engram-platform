"""Cloudflare DNS upsert for the GPU box's public IP (best-effort).

After a fresh launch the serve/onboard hostnames must point at the new instance IP. This does a
find-then-PUT-or-POST against the Cloudflare v4 API. It NEVER raises out: every failure is logged and
returns False, so a DNS hiccup can't fail the status read that drives the reconcile. Config (token +
zone id) is read from the config module so tests can set it post-import.
"""
import logging

import httpx

from . import config

logger = logging.getLogger(__name__)

_API_BASE = "https://api.cloudflare.com/client/v4"
_TIMEOUT = 10.0
# Short TTL so a re-point propagates fast; unproxied so health probes hit the box directly, not the
# Cloudflare edge (the serve/onboard planes are internal HTTP, not fronted by Cloudflare).
_TTL = 120
_PROXIED = False


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {config.CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json",
    }


def upsert_a_record(host: str, ip: str) -> bool:
    """Point the A record for `host` at `ip` (create if missing, update if drifted). Returns True on
    success, False on any error (logged). Requires CLOUDFLARE_API_TOKEN + CLOUDFLARE_ZONE_ID."""
    zone = config.CLOUDFLARE_ZONE_ID
    if not config.CLOUDFLARE_API_TOKEN or not zone:
        return False
    try:
        existing = httpx.get(
            f"{_API_BASE}/zones/{zone}/dns_records",
            params={"type": "A", "name": host},
            headers=_headers(),
            timeout=_TIMEOUT,
        )
        existing.raise_for_status()
        records = existing.json().get("result") or []

        payload = {"type": "A", "name": host, "content": ip, "ttl": _TTL, "proxied": _PROXIED}
        if records and records[0].get("content") == ip:
            # Already pointed here — no write. The reconcile runs on every status poll, so
            # skipping the no-op PUT keeps us clear of Cloudflare's write rate limits.
            return True
        if records:
            record_id = records[0]["id"]
            resp = httpx.put(
                f"{_API_BASE}/zones/{zone}/dns_records/{record_id}",
                json=payload,
                headers=_headers(),
                timeout=_TIMEOUT,
            )
        else:
            resp = httpx.post(
                f"{_API_BASE}/zones/{zone}/dns_records",
                json=payload,
                headers=_headers(),
                timeout=_TIMEOUT,
            )
        resp.raise_for_status()
        logger.info("Cloudflare A record %s -> %s upserted", host, ip)
        return True
    except Exception:  # noqa: BLE001  (reconcile is best-effort; never fail the caller)
        logger.warning("Cloudflare A record upsert failed for %s", host, exc_info=True)
        return False
