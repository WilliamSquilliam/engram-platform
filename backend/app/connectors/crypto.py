"""Token-at-rest encryption for connector OAuth credentials.

OAuth refresh/access tokens are LONG-LIVED keys to a customer's whole document store — they must
never sit in the database as plaintext. Every token is encrypted with Fernet (AES-128-CBC + HMAC,
the `cryptography` recipe layer) keyed by config.CONNECTOR_ENC_KEY before it touches a DB column, and
decrypted only in-process when a provider call needs it.

Contract:
  encrypt(plaintext) -> ciphertext str      (raises TokenCryptoUnavailable if no key configured)
  decrypt(ciphertext) -> plaintext str       (raises TokenCryptoError on a wrong/rotated key)

Both errors carry a clean, user-safe HTTP status so the router never leaks a 500 traceback: a missing
key is a 503 "connector needs reconfiguration" and a bad-ciphertext decrypt (rotated/corrupt key) is
the same 503 — the fix in both cases is operator/reconnect, not a crash.
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import HTTPException

from .. import config

# One shared, operator-facing message for BOTH failure modes (no key / wrong key). The remedy is the
# same — reconfigure CONNECTOR_ENC_KEY or reconnect the source — so we don't distinguish them to the
# caller (and never echo which token or key was involved).
_RECONFIG_MSG = "This source needs to be reconnected — its saved credentials could not be read."


class TokenCryptoUnavailable(HTTPException):
    """No CONNECTOR_ENC_KEY configured, so tokens can neither be stored nor read. 503 (not 500): the
    feature is optional and unconfigured, not broken."""

    def __init__(self) -> None:
        super().__init__(503, _RECONFIG_MSG)


class TokenCryptoError(HTTPException):
    """Decrypt failed (wrong/rotated key or corrupt ciphertext). Surfaced as a clean 503 so a stored
    connection whose key changed asks the user to reconnect instead of throwing a 500 traceback."""

    def __init__(self) -> None:
        super().__init__(503, _RECONFIG_MSG)


@lru_cache(maxsize=1)
def _fernet():
    """Build (once) the Fernet from the configured key. lru_cache so we don't re-parse the key per
    call; tests that swap config.CONNECTOR_ENC_KEY call reset() to clear it. Raises
    TokenCryptoUnavailable when unset OR malformed — either way tokens can't be handled safely."""
    key = config.CONNECTOR_ENC_KEY
    if not key:
        raise TokenCryptoUnavailable()
    from cryptography.fernet import Fernet

    try:
        return Fernet(key.encode() if isinstance(key, str) else key)
    except (ValueError, TypeError) as exc:  # malformed key (not 32-byte urlsafe-base64)
        raise TokenCryptoUnavailable() from exc


def reset() -> None:
    """Drop the cached Fernet so a changed config.CONNECTOR_ENC_KEY takes effect (tests / key rotation
    in a long-lived process)."""
    _fernet.cache_clear()


def available() -> bool:
    """Whether token crypto is usable right now (a valid key is configured). Used by the availability
    gate so a connector never lights up without a working key."""
    try:
        _fernet()
        return True
    except HTTPException:
        return False


def encrypt(plaintext: str) -> str:
    """Encrypt a token for storage. Empty/None -> "" (an absent access token stays absent, not an
    encryption of "")."""
    if not plaintext:
        return ""
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    """Decrypt a stored token. "" -> "" (absent stays absent). A wrong key or corrupt value raises
    TokenCryptoError (clean 503), never a bare cryptography exception."""
    if not ciphertext:
        return ""
    from cryptography.fernet import InvalidToken

    try:
        return _fernet().decrypt(ciphertext.encode()).decode()
    except InvalidToken as exc:  # rotated/wrong key or tampered ciphertext
        raise TokenCryptoError() from exc
