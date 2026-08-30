"""Auth primitives. Password hashing uses stdlib PBKDF2 (reliable everywhere, no
native-build deps); JWTs use PyJWT. This is the LOCAL auth — production swaps to
Keycloak (OIDC/SAML SSO) per the C2 architecture, at which point this module is
replaced by JWT verification against Keycloak's JWKS."""
import datetime
import hashlib
import hmac
import os

import jwt

from .config import (
    AUTH_BACKEND,
    JWT_ALG,
    JWT_EXPIRE_MIN,
    JWT_SECRET,
    OIDC_AUDIENCE,
    OIDC_ISSUER,
    OIDC_JWKS_URL,
)

# OWASP 2024 recommendation for PBKDF2-HMAC-SHA256. verify_password reads the
# iteration count from the stored hash, so older 200k hashes keep verifying and
# get the new cost the next time the password is (re)set.
_ITER = 600_000

# Stored as the password hash for OAuth (Google) users, who have no local
# password. Structurally invalid for verify_password (no PBKDF2 layout), so any
# password-login attempt against an OAuth-only account fails closed.
OAUTH_NO_PASSWORD = "!oauth-no-local-password"


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITER)
    return f"pbkdf2_sha256${_ITER}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iters, salt_hex, dk_hex = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(dk.hex(), dk_hex)
    except (ValueError, AttributeError):
        return False


# A structurally valid PBKDF2 hash of an unguessable random value, used to burn
# the same hashing time when login hits an unknown email — so response timing
# doesn't reveal whether an account exists.
_DUMMY_HASH = hash_password(os.urandom(32).hex())


def verify_password_or_burn(password: str, stored: str | None) -> bool:
    """verify_password, but hash against a dummy when there is no stored hash
    (unknown user) so both branches cost the same."""
    if stored is None:
        verify_password(password, _DUMMY_HASH)
        return False
    return verify_password(password, stored)


def create_access_token(user_id: str, tenant_id: str) -> str:
    now = datetime.datetime.now(datetime.UTC)
    exp = now + datetime.timedelta(minutes=JWT_EXPIRE_MIN)
    return jwt.encode({"sub": user_id, "tid": tenant_id, "exp": exp}, JWT_SECRET, algorithm=JWT_ALG)


_jwks_client = None


def _decode_oidc(token: str) -> dict:
    """Verify an IdP-issued RS256 JWT against the realm's JWKS (Keycloak etc.)."""
    global _jwks_client
    from jwt import PyJWKClient

    if _jwks_client is None:
        _jwks_client = PyJWKClient(OIDC_JWKS_URL)
    key = _jwks_client.get_signing_key_from_jwt(token).key
    return jwt.decode(
        token,
        key,
        algorithms=["RS256"],
        audience=OIDC_AUDIENCE or None,
        issuer=OIDC_ISSUER or None,
        options={"verify_aud": bool(OIDC_AUDIENCE)},
    )


def decode_token(token: str) -> dict:
    if AUTH_BACKEND == "oidc":
        return _decode_oidc(token)
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
