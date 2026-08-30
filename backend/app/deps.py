"""FastAPI dependencies: DB session + current-user resolution from the JWT.

Local auth: the token's `sub` is our User.id. OIDC (Keycloak): the token comes
from the IdP, so we find-or-create the tenant+user from its `email` claim — the
same provisioning the Google sign-in flow uses. Everything else depends only on
get_current_user, so the auth-backend swap stays contained here."""
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from .config import AUTH_BACKEND
from .db import SessionLocal
from .models import Tenant, User
from .security import OAUTH_NO_PASSWORD, decode_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _resolve_oidc_user(db: Session, claims: dict) -> User:
    email = claims.get("email")
    if not email:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "OIDC token missing email claim")
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        name = claims.get("name") or email.split("@")[0]
        tenant = Tenant(name=f"{name}'s workspace")
        db.add(tenant)
        db.flush()
        user = User(tenant_id=tenant.id, email=email, hashed_password=OAUTH_NO_PASSWORD)
        db.add(user)
        db.commit()
    return user


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    cred_exc = HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid authentication credentials")
    try:
        payload = decode_token(token)
    except Exception as exc:  # noqa: BLE001  (any decode failure -> 401)
        raise cred_exc from exc

    if AUTH_BACKEND == "oidc":
        return _resolve_oidc_user(db, payload)

    user_id = payload.get("sub")
    if not user_id:
        raise cred_exc
    user = db.get(User, user_id)
    if user is None:
        raise cred_exc
    return user
