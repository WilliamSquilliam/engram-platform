"""Registration / login. Each registration creates a Tenant + its first (admin)
User. Google sign-in (OIDC) finds-or-creates a Tenant+User on first login.
Production swaps all of this for Keycloak OIDC; the rest of the app only depends
on `get_current_user`, so that swap stays contained."""
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from ..config import ALLOW_REGISTRATION, FRONTEND_URL, GOOGLE_ENABLED, GOOGLE_REDIRECT_URI
from ..deps import get_current_user, get_db
from ..models import Tenant, User
from ..oauth import oauth
from ..ratelimit import limiter
from ..schemas import RegisterReq, TokenResp, UserResp
from ..security import (
    OAUTH_NO_PASSWORD,
    create_access_token,
    hash_password,
    verify_password_or_burn,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResp)
@limiter.limit("10/minute")
def register(request: Request, req: RegisterReq, db: Session = Depends(get_db)):
    if not ALLOW_REGISTRATION:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Self-service registration is disabled")
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")
    tenant = Tenant(name=req.tenant_name)
    db.add(tenant)
    db.flush()
    user = User(tenant_id=tenant.id, email=req.email, hashed_password=hash_password(req.password))
    db.add(user)
    db.commit()
    return TokenResp(access_token=create_access_token(user.id, tenant.id))


@router.post("/login", response_model=TokenResp)
@limiter.limit("10/minute")
def login(request: Request, form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form.username).first()
    # Burns the same PBKDF2 time when the email is unknown, so response timing
    # doesn't reveal which addresses have accounts.
    if not verify_password_or_burn(form.password, user.hashed_password if user else None):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    return TokenResp(access_token=create_access_token(user.id, user.tenant_id))


@router.get("/me", response_model=UserResp)
def me(user: User = Depends(get_current_user)):
    return UserResp(id=user.id, email=user.email, tenant_id=user.tenant_id, role=user.role)


@router.get("/config")
def auth_config():
    """Lets the SPA show the 'Continue with Google' button only when configured."""
    return {"google_enabled": GOOGLE_ENABLED}


def _require_google() -> None:
    if not (GOOGLE_ENABLED and oauth is not None):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Google sign-in is not configured")


@router.get("/google/login")
async def google_login(request: Request):
    """Step 1: bounce the browser to Google's consent screen."""
    _require_google()
    return await oauth.google.authorize_redirect(request, GOOGLE_REDIRECT_URI)


@router.get("/google/callback")
async def google_callback(request: Request, db: Session = Depends(get_db)):
    """Step 2: Google redirects back here. Verify, find-or-create the tenant+user,
    then hand the SPA our own JWT via the URL fragment (kept out of server logs)."""
    _require_google()
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception:  # noqa: BLE001  (any OAuth failure -> back to login)
        return RedirectResponse(f"{FRONTEND_URL}/login?error=google_auth_failed")

    info = token.get("userinfo") or await oauth.google.userinfo(token=token)
    email = (info or {}).get("email")
    if not email or not info.get("email_verified"):
        return RedirectResponse(f"{FRONTEND_URL}/login?error=google_email_unverified")

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        # First Google login for this email -> provision a tenant + admin user.
        name = info.get("name") or email.split("@")[0]
        tenant = Tenant(name=f"{name}'s workspace")
        db.add(tenant)
        db.flush()
        user = User(tenant_id=tenant.id, email=email, hashed_password=OAUTH_NO_PASSWORD)
        db.add(user)
        db.commit()

    jwt_token = create_access_token(user.id, user.tenant_id)
    return RedirectResponse(f"{FRONTEND_URL}/login#token={jwt_token}", status_code=302)
