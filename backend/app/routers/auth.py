"""Registration / login. Each registration creates a Tenant + its first (admin)
User. Google sign-in (OIDC) finds-or-creates a Tenant+User on first login.
Production swaps all of this for Keycloak OIDC; the rest of the app only depends
on `get_current_user`, so that swap stays contained."""
import datetime
import logging

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from ..config import (
    ALLOW_REGISTRATION,
    FRONTEND_URL,
    GOOGLE_ENABLED,
    GOOGLE_REDIRECT_URI,
    JWT_REMEMBER_EXPIRE_MIN,
    PASSWORD_RESET_EXPIRE_HOURS,
)
from ..deps import get_current_user, get_db
from ..email import email_enabled, send_email
from ..email_templates import password_reset_email
from ..models import AccessRequest, Invite, PasswordReset, Tenant, User
from ..oauth import oauth
from ..ratelimit import limiter
from ..schemas import (
    AcceptInviteReq,
    ForgotPasswordReq,
    InviteInfoReq,
    InviteInfoResp,
    LinkResp,
    RegisterReq,
    RequestAccessReq,
    ResetPasswordReq,
    TokenResp,
    UserResp,
)
from ..security import (
    OAUTH_NO_PASSWORD,
    create_access_token,
    generate_token,
    hash_password,
    hash_token,
    verify_password_or_burn,
    verify_token,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


def _now() -> datetime.datetime:
    # Naive UTC to match the DateTime columns (see models._now).
    return datetime.datetime.now(datetime.UTC).replace(tzinfo=None)


def _reset_link(token: str) -> str:
    return f"{FRONTEND_URL}/reset-password#token={token}"


@router.post("/register", response_model=TokenResp)
@limiter.limit("10/minute")
def register(request: Request, req: RegisterReq, db: Session = Depends(get_db)):
    # Invite-only beta: when open registration is off, self-serve signup is replaced
    # by /auth/request-access + /auth/accept-invite (login/google stay live).
    if not ALLOW_REGISTRATION:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Self-service registration is disabled")
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")
    tenant = Tenant(name=req.tenant_name)
    db.add(tenant)
    db.flush()
    # The creator is the first (and only) user of a brand-new tenant -> workspace admin.
    user = User(
        tenant_id=tenant.id,
        email=req.email,
        hashed_password=hash_password(req.password),
        role="admin",
        email_verified=True,  # they proved control by setting the password here
    )
    db.add(user)
    db.commit()
    return TokenResp(access_token=create_access_token(user.id, tenant.id))


@router.post("/request-access", response_model=LinkResp)
@limiter.limit("5/minute")
def request_access(request: Request, req: RequestAccessReq, db: Session = Depends(get_db)):
    """Invite-only-beta waitlist: record a pending access request a platform_admin
    approves later. Idempotent-ish: a repeat request from the same email while one is
    still pending doesn't stack duplicates. Always returns {status:'pending'} — it
    never reveals whether the email already has an account."""
    existing = (
        db.query(AccessRequest)
        .filter(AccessRequest.email == req.email, AccessRequest.status == "pending")
        .first()
    )
    if not existing:
        db.add(AccessRequest(
            email=req.email, name=req.name, tenant_name=req.tenant_name, reason=req.reason,
        ))
        db.commit()
    logger.info("Access request from %s (tenant=%r)", req.email, req.tenant_name)
    return LinkResp(status="pending")


@router.post("/accept-invite", response_model=TokenResp)
@limiter.limit("10/minute")
def accept_invite(request: Request, req: AcceptInviteReq, db: Session = Depends(get_db)):
    """Redeem a teammate/approval invite: create (or activate) the user with the given
    password + role, then return a session token so they're signed in immediately."""
    token_hash = hash_token(req.token)
    invite = db.query(Invite).filter(Invite.token_hash == token_hash).first()
    # Constant-time confirm + validity checks (expired / already accepted).
    if (
        invite is None
        or not verify_token(req.token, invite.token_hash)
        or invite.accepted_at is not None
        or invite.expires_at < _now()
    ):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid or expired invite")

    user = db.query(User).filter(User.email == invite.email).first()
    if user is None:
        user = User(
            tenant_id=invite.tenant_id,
            email=invite.email,
            name=req.name.strip(),
            hashed_password=hash_password(req.password),
            role=invite.role,
            email_verified=True,
            is_active=True,
        )
        db.add(user)
    else:
        # Emails are globally unique (User.email is unique), so an existing account already belongs to
        # exactly one tenant. Only an invite FOR THAT SAME tenant may re-activate/re-key it — a repeat
        # teammate invite or a re-invite after deactivation. An invite from a DIFFERENT tenant must
        # never overwrite this account (tenant, role, password): reject 409 and touch nothing, so
        # tenant B can't hijack tenant A's user by inviting their email.
        if user.tenant_id != invite.tenant_id:
            raise HTTPException(status.HTTP_409_CONFLICT, "email already in use")
        # Existing account of THIS tenant (e.g. re-invited / previously deactivated): activate + set pw.
        user.hashed_password = hash_password(req.password)
        user.name = req.name.strip()
        user.role = invite.role
        user.email_verified = True
        user.is_active = True
    invite.accepted_at = _now()
    db.commit()
    return TokenResp(access_token=create_access_token(user.id, user.tenant_id))


@router.post("/invite-info", response_model=InviteInfoResp)
@limiter.limit("20/minute")
def invite_info(request: Request, req: InviteInfoReq, db: Session = Depends(get_db)):
    """Who is this invite for? The accept page shows "joining {workspace} as {email}" so
    the invitee can confirm before setting a password. Reveals nothing the token holder
    couldn't learn by accepting (the token IS the proof of invitation); token rides the
    POST body so it stays out of URLs/logs."""
    token_hash = hash_token(req.token)
    invite = db.query(Invite).filter(Invite.token_hash == token_hash).first()
    if (
        invite is None
        or not verify_token(req.token, invite.token_hash)
        or invite.accepted_at is not None
        or invite.expires_at < _now()
    ):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid or expired invite")
    tenant = db.get(Tenant, invite.tenant_id)
    return InviteInfoResp(email=invite.email, workspace=tenant.name if tenant else "your team")


@router.post("/forgot-password", response_model=LinkResp)
@limiter.limit("5/minute")
def forgot_password(request: Request, req: ForgotPasswordReq, db: Session = Depends(get_db)):
    """Start a password reset. ALWAYS returns {status:'sent'} regardless of whether the
    email exists (no user enumeration). A grant is only created for a real, local-auth
    user; OAuth-only accounts (no local password) are silently skipped. When
    EMAIL_BACKEND=none the reset_link is included in the body so the flow is testable."""
    user = db.query(User).filter(User.email == req.email).first()
    reset_link: str | None = None
    if user is not None and user.hashed_password != OAUTH_NO_PASSWORD:
        token = generate_token()
        db.add(PasswordReset(
            user_id=user.id,
            token_hash=hash_token(token),
            expires_at=_now() + datetime.timedelta(hours=PASSWORD_RESET_EXPIRE_HOURS),
        ))
        db.commit()
        link = _reset_link(token)
        subject, text, html_body = password_reset_email(link)
        sent = send_email(user.email, subject, text, html_body)
        # SECURITY: this is a PUBLIC route — the link may only ride the response when no
        # real provider is configured (dev). A configured-but-FAILED send (SES sandbox,
        # provider outage) must return nothing, or anyone could mint themselves another
        # account's reset link by hitting this while sends are failing.
        if not sent and not email_enabled():
            reset_link = link
    # Uniform response whether or not the account existed.
    return LinkResp(status="sent", reset_link=reset_link)


@router.post("/reset-password", response_model=LinkResp)
@limiter.limit("10/minute")
def reset_password(request: Request, req: ResetPasswordReq, db: Session = Depends(get_db)):
    """Consume a reset grant and set the new password. Single-use: the grant is marked
    used so the link can't be replayed."""
    token_hash = hash_token(req.token)
    grant = db.query(PasswordReset).filter(PasswordReset.token_hash == token_hash).first()
    if (
        grant is None
        or not verify_token(req.token, grant.token_hash)
        or grant.used_at is not None
        or grant.expires_at < _now()
    ):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid or expired reset token")
    user = db.get(User, grant.user_id)
    if user is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid or expired reset token")
    user.hashed_password = hash_password(req.password)
    user.email_verified = True
    grant.used_at = _now()
    db.commit()
    return LinkResp(status="ok")


@router.post("/login", response_model=TokenResp)
@limiter.limit("10/minute")
def login(request: Request, form: OAuth2PasswordRequestForm = Depends(),
          remember_me: bool = Form(False), db: Session = Depends(get_db)):
    """`remember_me` (an extra form field beside the OAuth2 pair) mints the long-lived
    "remember me on this device" session; the client pairs it with persistent storage."""
    user = db.query(User).filter(User.email == form.username).first()
    # Burns the same PBKDF2 time when the email is unknown, so response timing
    # doesn't reveal which addresses have accounts.
    if not verify_password_or_burn(form.password, user.hashed_password if user else None):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    expire = JWT_REMEMBER_EXPIRE_MIN if remember_me else None
    return TokenResp(access_token=create_access_token(user.id, user.tenant_id, expire_min=expire))


@router.get("/me", response_model=UserResp)
def me(user: User = Depends(get_current_user)):
    return UserResp(id=user.id, email=user.email, name=user.name, tenant_id=user.tenant_id,
                    role=user.role, platform_admin=user.platform_admin)


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
        # A pending invite for this email? Google sign-in ACCEPTS it: join the inviting
        # tenant with the invited role (no password needed — Google verified the email).
        # Without this, an invited teammate clicking "Continue with Google" would land in
        # a brand-new empty workspace instead of the team that invited them.
        invite = (
            db.query(Invite)
            .filter(Invite.email == email, Invite.accepted_at.is_(None))
            .order_by(Invite.expires_at.desc())
            .first()
        )
        if invite is not None and invite.expires_at > _now():
            user = User(tenant_id=invite.tenant_id, email=email, role=invite.role,
                        name=info.get("name"), hashed_password=OAUTH_NO_PASSWORD,
                        email_verified=True)
            invite.accepted_at = _now()
            db.add(user)
            db.commit()
        elif ALLOW_REGISTRATION:
            # Open-registration mode only: first Google login provisions a workspace.
            name = info.get("name") or email.split("@")[0]
            tenant = Tenant(name=f"{name}'s workspace")
            db.add(tenant)
            db.flush()
            user = User(tenant_id=tenant.id, email=email, role="admin",
                        name=info.get("name"), hashed_password=OAUTH_NO_PASSWORD,
                        email_verified=True)
            db.add(user)
            db.commit()
        else:
            # Invite-only: an unknown Google account must not bypass the waitlist.
            return RedirectResponse(f"{FRONTEND_URL}/login?error=google_not_invited")

    jwt_token = create_access_token(user.id, user.tenant_id)
    return RedirectResponse(f"{FRONTEND_URL}/login#token={jwt_token}", status_code=302)
