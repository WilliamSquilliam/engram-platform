"""E1: self-serve auth + roles/invites + waitlist.

Covers the shared frontend contract end to end: request-access -> approve ->
accept-invite -> login; teammate invite -> accept; forgot -> reset; the authz gates
(member 403 on /admin/*, non-platform-admin 403 on /platform-admin/*); and the
EMAIL_BACKEND=none link-in-response behavior + no-user-enumeration on forgot.

EMAIL_BACKEND defaults to "none" in the test env, so every gated flow returns its
link in the response body — that's what these tests redeem.
"""
import uuid


def _email() -> str:
    return f"user-{uuid.uuid4().hex[:8]}@test.local"


def _headers(client, email: str, password: str = "pw123456") -> dict:
    r = client.post("/auth/login", data={"username": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _make_platform_admin(email: str) -> None:
    """Promote a user to platform_admin directly (the founder is seeded via env in
    prod; tests flip the flag on an already-registered user)."""
    from app.db import SessionLocal
    from app.models import User

    db = SessionLocal()
    try:
        u = db.query(User).filter(User.email == email).first()
        u.platform_admin = True
        db.commit()
    finally:
        db.close()


# --- the founder flow: request-access -> approve -> accept-invite -> login ---

def test_request_access_approve_accept_login(client):
    # A founder account that can approve requests.
    founder = _email()
    client.post("/auth/register",
                json={"email": founder, "password": "pw123456", "tenant_name": "HQ"})
    _make_platform_admin(founder)
    padmin = _headers(client, founder)

    # Public request-access -> pending (never reveals account existence).
    applicant = _email()
    r = client.post("/auth/request-access", json={
        "email": applicant, "name": "Ana", "tenant_name": "Ana Co", "reason": "kicking tires",
    })
    assert r.status_code == 200
    assert r.json()["status"] == "pending"

    # Platform admin sees it pending.
    r = client.get("/platform-admin/access-requests", headers=padmin)
    assert r.status_code == 200
    reqs = [x for x in r.json() if x["email"] == applicant]
    assert len(reqs) == 1
    req_id = reqs[0]["id"]

    # Approve -> invite_link is returned because EMAIL_BACKEND=none.
    r = client.post(f"/platform-admin/access-requests/{req_id}/approve", headers=padmin)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "approved"
    assert body["invite_link"], "link must be returned when EMAIL_BACKEND=none"
    token = body["invite_link"].split("token=")[1]

    # Accept-invite creates + signs in the user; they land as their tenant's admin.
    r = client.post("/auth/accept-invite", json={"token": token, "password": "newpass123"})
    assert r.status_code == 200
    access = r.json()["access_token"]
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {access}"})
    assert me.status_code == 200
    assert me.json()["email"] == applicant
    assert me.json()["role"] == "admin"

    # And they can log in normally afterward.
    assert _headers(client, applicant, "newpass123")

    # The request is no longer pending.
    r = client.get("/platform-admin/access-requests", headers=padmin)
    assert applicant not in [x["email"] for x in r.json()]


# --- teammate invite -> accept -------------------------------------------

def test_teammate_invite_and_accept(client):
    owner = _email()
    client.post("/auth/register",
                json={"email": owner, "password": "pw123456", "tenant_name": "Acme"})
    admin = _headers(client, owner)

    mate = _email()
    r = client.post("/admin/invites", json={"email": mate, "role": "member"}, headers=admin)
    assert r.status_code == 200
    link = r.json()["invite_link"]
    assert link
    token = link.split("token=")[1]

    # Pending invite shows in the members listing.
    r = client.get("/admin/members", headers=admin)
    assert mate in [i["email"] for i in r.json()["invites"]]

    # Redeem it.
    r = client.post("/auth/accept-invite", json={"token": token, "password": "matepass1"})
    assert r.status_code == 200
    mate_token = r.json()["access_token"]

    # The teammate is a MEMBER of the SAME tenant.
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {mate_token}"})
    assert me.json()["role"] == "member"
    assert me.json()["tenant_id"] == client.get("/auth/me", headers=admin).json()["tenant_id"]

    # Now a real member, no longer pending.
    r = client.get("/admin/members", headers=admin)
    assert mate in [m["email"] for m in r.json()["members"]]
    assert mate not in [i["email"] for i in r.json()["invites"]]


def test_accept_invite_rejects_bad_and_reused_token(client):
    owner = _email()
    client.post("/auth/register",
                json={"email": owner, "password": "pw123456", "tenant_name": "Acme"})
    admin = _headers(client, owner)
    mate = _email()
    link = client.post("/admin/invites", json={"email": mate, "role": "member"},
                       headers=admin).json()["invite_link"]
    token = link.split("token=")[1]

    # Garbage token -> 400.
    assert client.post("/auth/accept-invite",
                       json={"token": "not-a-real-token", "password": "x" * 8}).status_code == 400

    # First accept works; a second accept of the SAME token is rejected (single-use).
    assert client.post("/auth/accept-invite",
                       json={"token": token, "password": "matepass1"}).status_code == 200
    assert client.post("/auth/accept-invite",
                       json={"token": token, "password": "matepass2"}).status_code == 400


# --- forgot -> reset ------------------------------------------------------

def test_forgot_then_reset(client):
    email = _email()
    client.post("/auth/register",
                json={"email": email, "password": "pw123456", "tenant_name": "Acme"})

    r = client.post("/auth/forgot-password", json={"email": email})
    assert r.status_code == 200
    assert r.json()["status"] == "sent"
    reset_link = r.json()["reset_link"]
    assert reset_link, "reset_link must be returned when EMAIL_BACKEND=none"
    token = reset_link.split("token=")[1]

    r = client.post("/auth/reset-password", json={"token": token, "password": "brandnew1"})
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    # Old password no longer works; new one does.
    assert client.post("/auth/login",
                       data={"username": email, "password": "pw123456"}).status_code == 401
    assert client.post("/auth/login",
                       data={"username": email, "password": "brandnew1"}).status_code == 200


def test_forgot_no_user_enumeration(client):
    # Unknown email: still 200 {status:'sent'}, and NO reset_link is produced (no
    # grant for a non-existent account) — so the response can't confirm existence.
    r = client.post("/auth/forgot-password", json={"email": _email()})
    assert r.status_code == 200
    assert r.json()["status"] == "sent"
    assert r.json()["reset_link"] is None


def test_reset_token_is_single_use(client):
    email = _email()
    client.post("/auth/register",
                json={"email": email, "password": "pw123456", "tenant_name": "Acme"})
    token = client.post("/auth/forgot-password",
                        json={"email": email}).json()["reset_link"].split("token=")[1]
    assert client.post("/auth/reset-password",
                       json={"token": token, "password": "brandnew1"}).status_code == 200
    # Replay is rejected.
    assert client.post("/auth/reset-password",
                       json={"token": token, "password": "another11"}).status_code == 400


# --- authz gates (the 403s) ----------------------------------------------

def test_member_is_forbidden_from_admin_routes(client):
    owner = _email()
    client.post("/auth/register",
                json={"email": owner, "password": "pw123456", "tenant_name": "Acme"})
    admin = _headers(client, owner)

    mate = _email()
    token = client.post("/admin/invites", json={"email": mate, "role": "member"},
                        headers=admin).json()["invite_link"].split("token=")[1]
    mate_token = client.post("/auth/accept-invite",
                             json={"token": token, "password": "matepass1"}).json()["access_token"]
    mate_hdr = {"Authorization": f"Bearer {mate_token}"}

    assert client.get("/admin/members", headers=mate_hdr).status_code == 403
    assert client.post("/admin/invites", json={"email": _email(), "role": "member"},
                       headers=mate_hdr).status_code == 403


def test_non_platform_admin_is_forbidden(client):
    owner = _email()
    client.post("/auth/register",
                json={"email": owner, "password": "pw123456", "tenant_name": "Acme"})
    admin = _headers(client, owner)  # tenant admin, but NOT platform_admin
    assert client.get("/platform-admin/access-requests", headers=admin).status_code == 403


def test_admin_routes_require_auth(client):
    assert client.get("/admin/members").status_code == 401
    assert client.get("/platform-admin/access-requests").status_code == 401


# --- member management (tenant-scoped role change + removal) --------------

def test_admin_can_change_role_and_remove_member(client):
    owner = _email()
    client.post("/auth/register",
                json={"email": owner, "password": "pw123456", "tenant_name": "Acme"})
    admin = _headers(client, owner)

    mate = _email()
    token = client.post("/admin/invites", json={"email": mate, "role": "member"},
                        headers=admin).json()["invite_link"].split("token=")[1]
    client.post("/auth/accept-invite", json={"token": token, "password": "matepass1"})

    members = client.get("/admin/members", headers=admin).json()["members"]
    mate_id = next(m["id"] for m in members if m["email"] == mate)

    # Promote to admin.
    r = client.patch(f"/admin/members/{mate_id}", json={"role": "admin"}, headers=admin)
    assert r.status_code == 200 and r.json()["role"] == "admin"

    # Remove them.
    assert client.delete(f"/admin/members/{mate_id}", headers=admin).status_code == 204
    remaining = [m["email"] for m in client.get("/admin/members", headers=admin).json()["members"]]
    assert mate not in remaining


def test_cannot_demote_or_remove_last_admin(client):
    owner = _email()
    client.post("/auth/register",
                json={"email": owner, "password": "pw123456", "tenant_name": "Acme"})
    admin = _headers(client, owner)
    my_id = client.get("/auth/me", headers=admin).json()["id"]

    # Last admin can't demote self.
    assert client.patch(f"/admin/members/{my_id}", json={"role": "member"},
                        headers=admin).status_code == 400
    # ...or remove self.
    assert client.delete(f"/admin/members/{my_id}", headers=admin).status_code == 400


def test_admin_cannot_touch_other_tenant_member(client):
    # Two separate tenants.
    a_owner, b_owner = _email(), _email()
    client.post("/auth/register",
                json={"email": a_owner, "password": "pw123456", "tenant_name": "A"})
    client.post("/auth/register",
                json={"email": b_owner, "password": "pw123456", "tenant_name": "B"})
    a_admin = _headers(client, a_owner)
    b_admin = _headers(client, b_owner)
    b_id = client.get("/auth/me", headers=b_admin).json()["id"]

    # A's admin can't see or mutate B's user (404, not 403 — don't leak existence).
    assert client.patch(f"/admin/members/{b_id}", json={"role": "member"},
                        headers=a_admin).status_code == 404
    assert client.delete(f"/admin/members/{b_id}", headers=a_admin).status_code == 404


def test_invite_revoke(client):
    owner = _email()
    client.post("/auth/register",
                json={"email": owner, "password": "pw123456", "tenant_name": "Acme"})
    admin = _headers(client, owner)
    mate = _email()
    client.post("/admin/invites", json={"email": mate, "role": "member"}, headers=admin)
    inv_id = next(i["id"] for i in client.get("/admin/members", headers=admin).json()["invites"]
                  if i["email"] == mate)
    assert client.delete(f"/admin/invites/{inv_id}", headers=admin).status_code == 204
    assert mate not in [i["email"] for i in
                        client.get("/admin/members", headers=admin).json()["invites"]]


# --- deny path + email-gating toggle -------------------------------------

def test_deny_access_request(client):
    founder = _email()
    client.post("/auth/register",
                json={"email": founder, "password": "pw123456", "tenant_name": "HQ"})
    _make_platform_admin(founder)
    padmin = _headers(client, founder)

    applicant = _email()
    client.post("/auth/request-access",
                json={"email": applicant, "name": "Zed", "tenant_name": "Zed Co"})
    req_id = next(x["id"] for x in client.get("/platform-admin/access-requests",
                                              headers=padmin).json() if x["email"] == applicant)
    r = client.post(f"/platform-admin/access-requests/{req_id}/deny", headers=padmin)
    assert r.status_code == 200 and r.json()["status"] == "denied"
    # No longer pending; can't be approved after denial.
    assert applicant not in [x["email"] for x in
                             client.get("/platform-admin/access-requests", headers=padmin).json()]
    assert client.post(f"/platform-admin/access-requests/{req_id}/approve",
                       headers=padmin).status_code == 409


# --- F1: a deactivated account can't authenticate, even with a valid JWT --------------

def _set_active(email: str, active: bool) -> None:
    from app.db import SessionLocal
    from app.models import User

    db = SessionLocal()
    try:
        u = db.query(User).filter(User.email == email).first()
        u.is_active = active
        db.commit()
    finally:
        db.close()


def test_deactivated_user_jwt_rejected(client):
    """A soft-disabled user (is_active=False) must be rejected 401 on every authed request even though
    their JWT is still cryptographically valid (the token outlives the deactivation)."""
    email = _email()
    tok = client.post("/auth/register",
                      json={"email": email, "password": "pw123456", "tenant_name": "Acme"}
                      ).json()["access_token"]
    hdr = {"Authorization": f"Bearer {tok}"}

    # Active: the token works.
    assert client.get("/auth/me", headers=hdr).status_code == 200

    # Deactivate the account; the SAME token must now be rejected.
    _set_active(email, False)
    assert client.get("/auth/me", headers=hdr).status_code == 401
    # And a gated route too — not just /auth/me.
    assert client.get("/corpora", headers=hdr).status_code == 401

    # Reactivate -> works again (the check is per-request, not a one-way burn).
    _set_active(email, True)
    assert client.get("/auth/me", headers=hdr).status_code == 200


# --- F3: an invite is bound to ITS tenant; can't hijack a foreign account --------------

def test_cross_tenant_invite_rejected_and_leaves_account_intact(client):
    """Emails are globally unique. If tenant B invites an email that already belongs to tenant A,
    accepting must 409 and must NOT move the account to B or change its role/password."""
    # Tenant A owns the account (as its admin).
    victim = _email()
    client.post("/auth/register",
                json={"email": victim, "password": "apassword1", "tenant_name": "TenantA"})
    a_tid = client.get("/auth/me", headers=_headers(client, victim, "apassword1")).json()["tenant_id"]

    # Tenant B (a separate workspace) invites the victim's email as an ADMIN of B.
    b_owner = _email()
    client.post("/auth/register",
                json={"email": b_owner, "password": "pw123456", "tenant_name": "TenantB"})
    b_admin = _headers(client, b_owner)
    link = client.post("/admin/invites", json={"email": victim, "role": "admin"},
                       headers=b_admin).json()["invite_link"]
    token = link.split("token=")[1]

    # Redeeming B's invite for A's account is rejected 409 — no takeover.
    r = client.post("/auth/accept-invite", json={"token": token, "password": "hijacked1"})
    assert r.status_code == 409

    # The victim's account is untouched: still in tenant A, old password still works, new one doesn't.
    me = client.get("/auth/me", headers=_headers(client, victim, "apassword1")).json()
    assert me["tenant_id"] == a_tid
    assert client.post("/auth/login",
                       data={"username": victim, "password": "hijacked1"}).status_code == 401


def _mint_invite(tenant_id: str, email: str, role: str = "member") -> str:
    """Insert an Invite row directly and return its raw token (the accept-invite path is what we test,
    not the admin create-invite guard). Mirrors how the app hashes the token."""
    import datetime

    from app.db import SessionLocal
    from app.models import Invite
    from app.security import generate_token, hash_token

    token = generate_token()
    db = SessionLocal()
    try:
        db.add(Invite(
            tenant_id=tenant_id, email=email, role=role, token_hash=hash_token(token),
            expires_at=datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
            + datetime.timedelta(hours=24),
        ))
        db.commit()
    finally:
        db.close()
    return token


def test_same_tenant_reinvite_still_works(client):
    """The tenant-binding must NOT block a legitimate re-invite: accepting an invite from the user's
    OWN tenant re-keys/re-activates the existing account (this is the branch F3 guards)."""
    owner = _email()
    client.post("/auth/register",
                json={"email": owner, "password": "pw123456", "tenant_name": "Acme"})
    admin = _headers(client, owner)
    tid = client.get("/auth/me", headers=admin).json()["tenant_id"]

    # First invite + accept -> a member of this tenant.
    token = client.post("/admin/invites", json={"email": (mate := _email()), "role": "member"},
                        headers=admin).json()["invite_link"].split("token=")[1]
    assert client.post("/auth/accept-invite",
                       json={"token": token, "password": "matepass1"}).status_code == 200

    # A fresh invite FROM THE SAME TENANT (minted directly) for the existing member, with a new role.
    token2 = _mint_invite(tid, mate, role="admin")
    r = client.post("/auth/accept-invite", json={"token": token2, "password": "matepass2"})
    assert r.status_code == 200
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {r.json()['access_token']}"}).json()
    assert me["role"] == "admin"          # same-tenant re-invite updated the role
    assert me["tenant_id"] == tid         # still in the same tenant
    # New password now works.
    assert client.post("/auth/login",
                       data={"username": mate, "password": "matepass2"}).status_code == 200


def test_email_backend_real_hides_link(client, monkeypatch):
    """When a real EMAIL_BACKEND is configured the link is emailed, NOT returned.
    Patch send_email where the router BOUND it (routers.auth), so the real-provider
    path (returns True) is what runs."""
    from app.routers import auth as auth_router

    sent = {}
    monkeypatch.setattr(auth_router, "send_email",
                        lambda to, subject, body, html=None: sent.update(to=to, body=body) or True)

    email = _email()
    client.post("/auth/register",
                json={"email": email, "password": "pw123456", "tenant_name": "Acme"})
    r = client.post("/auth/forgot-password", json={"email": email})
    assert r.status_code == 200
    assert r.json()["reset_link"] is None, "link must NOT leak when email is configured"
    assert sent.get("to") == email
