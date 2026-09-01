def test_me_requires_auth(client):
    assert client.get("/auth/me").status_code == 401


def test_register_and_me(client, auth):
    headers, email = auth
    r = client.get("/auth/me", headers=headers)
    assert r.status_code == 200
    assert r.json()["email"] == email


def test_login_roundtrip(client, auth):
    _, email = auth
    r = client.post("/auth/login", data={"username": email, "password": "pw123456"})
    assert r.status_code == 200
    assert r.json()["access_token"]


def test_bad_password(client, auth):
    _, email = auth
    r = client.post("/auth/login", data={"username": email, "password": "wrong"})
    assert r.status_code == 401


def test_duplicate_email(client, auth):
    _, email = auth
    r = client.post(
        "/auth/register",
        json={"email": email, "password": "pw123456", "tenant_name": "x"},
    )
    assert r.status_code == 409


def test_auth_config_google_disabled(client):
    assert client.get("/auth/config").json() == {"google_enabled": False}


def _fake_google(monkeypatch, email: str):
    """Stub the authlib seam so the /auth/google/callback logic runs without Google:
    _require_google reads module-level GOOGLE_ENABLED + oauth, both patched here."""
    from app.routers import auth as auth_mod

    class _FakeGoogle:
        async def authorize_access_token(self, request):
            return {"userinfo": {"email": email, "email_verified": True, "name": "G User"}}

    class _FakeOAuth:
        google = _FakeGoogle()

    monkeypatch.setattr(auth_mod, "oauth", _FakeOAuth())
    monkeypatch.setattr(auth_mod, "GOOGLE_ENABLED", True)


def test_google_callback_invite_only_blocks_unknown(client, monkeypatch):
    """Invite-only mode: an unknown Google account must NOT bypass the waitlist by
    having a workspace auto-provisioned — it bounces back with google_not_invited."""
    from app.db import SessionLocal
    from app.models import User
    from app.routers import auth as auth_mod

    _fake_google(monkeypatch, "stranger@example.test")
    monkeypatch.setattr(auth_mod, "ALLOW_REGISTRATION", False)
    r = client.get("/auth/google/callback", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert "error=google_not_invited" in r.headers["location"]
    db = SessionLocal()
    try:
        assert db.query(User).filter(User.email == "stranger@example.test").first() is None
    finally:
        db.close()


def test_google_callback_accepts_pending_invite(client, auth, monkeypatch):
    """An invited teammate signing in with Google joins the INVITING tenant with the
    invited role (invite consumed) — not a fresh workspace of their own."""
    from app.db import SessionLocal
    from app.models import Invite, User
    from app.routers import auth as auth_mod

    headers, admin_email = auth
    invited = "gteammate@example.test"
    assert client.post("/admin/invites", json={"email": invited, "role": "member"},
                       headers=headers).status_code == 200

    _fake_google(monkeypatch, invited)
    monkeypatch.setattr(auth_mod, "ALLOW_REGISTRATION", False)
    r = client.get("/auth/google/callback", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert "#token=" in r.headers["location"]

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == invited).first()
        admin = db.query(User).filter(User.email == admin_email).first()
        assert user is not None and user.tenant_id == admin.tenant_id  # joined, not new tenant
        assert user.role == "member"
        inv = db.query(Invite).filter(Invite.email == invited).first()
        assert inv.accepted_at is not None  # invite consumed
    finally:
        db.close()


def test_accept_invite_requires_name(client, auth):
    """Name is REQUIRED at accept-invite (422 without it) and gets stored + surfaced
    via /auth/me and the members list."""
    headers, _ = auth
    link = client.post("/admin/invites", json={"email": "named@example.test", "role": "member"},
                       headers=headers).json()["invite_link"]
    token = link.split("token=")[1]
    # Missing name -> validation error, nothing consumed.
    assert client.post("/auth/accept-invite",
                       json={"token": token, "password": "pw123456"}).status_code == 422
    # With a name -> accepted, and the name is stored + returned.
    r = client.post("/auth/accept-invite",
                    json={"token": token, "password": "pw123456", "name": "Named Person"})
    assert r.status_code == 200
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {r.json()['access_token']}"})
    assert me.json()["name"] == "Named Person"
    members = client.get("/admin/members", headers=headers).json()["members"]
    assert any(m["email"] == "named@example.test" and m["name"] == "Named Person" for m in members)


def test_invite_info_reveals_email_and_workspace(client, auth):
    """The accept page's confirmation header: a valid token resolves to the invited
    email + workspace; garbage tokens 400."""
    headers, _ = auth
    link = client.post("/admin/invites", json={"email": "who@example.test", "role": "member"},
                       headers=headers).json()["invite_link"]
    token = link.split("token=")[1]
    r = client.post("/auth/invite-info", json={"token": token})
    assert r.status_code == 200
    assert r.json()["email"] == "who@example.test"
    assert r.json()["workspace"]  # the tenant name
    assert client.post("/auth/invite-info", json={"token": "garbage"}).status_code == 400


def test_no_duplicate_pending_invites(client, auth):
    """One live pending invite per (tenant, email): a second invite 409s until the
    first is revoked (or expires), then reissuing works again."""
    headers, _ = auth
    body = {"email": "dupe@example.test", "role": "member"}
    assert client.post("/admin/invites", json=body, headers=headers).status_code == 200
    assert client.post("/admin/invites", json=body, headers=headers).status_code == 409
    invites = client.get("/admin/members", headers=headers).json()["invites"]
    inv_id = next(i["id"] for i in invites if i["email"] == "dupe@example.test")
    assert client.delete(f"/admin/invites/{inv_id}", headers=headers).status_code == 204
    assert client.post("/admin/invites", json=body, headers=headers).status_code == 200


def test_forgot_password_never_leaks_link_on_send_failure(client, auth, monkeypatch):
    """PUBLIC route: with a real email backend configured, a FAILED send (SES sandbox,
    outage) must NOT fall back to returning the reset link — that would let anyone mint
    another account's reset link while sends are failing. Link-in-response is dev-only
    (EMAIL_BACKEND=none)."""
    from app import config
    from app import email as email_mod

    _, email = auth
    monkeypatch.setattr(config, "EMAIL_BACKEND", "ses")
    monkeypatch.setattr(email_mod, "_send_ses",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sandbox reject")))
    r = client.post("/auth/forgot-password", json={"email": email})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "sent"          # uniform response preserved
    assert body.get("reset_link") is None    # and no link leaked


def test_remember_me_mints_long_lived_token(client, auth):
    """'Remember me on this device' must be real server-side: the remember_me form field
    extends the JWT exp to JWT_REMEMBER_EXPIRE_MIN; a plain login stays at JWT_EXPIRE_MIN."""
    import jwt as pyjwt

    from app import config

    _, email = auth
    creds = {"username": email, "password": "pw123456"}
    short = client.post("/auth/login", data=creds).json()["access_token"]
    long_ = client.post("/auth/login", data={**creds, "remember_me": "true"}).json()["access_token"]
    exp_s = pyjwt.decode(short, config.JWT_SECRET, algorithms=[config.JWT_ALG])["exp"]
    exp_l = pyjwt.decode(long_, config.JWT_SECRET, algorithms=[config.JWT_ALG])["exp"]
    # The gap between the two expiries is the remember-vs-default lifetime difference
    # (allow a minute of clock slack between the two mints).
    expected_gap = (config.JWT_REMEMBER_EXPIRE_MIN - config.JWT_EXPIRE_MIN) * 60
    assert abs((exp_l - exp_s) - expected_gap) < 60
