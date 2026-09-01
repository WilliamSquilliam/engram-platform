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
