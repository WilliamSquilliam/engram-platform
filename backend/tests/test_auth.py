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
