import uuid


def test_create_list_get_delete(client, auth, make_corpus):
    headers, _ = auth
    cid = make_corpus(headers, "My KB")

    listing = client.get("/corpora", headers=headers).json()
    assert any(c["id"] == cid for c in listing)

    got = client.get(f"/corpora/{cid}", headers=headers).json()
    assert got["name"] == "My KB"
    assert got["status"] == "new"

    assert client.delete(f"/corpora/{cid}", headers=headers).status_code == 204
    assert client.get(f"/corpora/{cid}", headers=headers).status_code == 404


def test_upload_documents(client, auth, make_corpus, upload_doc):
    headers, _ = auth
    cid = make_corpus(headers)
    upload_doc(headers, cid, "doc text")
    docs = client.get(f"/corpora/{cid}/documents", headers=headers).json()
    assert len(docs) == 1
    assert docs[0]["filename"] == "a.txt"


def _register(client) -> dict:
    email = f"{uuid.uuid4().hex[:6]}@t.local"
    tok = client.post(
        "/auth/register",
        json={"email": email, "password": "pw123456", "tenant_name": "t"},
    ).json()["access_token"]
    return {"Authorization": f"Bearer {tok}"}


def test_tenant_isolation(client, make_corpus):
    h1, h2 = _register(client), _register(client)
    cid = make_corpus(h1, "secret")
    # Tenant 2 must not see, read, or delete tenant 1's corpus.
    assert all(c["id"] != cid for c in client.get("/corpora", headers=h2).json())
    assert client.get(f"/corpora/{cid}", headers=h2).status_code == 404
    assert client.delete(f"/corpora/{cid}", headers=h2).status_code == 404
    # Owner still can.
    assert client.get(f"/corpora/{cid}", headers=h1).status_code == 200
