"""Per-tenant cart namespacing (E6). The bug being fixed: cart ids used to be a bare filename slug
in a store shared across ALL tenants, so two tenants uploading a same-named file collided onto ONE KV
blob — a data-isolation leak. Cart ids are now '<tenant_id>__<slug>', so no two tenants can ever
address the same cart, while a tenant reusing a doc across its OWN corpora still dedups to one cart.

These assert the CONTROL-PLANE contract (the GPU ML plane is mocked via conftest.mock_ml): distinct
ids across tenants, no cross-tenant addressing, intra-tenant dedup kept, delete + GC scoping. No
torch/CUDA needed — the id derivation is pure python."""
import re
import uuid

from app import config
from app.retrieval import _NS_SEP, cart_id_for, doc_id_for


def _register(client) -> dict:
    email = f"{uuid.uuid4().hex[:8]}@t.local"
    tok = client.post(
        "/auth/register",
        json={"email": email, "password": "pw123456", "tenant_name": "t"},
    ).json()["access_token"]
    return {"Authorization": f"Bearer {tok}"}


def _corpus(client, headers, name="KB") -> str:
    return client.post("/corpora", json={"name": name}, headers=headers).json()["id"]


def _upload(client, headers, corpus_id, filename, text="alpha beta gamma"):
    r = client.post(
        f"/corpora/{corpus_id}/documents",
        files=[("files", (filename, text, "text/plain"))],
        headers=headers,
    )
    assert r.status_code == 200, r.text


# --- id derivation (pure, no HTTP) -------------------------------------------------------------

def test_namespaced_id_is_tenant_plus_slug():
    """The scheme: '<tenant_id>__<doc_id_for(filename)>'. Deterministic and reversible-by-prefix."""
    tid = uuid.uuid4().hex
    assert cart_id_for(tid, "handbook/ch1/intro.md") == f"{tid}{_NS_SEP}handbook_ch1_intro"
    assert cart_id_for(tid, "Report 2024.txt") == f"{tid}{_NS_SEP}Report_2024"


def test_same_tenant_same_slug_dedups():
    """Same tenant + same doc (across its own corpora) -> the SAME cart id, so cross-corpus reuse still
    resolves to one cart (dedup preserved). Different filenames that slug identically also collapse."""
    tid = uuid.uuid4().hex
    assert cart_id_for(tid, "report.md") == cart_id_for(tid, "report.txt")  # slug 'report' both


def test_different_tenants_same_filename_differ():
    """The isolation guarantee: two tenants, identical filename -> DIFFERENT cart ids (no collision)."""
    t1, t2 = uuid.uuid4().hex, uuid.uuid4().hex
    assert cart_id_for(t1, "same.txt") != cart_id_for(t2, "same.txt")
    # ...and neither shares the other's prefix, so one can never be mistaken for the other.
    assert cart_id_for(t1, "same.txt").startswith(t1)
    assert cart_id_for(t2, "same.txt").startswith(t2)


def test_namespaced_id_passes_cart_id_charset():
    """The namespaced id MUST be a legal cart-store key (validate_cart_id: alnum start, then
    [A-Za-z0-9._-], <=128 chars, no '/', no '..'). We mirror that regex; cart_id_for raises if a
    derived id would violate it, so this is the guardrail the store relies on."""
    charset = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    tid = uuid.uuid4().hex
    for fn in ["handbook/ch1/intro.md", "Report 2024.txt", "a b.c.md", "weird!!name##.pdf"]:
        cid = cart_id_for(tid, fn)
        assert charset.fullmatch(cid) and ".." not in cid and "/" not in cid


def test_namespaced_id_passes_real_validate_cart_id():
    """Cross-check against the ACTUAL IP validator (cartridges.cart_store.validate_cart_id) when it's
    importable — the source of truth our mirrored regex must not drift from. Skipped (not failed) if
    the IP repo isn't on the path, so the control-plane suite never hard-depends on it (or on torch)."""
    import sys
    from pathlib import Path

    import pytest

    ip_root = Path(__file__).resolve().parents[3] / "Engram-Smart-CAG"
    if not ip_root.exists():
        pytest.skip("IP repo (Engram-Smart-CAG) not present")
    sys.path.insert(0, str(ip_root))
    try:
        from cartridges.cart_store import validate_cart_id
    except Exception as e:  # noqa: BLE001 - optional cross-check; missing deps -> skip
        pytest.skip(f"could not import validate_cart_id: {e}")
    tid = uuid.uuid4().hex
    cid = cart_id_for(tid, "handbook/ch1/intro.md")
    assert validate_cart_id(cid) == cid  # the real validator accepts our namespaced id unchanged


# --- retrieve() returns namespaced ids scoped to the corpus's tenant --------------------------

def test_retrieve_returns_tenant_namespaced_ids(client, mock_ml):
    """End-to-end: retrieve(corpus.id, ...) resolves the tenant from the corpus_id it already gets
    (chat.py's call site is unchanged) and returns namespaced ids matching what onboarding stored."""
    from app import retrieval
    h1 = _register(client)
    c1 = _corpus(client, h1)
    _upload(client, h1, c1, "policy.txt", "vacation policy details here")

    ids = retrieval.retrieve(c1, "what is the policy?", 3)
    assert ids  # non-empty
    # The returned id is the tenant-namespaced one, not the bare slug.
    assert all(_NS_SEP in i for i in ids)
    assert "policy" in ids[0] and ids[0] != "policy"


def test_two_tenants_cannot_address_each_others_carts(client, mock_ml):
    """Two tenants upload identically-named docs. Each corpus's namespaced id is distinct, and a client
    echoing the OTHER tenant's id is rejected by validate_doc_ids (the pin path can't cross tenants)."""
    from app import retrieval
    h1, h2 = _register(client), _register(client)
    c1, c2 = _corpus(client, h1), _corpus(client, h2)
    _upload(client, h1, c1, "shared.txt", "tenant one private content")
    _upload(client, h2, c2, "shared.txt", "tenant two private content")

    id1 = retrieval.retrieve(c1, "content", 3)[0]
    id2 = retrieval.retrieve(c2, "content", 3)[0]
    assert id1 != id2  # same filename, different tenants -> different carts

    # Tenant 1's own id validates against c1; tenant 2's id does NOT (foreign namespace).
    retrieval.validate_doc_ids(c1, [id1])  # no raise
    try:
        retrieval.validate_doc_ids(c1, [id2])
        raise AssertionError("c1 must not be able to address t2's cart id")
    except KeyError:
        pass
    # ...and context_for over a foreign id raises too (can't read the other tenant's text).
    try:
        retrieval.context_for(c1, [id2])
        raise AssertionError("c1 must not resolve text for t2's cart id")
    except KeyError:
        pass


# --- delete scoping ---------------------------------------------------------------------------

def test_delete_offboards_only_this_tenants_namespaced_ids(client, mock_ml):
    """Deleting a corpus offboards its OWN namespaced carts. A different tenant's same-named cart is a
    distinct id and is never touched (the cross-tenant 'shared slug' retention is gone)."""
    h1, h2 = _register(client), _register(client)
    c1, c2 = _corpus(client, h1), _corpus(client, h2)
    _upload(client, h1, c1, "doc.txt")
    _upload(client, h2, c2, "doc.txt")  # same filename, other tenant -> different cart id
    want = cart_id_for(_tenant_of(c1), "doc.txt")
    c2_id = cart_id_for(_tenant_of(c2), "doc.txt")  # capture before delete (c2 survives anyway)
    assert c2_id != want

    assert client.delete(f"/corpora/{c1}", headers=h1).status_code == 204
    assert mock_ml.offboard_ids == [[want]]
    assert mock_ml.invalidate_ids == [[want]]
    # c2's cart (a different namespaced id) was never in the offboard set.
    assert [c2_id] not in mock_ml.offboard_ids


# --- GC scoping -------------------------------------------------------------------------------

def test_gc_is_tenant_safe(client, mock_ml, monkeypatch):
    """The store-wide GC references every document by its TENANT-NAMESPACED id, and — because
    prod and the UAT stage share one cart store — only ids whose tenant prefix belongs to a
    tenant THIS DB knows are sweepable. Foreign-environment carts and legacy un-namespaced ids
    are skipped (never delete what we can't attribute), and one tenant's live cart can never be
    another tenant's orphan."""
    monkeypatch.setattr(config, "INTERNAL_API_TOKEN", "s3cr3t-token")
    tok = {"X-Internal-Token": "s3cr3t-token"}
    h1, h2 = _register(client), _register(client)
    c1, c2 = _corpus(client, h1), _corpus(client, h2)
    _upload(client, h1, c1, "live.txt")
    _upload(client, h2, c2, "live.txt")  # same filename, different tenant
    live1 = cart_id_for(_tenant_of(c1), "live.txt")
    live2 = cart_id_for(_tenant_of(c2), "live.txt")
    assert live1 != live2

    # Store holds: both tenants' live carts, a TRUE local orphan (known tenant, no doc),
    # a FOREIGN cart (unknown tenant prefix — e.g. onboarded by the UAT stage's control
    # plane), and a legacy un-namespaced id.
    local_orphan = cart_id_for(_tenant_of(c1), "deleted_doc.txt")
    foreign_cart = ("f" * 32) + "__live"
    mock_ml.carts = [live1, live2, local_orphan, foreign_cart, "orphan_x"]
    r = client.post("/internal/gc/carts", json={"confirm": False}, headers=tok)
    assert r.status_code == 200, r.text
    # Only the locally-attributable unreferenced id is an orphan; live carts stay, the
    # foreign cart and the unattributable legacy id are skipped, counted separately.
    assert r.json()["orphans"] == [local_orphan]
    assert r.json()["n_foreign_skipped"] == 2


def _tenant_of(corpus_id) -> str:
    """Resolve a corpus's tenant id the same way the app does (fresh session), for deriving the
    expected namespaced id in-test without threading the uuid through the HTTP layer."""
    from app.retrieval import _tenant_for_corpus
    return _tenant_for_corpus(corpus_id)


def test_doc_id_for_still_bare_slug():
    """doc_id_for stays the intra-tenant dedup key (bare slug) — cart_id_for is what namespaces it.
    Guards against a refactor accidentally making doc_id_for itself tenant-aware."""
    assert doc_id_for("handbook/ch1/intro.md") == "handbook_ch1_intro"
    assert _NS_SEP not in doc_id_for("handbook/ch1/intro.md")
