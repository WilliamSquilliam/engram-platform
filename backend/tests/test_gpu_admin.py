"""Platform-admin GPU controls for the Lambda Cloud serving box (E12).

Covers the frontend contract exactly (the UI is built against it):
  GET  /platform-admin/gpu/status  -> derived lifecycle state + health probes (200 even when disabled)
  POST /platform-admin/gpu/start   -> launch (202 booting) / 409 already up / 503 disabled|no-capacity
  POST /platform-admin/gpu/stop    -> terminate (202 terminating) / 409 nothing running / 503 disabled

Lambda Cloud + Cloudflare are never touched: the lambda_cloud / cloudflare_dns module functions and
the config attrs are monkeypatched. A tenant admin (not platform_admin) must 403 on every route.
"""
import uuid

import pytest
from app import cloudflare_dns, config, lambda_cloud
from app.db import SessionLocal
from app.models import User
from app.routers import gpu_admin

# --- auth fixtures: a platform-admin user vs a plain tenant admin ----------

def _email() -> str:
    return f"user-{uuid.uuid4().hex[:8]}@test.local"


def _register(client, tenant_name: str = "Acme") -> tuple[dict, str]:
    email = _email()
    r = client.post("/auth/register",
                    json={"email": email, "password": "pw123456", "tenant_name": tenant_name})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}, email


def _make_platform_admin(email: str) -> None:
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.email == email).first()
        u.platform_admin = True
        db.commit()
    finally:
        db.close()


@pytest.fixture()
def admin_hdr(client):
    """A platform-admin (founder) auth header."""
    hdr, email = _register(client, "HQ")
    _make_platform_admin(email)
    return hdr


@pytest.fixture()
def tenant_hdr(client):
    """A plain tenant admin — NOT a platform admin (must 403 on gpu routes)."""
    hdr, _ = _register(client, "Tenant")
    return hdr


@pytest.fixture()
def gpu_enabled(monkeypatch):
    """Flip the master switch on for a test (config reads GPU_CONTROL_ENABLED fresh per call)."""
    monkeypatch.setattr(config, "GPU_CONTROL_ENABLED", True)
    monkeypatch.setattr(config, "LAMBDA_API_KEY", "test-key")
    monkeypatch.setattr(config, "LAMBDA_INSTANCE_NAME", "engram-serving")
    monkeypatch.setattr(config, "LAMBDA_FS_NAME", "engram-fs")


def _instance(status="active", ip="203.0.113.10", name="engram-serving",
              type_name="gpu_8x_b200", price_cents=32000, region="us-east-1"):
    """A Lambda instance object shaped like the real API (embeds instance_type + region)."""
    return {
        "id": "i-abc123",
        "name": name,
        "status": status,
        "ip": ip,
        "instance_type": {"name": type_name, "price_cents_per_hour": price_cents},
        "region": {"name": region},
    }


# --- authz: tenant admin is forbidden on every route ----------------------

def test_tenant_admin_forbidden(client, tenant_hdr, gpu_enabled):
    assert client.get("/platform-admin/gpu/status", headers=tenant_hdr).status_code == 403
    assert client.post("/platform-admin/gpu/start", headers=tenant_hdr).status_code == 403
    assert client.post("/platform-admin/gpu/stop", headers=tenant_hdr).status_code == 403


def test_routes_require_auth(client):
    assert client.get("/platform-admin/gpu/status").status_code == 401
    assert client.post("/platform-admin/gpu/start").status_code == 401
    assert client.post("/platform-admin/gpu/stop").status_code == 401


# --- status ---------------------------------------------------------------

def test_status_disabled_is_offline_200(client, admin_hdr, monkeypatch):
    """LAMBDA_API_KEY unset -> enabled false, state offline, instance null, still 200."""
    monkeypatch.setattr(config, "GPU_CONTROL_ENABLED", False)
    r = client.get("/platform-admin/gpu/status", headers=admin_hdr)
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is False
    assert body["state"] == "offline"
    assert body["instance"] is None


def test_status_offline_when_no_instance(client, admin_hdr, gpu_enabled, monkeypatch):
    monkeypatch.setattr(lambda_cloud, "list_instances", lambda: [])
    r = client.get("/platform-admin/gpu/status", headers=admin_hdr)
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["state"] == "offline"
    assert body["instance"] is None


def test_status_ignores_other_named_and_terminated(client, admin_hdr, gpu_enabled, monkeypatch):
    """Instances with a different name, or our name but terminated, don't count as the box."""
    monkeypatch.setattr(lambda_cloud, "list_instances", lambda: [
        _instance(name="someone-elses-box"),
        _instance(status="terminated"),
    ])
    r = client.get("/platform-admin/gpu/status", headers=admin_hdr)
    assert r.json()["state"] == "offline"


def test_status_serving_when_engine_ready(client, admin_hdr, gpu_enabled, monkeypatch):
    """active + IP + serve health ok + engine_ready true -> serving, with price/hourly derived."""
    monkeypatch.setattr(lambda_cloud, "list_instances", lambda: [_instance()])
    # Both planes reachable; serve reports engine_ready True.
    monkeypatch.setattr(gpu_admin, "_probe", lambda url: (True, True))
    monkeypatch.setattr(gpu_admin, "_reconcile_dns", lambda ip: None)

    r = client.get("/platform-admin/gpu/status", headers=admin_hdr)
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "serving"
    assert body["serve_reachable"] is True
    assert body["engine_ready"] is True
    assert body["onboard_reachable"] is True
    assert body["instance"]["ip"] == "203.0.113.10"
    assert body["instance"]["price_cents_per_hour"] == 32000
    assert body["hourly_usd"] == pytest.approx(320.0)


def test_status_warming_when_engine_not_ready(client, admin_hdr, gpu_enabled, monkeypatch):
    monkeypatch.setattr(lambda_cloud, "list_instances", lambda: [_instance()])
    # serve reachable but engine_ready false -> warming.
    monkeypatch.setattr(gpu_admin, "_probe", lambda url: (True, False))
    monkeypatch.setattr(gpu_admin, "_reconcile_dns", lambda ip: None)
    r = client.get("/platform-admin/gpu/status", headers=admin_hdr)
    assert r.json()["state"] == "warming"


def test_status_booting_when_no_ip(client, admin_hdr, gpu_enabled, monkeypatch):
    monkeypatch.setattr(lambda_cloud, "list_instances", lambda: [_instance(status="booting", ip=None)])
    r = client.get("/platform-admin/gpu/status", headers=admin_hdr)
    assert r.json()["state"] == "booting"


def test_status_terminating(client, admin_hdr, gpu_enabled, monkeypatch):
    monkeypatch.setattr(lambda_cloud, "list_instances", lambda: [_instance(status="terminating")])
    r = client.get("/platform-admin/gpu/status", headers=admin_hdr)
    assert r.json()["state"] == "terminating"


def test_status_health_probe_failure_is_provisioning_not_500(
    client, admin_hdr, gpu_enabled, monkeypatch
):
    """A dead serve plane -> provisioning (probe swallowed), never a 500."""
    monkeypatch.setattr(lambda_cloud, "list_instances", lambda: [_instance()])

    def _boom(*a, **k):
        raise RuntimeError("connection refused")

    # Real _probe wraps httpx; force httpx.get to raise to prove the probe swallows it.
    monkeypatch.setattr(gpu_admin.httpx, "get", _boom)
    monkeypatch.setattr(gpu_admin, "_reconcile_dns", lambda ip: None)

    r = client.get("/platform-admin/gpu/status", headers=admin_hdr)
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "provisioning"
    assert body["serve_reachable"] is False
    assert body["engine_ready"] is False


def test_status_dns_reconcile_upserts_on_drift(client, admin_hdr, gpu_enabled, monkeypatch):
    """With Cloudflare creds + DNS hostnames, a drifted IP triggers an upsert of both hosts and sets
    dns_pointed true."""
    monkeypatch.setattr(lambda_cloud, "list_instances", lambda: [_instance()])
    monkeypatch.setattr(gpu_admin, "_probe", lambda url: (True, True))
    # DNS hostnames (not localhost) + creds present.
    monkeypatch.setattr(config, "INFERENCE_SERVICE_URL", "https://serve.engram.example")
    monkeypatch.setattr(config, "ML_SERVICE_URL", "https://onboard.engram.example")
    monkeypatch.setattr(config, "CLOUDFLARE_API_TOKEN", "cf-token")
    monkeypatch.setattr(config, "CLOUDFLARE_ZONE_ID", "zone-1")

    upserts: list[tuple[str, str]] = []

    def fake_upsert(host, ip):
        upserts.append((host, ip))
        return True

    monkeypatch.setattr(cloudflare_dns, "upsert_a_record", fake_upsert)

    r = client.get("/platform-admin/gpu/status", headers=admin_hdr)
    assert r.status_code == 200
    assert r.json()["dns_pointed"] is True
    hosts = {h for h, _ in upserts}
    assert hosts == {"serve.engram.example", "onboard.engram.example"}
    assert all(ip == "203.0.113.10" for _, ip in upserts)


def test_status_dns_null_without_creds(client, admin_hdr, gpu_enabled, monkeypatch):
    """No Cloudflare creds -> reconcile skipped, dns_pointed null."""
    monkeypatch.setattr(lambda_cloud, "list_instances", lambda: [_instance()])
    monkeypatch.setattr(gpu_admin, "_probe", lambda url: (True, True))
    monkeypatch.setattr(config, "CLOUDFLARE_API_TOKEN", "")
    monkeypatch.setattr(config, "CLOUDFLARE_ZONE_ID", "")
    r = client.get("/platform-admin/gpu/status", headers=admin_hdr)
    assert r.json()["dns_pointed"] is None


def test_status_dns_null_for_localhost_hosts(client, admin_hdr, gpu_enabled, monkeypatch):
    """Creds present but the hosts are localhost/IP literals -> reconcile skipped, dns_pointed null."""
    monkeypatch.setattr(lambda_cloud, "list_instances", lambda: [_instance()])
    monkeypatch.setattr(gpu_admin, "_probe", lambda url: (True, True))
    monkeypatch.setattr(config, "INFERENCE_SERVICE_URL", "http://localhost:8002")
    monkeypatch.setattr(config, "ML_SERVICE_URL", "http://127.0.0.1:8001")
    monkeypatch.setattr(config, "CLOUDFLARE_API_TOKEN", "cf-token")
    monkeypatch.setattr(config, "CLOUDFLARE_ZONE_ID", "zone-1")
    r = client.get("/platform-admin/gpu/status", headers=admin_hdr)
    assert r.json()["dns_pointed"] is None


# --- start ----------------------------------------------------------------

def _catalog(type_name="gpu_8x_b200", regions=("us-east-1",)):
    return {
        type_name: {
            "instance_type": {"name": type_name, "price_cents_per_hour": 32000},
            "regions_with_capacity_available": list(regions),
        },
        # A distractor type without our filter substring / capacity.
        "cpu_4x": {"instance_type": {"name": "cpu_4x"}, "regions_with_capacity_available": []},
    }


def _filesystems(name="engram-fs", region="us-east-1"):
    return [{"id": "fs-1", "name": name, "region": {"name": region}}]


def test_start_disabled_503(client, admin_hdr, monkeypatch):
    monkeypatch.setattr(config, "GPU_CONTROL_ENABLED", False)
    assert client.post("/platform-admin/gpu/start", headers=admin_hdr).status_code == 503


def test_start_happy_path(client, admin_hdr, gpu_enabled, monkeypatch):
    """No box running -> firewall ensured, type+region discovered (capacity ∩ FS region), launched.
    user_data must reference the self-provision bundle; 202 booting with the new instance id."""
    monkeypatch.setattr(config, "LAMBDA_TYPE_FILTER", "b200")
    monkeypatch.setattr(lambda_cloud, "list_instances", lambda: [])  # nothing running
    monkeypatch.setattr(lambda_cloud, "list_instance_types", lambda: _catalog())
    monkeypatch.setattr(lambda_cloud, "list_file_systems", lambda: _filesystems())

    firewall_calls: list = []
    launch_calls: list = []
    monkeypatch.setattr(lambda_cloud, "put_firewall_rules",
                        lambda rules: firewall_calls.append(rules) or {"data": rules})

    def fake_launch(region, itype, ssh, fs, name, user_data):
        launch_calls.append(dict(region=region, itype=itype, ssh=ssh, fs=fs, name=name,
                                 user_data=user_data))
        return {"instance_ids": ["i-new-999"]}

    monkeypatch.setattr(lambda_cloud, "launch_instance", fake_launch)

    r = client.post("/platform-admin/gpu/start", headers=admin_hdr)
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["instance_id"] == "i-new-999"
    assert body["state"] == "booting"

    # Firewall was ensured before launch.
    assert len(firewall_calls) == 1
    # Launch used the discovered type/region, our FS + ssh key + name.
    call = launch_calls[0]
    assert call["region"] == "us-east-1"
    assert call["itype"] == "gpu_8x_b200"
    assert call["fs"] == ["engram-fs"]
    assert call["name"] == "engram-serving"
    # cloud-init runs the self-provision bundle from the (config) FS path.
    assert "self-provision.sh" in call["user_data"]
    assert "/lambda/nfs/engram-fs/engram/self-provision.sh" in call["user_data"]


def test_start_conflict_when_running(client, admin_hdr, gpu_enabled, monkeypatch):
    monkeypatch.setattr(lambda_cloud, "list_instances", lambda: [_instance()])
    r = client.post("/platform-admin/gpu/start", headers=admin_hdr)
    assert r.status_code == 409


def test_start_fallback_substring_and_region_objects(client, admin_hdr, gpu_enabled, monkeypatch):
    """The fallback is a SUBSTRING filter (real type name gpu_2x_h100_sxm5, filter '2x_h100_sxm')
    and capacity regions can arrive as {"name": ...} objects — both must resolve. This is the
    no-B200-capacity path the fleet actually depends on."""
    monkeypatch.setattr(config, "LAMBDA_TYPE_FILTER", "b200")
    monkeypatch.setattr(config, "LAMBDA_TYPE_FALLBACK", "2x_h100_sxm")
    monkeypatch.setattr(lambda_cloud, "list_instances", lambda: [])
    # b200 exists in the catalog but has NO capacity anywhere; the h100 has capacity, expressed
    # as region OBJECTS, in the FS region.
    catalog = {
        "gpu_8x_b200": {"instance_type": {"name": "gpu_8x_b200"},
                        "regions_with_capacity_available": []},
        "gpu_2x_h100_sxm5": {"instance_type": {"name": "gpu_2x_h100_sxm5",
                                               "price_cents_per_hour": 838},
                             "regions_with_capacity_available": [{"name": "us-southeast-1"}]},
    }
    monkeypatch.setattr(lambda_cloud, "list_instance_types", lambda: catalog)
    monkeypatch.setattr(lambda_cloud, "list_file_systems",
                        lambda: _filesystems(region="us-southeast-1"))
    monkeypatch.setattr(lambda_cloud, "put_firewall_rules", lambda rules: {"data": rules})

    launch_calls: list = []

    def fake_launch(region, itype, ssh, fs, name, user_data):
        launch_calls.append((region, itype))
        return {"instance_ids": ["i-h100"]}

    monkeypatch.setattr(lambda_cloud, "launch_instance", fake_launch)

    r = client.post("/platform-admin/gpu/start", headers=admin_hdr)
    assert r.status_code == 202, r.text
    assert launch_calls == [("us-southeast-1", "gpu_2x_h100_sxm5")]


def test_start_503_no_capacity_or_fs_intersection(client, admin_hdr, gpu_enabled, monkeypatch):
    """Capacity is in eu-central-1 but the FS lives in us-east-1 -> no intersection -> 503 naming the
    filters + FS. Firewall may still be ensured; the point is no launch happens."""
    monkeypatch.setattr(config, "LAMBDA_TYPE_FILTER", "b200")
    monkeypatch.setattr(config, "LAMBDA_TYPE_FALLBACK", "2x_h100_sxm")
    monkeypatch.setattr(lambda_cloud, "list_instances", lambda: [])
    monkeypatch.setattr(lambda_cloud, "list_instance_types",
                        lambda: _catalog(regions=("eu-central-1",)))  # capacity elsewhere
    monkeypatch.setattr(lambda_cloud, "list_file_systems",
                        lambda: _filesystems(region="us-east-1"))  # FS here
    monkeypatch.setattr(lambda_cloud, "put_firewall_rules", lambda rules: {"data": rules})

    launched = []
    monkeypatch.setattr(lambda_cloud, "launch_instance",
                        lambda *a, **k: launched.append(a) or {"instance_ids": ["x"]})

    r = client.post("/platform-admin/gpu/start", headers=admin_hdr)
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert "b200" in detail and "engram-fs" in detail
    assert launched == []  # never launched


# --- stop -----------------------------------------------------------------

def test_stop_disabled_503(client, admin_hdr, monkeypatch):
    monkeypatch.setattr(config, "GPU_CONTROL_ENABLED", False)
    assert client.post("/platform-admin/gpu/stop", headers=admin_hdr).status_code == 503


def test_stop_happy_path(client, admin_hdr, gpu_enabled, monkeypatch):
    monkeypatch.setattr(lambda_cloud, "list_instances", lambda: [_instance()])
    terminated: list = []
    monkeypatch.setattr(lambda_cloud, "terminate_instances",
                        lambda ids: terminated.append(ids) or {})
    r = client.post("/platform-admin/gpu/stop", headers=admin_hdr)
    assert r.status_code == 202, r.text
    assert r.json()["state"] == "terminating"
    assert terminated == [["i-abc123"]]


def test_stop_conflict_when_nothing_running(client, admin_hdr, gpu_enabled, monkeypatch):
    monkeypatch.setattr(lambda_cloud, "list_instances", lambda: [])
    r = client.post("/platform-admin/gpu/stop", headers=admin_hdr)
    assert r.status_code == 409
