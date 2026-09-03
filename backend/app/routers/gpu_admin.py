"""Platform-admin GPU controls for the Lambda Cloud serving box (E12).

Gated by require_platform_admin on EVERY route (a normal tenant admin 403s). Lambda has no stop
state: "stop" == terminate (billing $0), "start" == launch a fresh box. A persistent filesystem
(LAMBDA_FS_NAME) holds the model weights + a self-provision bundle, and a cloud-init user_data script
provisions the fresh box unattended, so terminate/relaunch is the intended flow.

The status read derives a lifecycle state from the Lambda instance + two unauthenticated health
probes (the serve plane and the onboard plane) and, best-effort, reconciles the serve/onboard DNS
records to the box IP. Neither the health probes nor the DNS reconcile can fail the status read.
"""
import logging
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, status

from .. import cloudflare_dns, config, lambda_cloud
from ..deps import require_platform_admin
from ..lambda_cloud import LambdaAPIError
from ..models import User
from ..schemas import GpuInstanceResp, GpuStartResp, GpuStatusResp, GpuStopResp

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/platform-admin/gpu", tags=["platform-admin"])

# A dead (or dying) instance carrying our name doesn't count as "the box is up".
_DEAD_STATUSES = ("terminated", "terminating")
_HEALTH_TIMEOUT = 3.0

# Account-global inbound firewall: SSH, the ACME http-01 challenge + http->https redirect, HTTPS
# (Caddy terminates TLS), and ping. Set idempotently before every launch.
_FIREWALL_RULES = [
    {"protocol": "tcp", "port_range": [22, 22], "source_network": "0.0.0.0/0", "description": "ssh"},
    {"protocol": "tcp", "port_range": [80, 80], "source_network": "0.0.0.0/0",
     "description": "http-01 + redirect"},
    {"protocol": "tcp", "port_range": [443, 443], "source_network": "0.0.0.0/0",
     "description": "https (Caddy)"},
    {"protocol": "icmp", "source_network": "0.0.0.0/0", "description": "ping"},
]


def _user_data() -> str:
    """cloud-init that waits (up to ~15 min) for the persistent FS to mount, then runs the
    self-provision bundle. The FS name comes from config so a rename is one env change, not a code
    edit."""
    fs = config.LAMBDA_FS_NAME
    script = (
        f'for i in $(seq 1 90); do [ -f /lambda/nfs/{fs}/engram/self-provision.sh ] && break; '
        f'sleep 10; done; bash /lambda/nfs/{fs}/engram/self-provision.sh '
        f'>> /var/log/engram-self-provision.log 2>&1'
    )
    return (
        "#cloud-config\n"
        "runcmd:\n"
        f'  - [ bash, -c, "{script}" ]\n'
    )


def _our_instance() -> dict | None:
    """The one non-dead instance named LAMBDA_INSTANCE_NAME, if any. A "terminating" one is returned
    too (so status can report the transient state); "terminated" is treated as gone."""
    name = config.LAMBDA_INSTANCE_NAME
    ours = [
        i for i in lambda_cloud.list_instances()
        if i.get("name") == name and i.get("status") != "terminated"
    ]
    return ours[0] if ours else None


def _instance_ip(inst: dict) -> str | None:
    return inst.get("ip") or None


def _price_cents(inst: dict) -> int | None:
    itype = inst.get("instance_type") or {}
    price = itype.get("price_cents_per_hour")
    return int(price) if price is not None else None


def _to_instance_resp(inst: dict) -> GpuInstanceResp:
    itype = inst.get("instance_type") or {}
    region = inst.get("region") or {}
    return GpuInstanceResp(
        id=inst.get("id", ""),
        name=inst.get("name") or config.LAMBDA_INSTANCE_NAME,
        type=itype.get("name", ""),
        region=region.get("name", "") if isinstance(region, dict) else str(region),
        ip=_instance_ip(inst),
        price_cents_per_hour=_price_cents(inst),
    )


def _probe(base_url: str) -> tuple[bool, bool]:
    """Unauthenticated GET {base_url}/health, 3s. Returns (reachable, engine_ready). Any failure is
    (False, False) — a probe must never raise a 500 into the status read."""
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/health", timeout=_HEALTH_TIMEOUT)
        if r.status_code != 200:
            return False, False
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        return True, bool(body.get("engine_ready")) if isinstance(body, dict) else True
    except Exception:  # noqa: BLE001  (unreachable, not an error)
        return False, False


def _is_dns_host(host: str | None) -> bool:
    """A reconcilable host is a real DNS name — not localhost and not an IP literal."""
    if not host or host == "localhost":
        return False
    # An IP literal has no letters; a DNS name does. Cheap and sufficient here.
    return any(c.isalpha() for c in host)


def _reconcile_dns(ip: str) -> bool | None:
    """Best-effort: point the serve + onboard hostnames at `ip` when they've drifted. Returns True
    (both point at ip), False (a needed upsert failed), or None (nothing to do — no creds or the
    hosts aren't DNS names). Wrapped so it can NEVER fail the status read."""
    if not config.CLOUDFLARE_API_TOKEN or not config.CLOUDFLARE_ZONE_ID:
        return None
    serve_host = urlparse(config.INFERENCE_SERVICE_URL).hostname
    onboard_host = urlparse(config.ML_SERVICE_URL).hostname
    hosts = {h for h in (serve_host, onboard_host) if _is_dns_host(h)}
    if not hosts:
        return None
    try:
        ok = True
        for host in hosts:
            if not cloudflare_dns.upsert_a_record(host, ip):
                ok = False
        return ok
    except Exception:  # noqa: BLE001  (reconcile is a side effect; never fail status)
        logger.warning("DNS reconcile raised; reporting dns_pointed=False", exc_info=True)
        return False


@router.get("/status", response_model=GpuStatusResp)
def gpu_status(_: User = Depends(require_platform_admin)) -> GpuStatusResp:
    """Derived lifecycle state of the serving box + health probes. 200 even when disabled (enabled
    false, state "offline")."""
    if not config.GPU_CONTROL_ENABLED:
        return GpuStatusResp(enabled=False, state="offline", instance=None)

    try:
        inst = _our_instance()
    except LambdaAPIError:
        # Can't reach Lambda -> treat as offline rather than 500 the console.
        logger.warning("Lambda list_instances failed during status read", exc_info=True)
        return GpuStatusResp(enabled=True, state="offline", instance=None)

    if inst is None:
        return GpuStatusResp(enabled=True, state="offline", instance=None)

    inst_status = inst.get("status")
    if inst_status == "terminating":
        return GpuStatusResp(
            enabled=True, state="terminating", instance=_to_instance_resp(inst),
            hourly_usd=(lambda c: c / 100 if c is not None else None)(_price_cents(inst)),
        )

    ip = _instance_ip(inst)
    resp = GpuStatusResp(enabled=True, state="booting", instance=_to_instance_resp(inst))
    price_cents = _price_cents(inst)
    resp.hourly_usd = price_cents / 100 if price_cents is not None else None

    # Not active or no IP yet -> still coming up.
    if inst_status != "active" or not ip:
        resp.state = "booting"
        return resp

    # Active + IP: probe the two planes and reconcile DNS (best-effort).
    serve_reachable, engine_ready = _probe(config.INFERENCE_SERVICE_URL)
    onboard_reachable, _onboard_engine = _probe(config.ML_SERVICE_URL)
    resp.serve_reachable = serve_reachable
    resp.engine_ready = engine_ready
    resp.onboard_reachable = onboard_reachable

    try:
        resp.dns_pointed = _reconcile_dns(ip)
    except Exception:  # noqa: BLE001  (belt-and-suspenders — reconcile must never 500)
        logger.warning("DNS reconcile failed unexpectedly", exc_info=True)
        resp.dns_pointed = None

    if not serve_reachable:
        resp.state = "provisioning"
    elif not engine_ready:
        resp.state = "warming"
    else:
        resp.state = "serving"
    return resp


def _ensure_firewall() -> None:
    """Idempotent, account-global firewall. Best-effort: a failure is logged and the launch proceeds
    (the rules may already be in place from a prior launch)."""
    try:
        lambda_cloud.put_firewall_rules(_FIREWALL_RULES)
    except LambdaAPIError:
        logger.warning("Firewall rule update failed; continuing with launch", exc_info=True)


def _region_names(regions: list | None) -> set[str]:
    """Region lists appear as plain names OR as {"name": ...} objects depending on the endpoint —
    normalize to names (same dual handling launch.sh does)."""
    out = set()
    for r in regions or []:
        name = r.get("name") if isinstance(r, dict) else r
        if name:
            out.add(name)
    return out


def _discover_type_and_region() -> tuple[str, str]:
    """Pick an instance type + a region that BOTH has capacity for it AND hosts LAMBDA_FS_NAME (the
    weights + self-provision bundle live there). Raises HTTPException 503 when no such pairing exists,
    naming the type filters and the FS so the operator knows exactly what's missing."""
    types = lambda_cloud.list_instance_types()

    # BOTH filters are substrings (the fallback's real type name is e.g. gpu_2x_h100_sxm5, not
    # "2x_h100_sxm"). Preferred matches first, then fallback matches — and the fallback must still
    # be tried when a preferred type exists but has no usable region.
    preferred = sorted(n for n in types if config.LAMBDA_TYPE_FILTER in n)
    fallback = sorted(
        n for n in types if config.LAMBDA_TYPE_FALLBACK in n and n not in preferred
    )
    candidate_names = preferred + fallback

    # Regions where LAMBDA_FS_NAME exists (the FS is region-pinned).
    fs_regions = _region_names(
        [fs.get("region") for fs in lambda_cloud.list_file_systems()
         if fs.get("name") == config.LAMBDA_FS_NAME]
    )

    for name in candidate_names:
        entry = types.get(name) or {}
        # Capacity regions live at the top level or under instance_type depending on API shape.
        regions = (entry.get("regions_with_capacity_available")
                   or (entry.get("instance_type") or {}).get("regions_with_capacity_available"))
        usable = _region_names(regions) & fs_regions
        if usable:
            return name, sorted(usable)[0]

    fs_desc = f"'{config.LAMBDA_FS_NAME}'" + (
        f" (in region(s) {sorted(r for r in fs_regions if r)})" if fs_regions else " (not found)"
    )
    raise HTTPException(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        f"No Lambda capacity for type filter '{config.LAMBDA_TYPE_FILTER}' / fallback "
        f"'{config.LAMBDA_TYPE_FALLBACK}' in a region that also hosts the filesystem {fs_desc}.",
    )


@router.post("/start", response_model=GpuStartResp, status_code=status.HTTP_202_ACCEPTED)
def gpu_start(_: User = Depends(require_platform_admin)) -> GpuStartResp:
    """Launch the serving box: ensure the firewall, discover a type+region pairing that has capacity
    AND hosts the FS, then launch with the self-provision cloud-init. 503 when disabled or no
    capacity; 409 when a box is already up."""
    if not config.GPU_CONTROL_ENABLED:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "GPU control is not enabled")

    if _our_instance() is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"An instance named '{config.LAMBDA_INSTANCE_NAME}' is already running",
        )

    _ensure_firewall()
    type_name, region_name = _discover_type_and_region()

    try:
        result = lambda_cloud.launch_instance(
            region_name,
            type_name,
            [config.LAMBDA_SSH_KEY_NAME],
            [config.LAMBDA_FS_NAME],
            config.LAMBDA_INSTANCE_NAME,
            _user_data(),
        )
    except LambdaAPIError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Launch failed: {exc.message}") from exc

    ids = result.get("instance_ids") or []
    instance_id = ids[0] if ids else ""
    logger.info(
        "Launched GPU serving box %s: type=%s region=%s", instance_id, type_name, region_name
    )
    return GpuStartResp(instance_id=instance_id, state="booting")


@router.post("/stop", response_model=GpuStopResp, status_code=status.HTTP_202_ACCEPTED)
def gpu_stop(_: User = Depends(require_platform_admin)) -> GpuStopResp:
    """Terminate the serving box (Lambda "stop" == terminate, billing $0; the FS persists). 503 when
    disabled; 409 when nothing is running."""
    if not config.GPU_CONTROL_ENABLED:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "GPU control is not enabled")

    inst = _our_instance()
    if inst is None or inst.get("status") in _DEAD_STATUSES:
        raise HTTPException(status.HTTP_409_CONFLICT, "No running instance to stop")

    itype = (inst.get("instance_type") or {}).get("name", "")
    region = inst.get("region")
    region_name = region.get("name") if isinstance(region, dict) else region
    lambda_cloud.terminate_instances([inst["id"]])
    # Dangling-DNS guard: delete the gpu A records now that their IP is being released (an
    # inherited IP + still-pointing DNS lets a stranger mint a valid cert for our hostname
    # via HTTP-01 and harvest the ML bearer token). Best-effort — the next start's DNS
    # reconcile recreates them; a delete failure never fails the stop.
    for url in (config.INFERENCE_SERVICE_URL, config.ML_SERVICE_URL):
        host = urlparse(url).hostname
        if _is_dns_host(host):
            cloudflare_dns.delete_a_record(host)
    logger.info(
        "Terminating GPU serving box %s: type=%s region=%s", inst["id"], itype, region_name
    )
    return GpuStopResp(state="terminating")
