"""Thin Lambda Cloud REST client (https://docs.lambda.ai, base https://cloud.lambda.ai/api/v1).

Torch-free control-plane HTTP, same house style as ml_client.py (sync httpx, matching the sync
routers). Lambda has NO stop state: terminate == billing $0, launch == a fresh box. The persistent
filesystem holds the weights + self-provision bundle so terminate/relaunch is the intended flow.

Config is read via the config module (not captured at import) so a test that sets LAMBDA_API_KEY /
LAMBDA_API_BASE after import is honored. The API key is NEVER logged or placed in an error message.
"""
import httpx

from . import config

# Reads are quick; a launch spins up a fresh box, so give it more headroom.
_TIMEOUT = 15.0
_LAUNCH_TIMEOUT = 30.0


class LambdaAPIError(Exception):
    """A non-2xx from the Lambda Cloud API. Carries the HTTP status and the API's error message
    (never the key). `str()` is human-readable for surfacing in a router response."""

    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(f"Lambda API error {status}: {message}")


def _headers() -> dict:
    # Bearer auth. Read the key fresh from config so tests can set it post-import.
    return {"Authorization": f"Bearer {config.LAMBDA_API_KEY}"}


def _extract_message(resp: httpx.Response) -> str:
    """Lambda wraps errors as {"error": {"code", "message", ...}}. Pull the message; fall back to the
    reason phrase. NEVER echo request headers/body — they carry the bearer key."""
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001  (non-JSON error body)
        return resp.reason_phrase or "request failed"
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict) and err.get("message"):
        return str(err["message"])
    if isinstance(err, str):
        return err
    return resp.reason_phrase or "request failed"


def _request(method: str, path: str, *, json: dict | None = None, timeout: float = _TIMEOUT) -> dict:
    url = f"{config.LAMBDA_API_BASE.rstrip('/')}/{path.lstrip('/')}"
    resp = httpx.request(method, url, headers=_headers(), json=json, timeout=timeout)
    if resp.status_code // 100 != 2:
        raise LambdaAPIError(resp.status_code, _extract_message(resp))
    # Lambda envelopes successful bodies under {"data": ...}.
    body = resp.json()
    return body.get("data", body) if isinstance(body, dict) else body


def list_instances() -> list[dict]:
    """All instances on the account (running/booting/terminating)."""
    return _request("GET", "/instances")


def get_instance(instance_id: str) -> dict:
    return _request("GET", f"/instances/{instance_id}")


def launch_instance(
    region_name: str,
    instance_type_name: str,
    ssh_key_names: list[str],
    file_system_names: list[str],
    name: str,
    user_data: str,
) -> dict:
    """Launch a fresh box. file_system_names attaches the persistent FS (weights + self-provision
    bundle); user_data is the cloud-init script that provisions the box unattended. Returns the
    launch response (contains instance_ids)."""
    payload = {
        "region_name": region_name,
        "instance_type_name": instance_type_name,
        "ssh_key_names": ssh_key_names,
        "file_system_names": file_system_names,
        "name": name,
        "user_data": user_data,
    }
    return _request("POST", "/instance-operations/launch", json=payload, timeout=_LAUNCH_TIMEOUT)


def terminate_instances(instance_ids: list[str]) -> dict:
    """Terminate == the Lambda "stop" (billing $0). Safe because the FS is persistent."""
    return _request(
        "POST", "/instance-operations/terminate", json={"instance_ids": instance_ids}
    )


def list_instance_types() -> dict:
    """The type catalog: {type_name: {instance_type: {...price_cents_per_hour...},
    regions_with_capacity_available: [...]}}. Used for type + region discovery at launch."""
    return _request("GET", "/instance-types")


def list_file_systems() -> list[dict]:
    """Persistent filesystems on the account (each carries its region). The launch region MUST host
    LAMBDA_FS_NAME or the fresh box has no weights/self-provision bundle."""
    return _request("GET", "/file-systems")


def put_firewall_rules(rules: list[dict]) -> dict:
    """Replace the account-global inbound firewall rules (idempotent). We set the same tcp 22/80/443
    + icmp set every launch, so a repeat is a no-op."""
    return _request("PUT", "/firewall-rules", json={"data": rules})
