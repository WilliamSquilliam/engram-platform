"""POST /invalidate on the vLLM serving frontend (vllm_inference.py) — the serving-side half
of the data-deletion path. No GPU/vLLM: the endpoint must work BEFORE any engine is built (an
offboard right after boot), which is exactly the path this exercises. Needs torch + fastapi +
the engram-cartridge wheel (the handler imports cartridges.cart_store for id validation), so
it runs where those are installed (dev box / GPU box), not in the torch-free backend suite.

Ported from Engram-Smart-CAG cartridges/tests when the platform split — the module under test
lives HERE now. The module is imported from its file path with the registry dir pointed at a
pytest tmp dir FIRST — vllm_inference sets CARTRIDGE_REGISTRY_DIR at import time and the
registry mkdirs it."""
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("cartridges")

_MOD_PATH = Path(__file__).resolve().parent / "vllm_inference.py"


@pytest.fixture()
def frontend(tmp_path, monkeypatch):
    """(module, TestClient, registry_dir) with a tmp registry dir and auth open."""
    reg_dir = tmp_path / "reg"
    monkeypatch.setenv("CARTRIDGE_REGISTRY_DIR", str(reg_dir))
    monkeypatch.setenv("CARTRIDGE_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("ML_AUTH_TOKEN", "")
    spec = importlib.util.spec_from_file_location("vllm_inference_under_test", _MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
        from fastapi.testclient import TestClient
        yield mod, TestClient(mod.app), reg_dir
    finally:
        sys.modules.pop(spec.name, None)


def test_invalidate_purges_frontend_caches_and_publishes_tombstones(frontend):
    mod, client, reg_dir = frontend
    # Prime the frontend caches the way a served query would (values are arbitrary).
    mod._kv_len["docA"] = 512
    mod._cart_ids["docA"] = None
    mod._cart_binding_ok["docA"] = True
    r = client.post("/invalidate", json={"cart_ids": ["docA", "never-served"]})
    assert r.status_code == 200
    assert r.json() == {"invalidated": 2, "backend": "dir"}
    # Frontend caches are cleared; ids that were never served are harmless (idempotent).
    assert "docA" not in mod._kv_len and "docA" not in mod._cart_ids
    assert "docA" not in mod._cart_binding_ok
    # Tombstones landed where the EngineCore-subprocess connectors poll for them.
    stems = {p.stem for p in (reg_dir / "invalidations").glob("*.json")}
    assert stems == {"docA", "never-served"}


def test_invalidate_rejects_traversal_ids(frontend):
    _, client, reg_dir = frontend
    r = client.post("/invalidate", json={"cart_ids": ["ok-id", "../evil"]})
    assert r.status_code == 400
    # All-or-nothing on validation: the bad batch published no tombstones at all.
    assert not (reg_dir / "invalidations").exists()


def test_invalidate_sits_behind_ml_auth(frontend, monkeypatch):
    """A lifecycle action must not be callable by anything that can reach the port when the
    deployment runs with the shared ML token (BYOC does)."""
    _, client, _ = frontend
    monkeypatch.setenv("ML_AUTH_TOKEN", "sekrit")
    assert client.post("/invalidate", json={"cart_ids": ["d"]}).status_code == 401
    r = client.post("/invalidate", json={"cart_ids": ["d"]},
                    headers={"Authorization": "Bearer sekrit"})
    assert r.status_code == 200
