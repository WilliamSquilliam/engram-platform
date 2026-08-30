def test_health_liveness(client):
    assert client.get("/health").json() == {"ok": True}


def test_ready_reports_dependencies(client):
    r = client.get("/ready")
    body = r.json()
    assert body["db"] == "ok"  # the test DB is reachable
    # The ML service isn't running under test, so readiness is db-ok / ml-fail.
    assert "ml_service" in body
    assert r.status_code in (200, 503)
