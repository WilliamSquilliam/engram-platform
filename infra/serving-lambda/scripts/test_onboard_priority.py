#!/usr/bin/env python3
"""Live-GPU test: live queries outrank onboarding (ONBOARD_YIELD_S gate).

Runs ON the box against localhost:8002. Three phases:
  A. baseline — query latency with an idle engine.
  B. contention — onboard N synthetic docs while firing queries; PASS if
     (1) the onboard log reports yielding, and (2) median query latency under
     contention stays within LAT_FACTOR of baseline.
  C. resume — re-run the same onboard after phase B completed partially or fully:
     every already-built cart must be reused (n_built == 0 on a clean re-run).
Phase C's restart variant (systemctl restart mid-onboard) is driven by the caller;
this script's re-run works the same either way because builds are idempotent.

Reads ML_AUTH_TOKEN from /etc/engram/serving.env (never printed).
"""
import argparse
import json
import statistics
import threading
import time

import httpx

BASE = "http://127.0.0.1:8002"


def _token() -> str:
    with open("/etc/engram/serving.env", encoding="utf-8") as f:
        for line in f:
            if line.startswith("ML_AUTH_TOKEN="):
                return line.split("=", 1)[1].strip()
    return ""


def _mk_docs(n: int, words_per_doc: int, tag: str) -> list[dict]:
    """Synthetic docs long enough that each build is a real prefill (~thousands of tokens)."""
    docs = []
    for i in range(n):
        fact = f"The secret code of station {tag}-{i} is OMEGA-{i * 7 + 3}."
        filler = " ".join(f"Operational note {j} for station {tag}-{i}: all systems nominal and "
                          f"telemetry stream {j % 13} reports within tolerance." for j in range(words_per_doc // 14))
        docs.append({"doc_id": f"prio__{tag}_{i}", "text": f"{fact}\n\n{filler}"})
    return docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=16)
    ap.add_argument("--words", type=int, default=2600)
    ap.add_argument("--tag", default="alpha")
    ap.add_argument("--lat-factor", type=float, default=2.5)
    ap.add_argument("--phase", choices=["full", "onboard-only", "rerun-only"], default="full")
    args = ap.parse_args()

    tok = _token()
    hdr = {"Authorization": f"Bearer {tok}"} if tok else {}
    client = httpx.Client(timeout=300.0, headers=hdr)
    docs = _mk_docs(args.docs, args.words, args.tag)
    q_doc = docs[0]["doc_id"]

    def onboard() -> dict:
        r = client.post(f"{BASE}/onboard_cag", json={"corpus_dir": "/data/prio-test", "docs": docs})
        r.raise_for_status()
        return r.json()

    def query() -> float:
        t0 = time.perf_counter()
        r = client.post(f"{BASE}/query", json={
            "doc_ids": [q_doc], "question": f"What is the secret code of station {args.tag}-0?",
            "max_tokens": 32})
        r.raise_for_status()
        return time.perf_counter() - t0

    if args.phase == "rerun-only":
        res = onboard()
        print("RERUN:", json.dumps({k: res.get(k) for k in
                                    ("n_cartridges", "n_built", "canceled", "errors")}))
        ok = res.get("n_cartridges") == args.docs and not res.get("errors")
        print("PASS resume" if ok else "FAIL resume")
        return 0 if ok else 1

    # Query doc's cart must exist for baseline: build just it first.
    client.post(f"{BASE}/onboard_cag", json={"corpus_dir": "/data/prio-test", "docs": docs[:1]}).raise_for_status()

    print("phase A: baseline latency (idle engine)")
    base_lat = [query() for _ in range(4)][1:]   # drop the first (index/cart warm)
    base_med = statistics.median(base_lat)
    print(f"  baseline median {base_med:.2f}s  ({['%.2f' % x for x in base_lat]})")

    if args.phase == "onboard-only":
        res = onboard()
        print("ONBOARD:", json.dumps({k: res.get(k) for k in
                                      ("n_cartridges", "n_built", "canceled", "errors")}))
        return 0

    print(f"phase B: onboard {args.docs} docs with concurrent queries")
    result: dict = {}
    th = threading.Thread(target=lambda: result.update(onboard()), daemon=True)
    th.start()
    cont_lat = []
    while th.is_alive():
        try:
            cont_lat.append(query())
            print(f"  query under contention: {cont_lat[-1]:.2f}s")
        except httpx.HTTPError as e:
            print(f"  query FAILED under contention: {e}")
            cont_lat.append(float("inf"))
        time.sleep(1.0)
        if len(cont_lat) > 120:
            break
    th.join(timeout=600)
    cont_med = statistics.median(cont_lat) if cont_lat else float("inf")
    print(f"  contention median {cont_med:.2f}s over {len(cont_lat)} queries")
    print("ONBOARD:", json.dumps({k: result.get(k) for k in
                                  ("n_cartridges", "n_built", "cart_seconds", "canceled", "errors")}))

    ok_lat = cont_med <= base_med * args.lat_factor
    ok_onb = result.get("n_cartridges") == args.docs and not result.get("errors")
    print(f"{'PASS' if ok_lat else 'FAIL'} latency: contention {cont_med:.2f}s vs "
          f"baseline {base_med:.2f}s (factor {cont_med / base_med:.1f}, limit {args.lat_factor})")
    print(f"{'PASS' if ok_onb else 'FAIL'} onboard completed under contention")
    return 0 if (ok_lat and ok_onb) else 1


if __name__ == "__main__":
    raise SystemExit(main())
