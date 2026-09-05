#!/usr/bin/env python3
"""KV-dtype accuracy/latency gate: run a fixed question set against fixed carts and
save answers + timings to a JSON file. Run once per engine config (bf16 baseline,
fp8 candidate), then diff the files: greedy decode means a lossless change yields
IDENTICAL answers; divergence is the signal to inspect.

Runs ON the box against localhost:8002. Usage:
    kv_gate.py --out /tmp/gate_bf16.json
    kv_gate.py --out /tmp/gate_fp8.json
    kv_gate.py --compare /tmp/gate_bf16.json /tmp/gate_fp8.json
"""
import argparse
import json
import statistics
import time

import httpx

BASE = "http://127.0.0.1:8002"
TENANT = "5cb99fe2f5954b4aa53ded88cf6a89e5"

# (cart doc_ids, question) — T1's job-document corpus, single- and multi-cart routes.
CASES = [
    ([f"{TENANT}__JPMC_Job_Desc"], "What are the required qualifications for this role?"),
    ([f"{TENANT}__JPMC_Job_Desc"], "What is the compensation or salary range?"),
    ([f"{TENANT}__JPMC_Job_Offer_Software_Engineer_III"], "What position and start terms does the offer letter state?"),
    ([f"{TENANT}__JPMC_JPMC_Interview_Process"], "What are the interview stages and how long is each?"),
    ([f"{TENANT}__JPMC_Placement_Call_Notes"], "Who are the two professionals described and what are their roles?"),
    ([f"{TENANT}__JPMC_William_Stephenson_Application_Cover_Letter"], "What experience does the applicant highlight?"),
    ([f"{TENANT}__JPMC_Job_Desc", f"{TENANT}__JPMC_Job_Offer_Software_Engineer_III"],
     "Compare the responsibilities in the job description with what the offer letter says."),
    ([f"{TENANT}__JPMC_JPMC_Interview_Process", f"{TENANT}__JPMC_Job_Desc"],
     "Given the interview stages, which required skills should a candidate prepare most?"),
]


KEYS = [
    ["Java", "Python"],
    ["salary"],
    ["Associate Software Engineer III"],
    ["45"],
    ["Shilpa", "Werner"],
    ["Software Engineer"],
    ["responsibilit"],
    ["coding"],
]


def _token() -> str:
    with open("/etc/engram/serving.env", encoding="utf-8") as f:
        for line in f:
            if line.startswith("ML_AUTH_TOKEN="):
                return line.split("=", 1)[1].strip()
    return ""


def run(out_path: str) -> None:
    client = httpx.Client(timeout=300.0, headers={"Authorization": f"Bearer {_token()}"})
    results = []
    for doc_ids, q in CASES:
        t0 = time.perf_counter()
        r = client.post(f"{BASE}/query", json={"doc_ids": doc_ids, "question": q, "max_tokens": 256})
        r.raise_for_status()
        body = r.json()
        wall = round(time.perf_counter() - t0, 2)
        keys = KEYS[len(results)]
        missing = [k for k in keys if k.lower() not in body["answer"].lower()]
        leaked = body["answer"].lstrip().startswith(("We need", "The user", "The need"))
        results.append({"doc_ids": doc_ids, "q": q, "answer": body["answer"],
                        "wall_s": wall, "missing_keys": missing, "leaked_thinking": leaked,
                        "metrics": body.get("metrics", {})})
        print(f"  {wall:5.2f}s  {q[:60]}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    med = statistics.median(r["wall_s"] for r in results)
    print(f"saved {len(results)} answers to {out_path} (median wall {med:.2f}s)")


def compare(a_path: str, b_path: str) -> int:
    a = json.load(open(a_path, encoding="utf-8"))
    b = json.load(open(b_path, encoding="utf-8"))
    ident = 0
    for ra, rb in zip(a, b):
        same = ra["answer"] == rb["answer"]
        ident += same
        mark = "IDENTICAL" if same else "DIFFERS"
        flags = lambda r: ("" if not r.get("missing_keys") else f" MISSING{r['missing_keys']}") +                           (" LEAKED" if r.get("leaked_thinking") else "")
        print(f"[{mark}] {ra['q'][:55]}  A:{flags(ra) or ' ok'}  B:{flags(rb) or ' ok'}")
        if not same:
            print(f"  A: {ra['answer'][:180]}")
            print(f"  B: {rb['answer'][:180]}")
    med_a = statistics.median(r["wall_s"] for r in a)
    med_b = statistics.median(r["wall_s"] for r in b)
    print(f"\n{ident}/{len(a)} identical · median wall A={med_a:.2f}s B={med_b:.2f}s")
    return 0 if ident == len(a) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        raise SystemExit(compare(*args.compare))
    run(args.out or "/tmp/gate.json")
