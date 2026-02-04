#!/usr/bin/env python3
"""
Simple stress test for the FRAMES wiki retrieval service (/retrieve).

Example:
  python stress_test_wiki_retrieval.py --base-url http://research-secure-17:8766 --requests 256 --concurrency 32
"""

from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = int(round((p / 100.0) * (len(s) - 1)))
    k = max(0, min(k, len(s) - 1))
    return s[k]


def one_call(url: str, payload: dict[str, Any], timeout_s: float) -> float:
    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    _ = r.json()
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="e.g. http://research-secure-17:8766")
    ap.add_argument("--requests", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--timeout-s", type=float, default=60.0)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--internal-k", type=int, default=1000)
    ap.add_argument(
        "--query",
        type=str,
        default="What is the capital of France?",
        help="Query string to send repeatedly.",
    )
    args = ap.parse_args()

    url = args.base_url.rstrip("/") + "/retrieve"
    payload = {
        "queries": [args.query],
        "topk": args.topk,
        "return_scores": True,
        # Server ignores/accepts extra fields; keep stable.
        "eid": "stress_test",
    }

    print(f"URL: {url}")
    print(f"Requests: {args.requests}  Concurrency: {args.concurrency}  Timeout: {args.timeout_s}s")

    times: list[float] = []
    errors = 0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(one_call, url, payload, args.timeout_s) for _ in range(args.requests)]
        for f in as_completed(futs):
            try:
                times.append(f.result())
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"ERROR: {repr(e)}")
    wall = time.perf_counter() - t0

    ok = len(times)
    print(f"Done: ok={ok} errors={errors} wall={wall:.2f}s throughput={ok / wall:.2f} rps")
    if not times:
        return

    mean = statistics.mean(times)
    p50 = pct(times, 50)
    p80 = pct(times, 80)
    p90 = pct(times, 90)
    p95 = pct(times, 95)
    p99 = pct(times, 99)

    print(
        "Latency (seconds): "
        f"mean={mean:.3f} p50={p50:.3f} p80={p80:.3f} p90={p90:.3f} p95={p95:.3f} p99={p99:.3f} "
        f"min={min(times):.3f} max={max(times):.3f}"
    )


if __name__ == "__main__":
    main()


