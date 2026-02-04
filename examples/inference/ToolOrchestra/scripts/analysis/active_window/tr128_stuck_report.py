#!/usr/bin/env python3
"""
TR128 stuck incident report.

Inputs:
- <run_dir>/kv_cache_timeseries.csv
- <run_dir>/driver.log

Outputs:
- how many tasks completed vs expected
- last completed task timestamp
- KV cache samples around last completed task (+/- 10 samples)
- KV cache min/max distribution from last task -> end of KV log
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# Allow running as a standalone script (no package install needed).
sys.path.insert(0, os.path.dirname(__file__))

from common import KvSample, parse_kv_cache_timeseries, parse_task_done_timestamps  # type: ignore


def closest_index(kv: List[KvSample], target_ts: int) -> int:
    return min(range(len(kv)), key=lambda i: abs(kv[i].ts_unix - target_ts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--expected-total", type=int, default=278)
    args = ap.parse_args()

    run_dir = args.run_dir
    kv = parse_kv_cache_timeseries(run_dir / "kv_cache_timeseries.csv")
    tasks = parse_task_done_timestamps(run_dir / "driver.log")

    completed = len(tasks)
    stuck = args.expected_total - completed

    print("=" * 80)
    print("TR128 stuck report")
    print("=" * 80)
    print(f"run_dir: {run_dir}")
    print(f"expected_total: {args.expected_total}")
    print(f"completed_tasks: {completed}")
    print(f"not_completed: {stuck}")

    if not tasks or not kv:
        print("ERROR: missing tasks or kv samples")
        return 1

    last_task_ts = tasks[-1]
    print(f"\nlast_completed_task_ts: {datetime.fromtimestamp(last_task_ts).strftime('%Y-%m-%d %H:%M:%S')}")

    idx = closest_index(kv, last_task_ts)
    start = max(0, idx - 10)
    end = min(len(kv), idx + 11)

    print("\nKV around last completed task (±10 samples):")
    print("rel_s  time       kv_pct")
    for i in range(start, end):
        s = kv[i]
        rel = s.ts_unix - last_task_ts
        mark = "  <==" if i == idx else ""
        print(f"{rel:+5d}  {datetime.fromtimestamp(s.ts_unix).strftime('%H:%M:%S')}  {s.usage_pct:6.2f}%{mark}")

    kv_end_ts = kv[-1].ts_unix
    stuck_duration_s = kv_end_ts - last_task_ts
    print(f"\nKV log end: {datetime.fromtimestamp(kv_end_ts).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"stuck_duration: {stuck_duration_s}s = {stuck_duration_s/60:.1f}min = {stuck_duration_s/3600:.2f}hr")

    stuck_kv = [s.usage_pct for s in kv if s.ts_unix >= last_task_ts]
    print("\nKV during stuck period (from last task -> end of KV log):")
    print(f"samples: {len(stuck_kv)}")
    print(f"min: {min(stuck_kv):.2f}%")
    print(f"max: {max(stuck_kv):.2f}%")
    print(f"avg: {sum(stuck_kv)/len(stuck_kv):.2f}%")

    # Coarse buckets
    buckets = [(0, 1), (1, 30), (30, 70), (70, 100.0001)]
    print("\nDistribution:")
    for lo, hi in buckets:
        cnt = sum(1 for u in stuck_kv if lo <= u < hi)
        pct = 100.0 * cnt / len(stuck_kv)
        print(f"  {lo:>4.0f}-{hi:<4.0f}%: {cnt:>5d} ({pct:>5.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


