#!/usr/bin/env python3
"""
Compute throughput tables under the "active window" definition:
  active window = first KV > threshold  ->  last KV > threshold

Outputs:
- tasks/min table (tasks done per minute)
- steps/s table  (steps done per second)
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

# Allow running as a standalone script (no package install needed).
sys.path.insert(0, os.path.dirname(__file__))

from common import (  # type: ignore
    active_window_first_last_above,
    parse_kv_cache_timeseries,
    parse_step_done_timestamps,
    parse_task_done_timestamps,
)


@dataclass(frozen=True)
class RunStats:
    window_s: int
    tasks_in_window: int
    steps_in_window: int

    @property
    def window_min(self) -> float:
        return self.window_s / 60.0

    @property
    def tasks_per_min(self) -> float:
        return self.tasks_in_window / self.window_min

    @property
    def steps_per_sec(self) -> float:
        return self.steps_in_window / self.window_s


def compute_run_stats(run_dir: Path, threshold_pct: float) -> RunStats:
    kv_path = run_dir / "kv_cache_timeseries.csv"
    driver_log = run_dir / "driver.log"
    kv = parse_kv_cache_timeseries(kv_path)
    start_ts, end_ts = active_window_first_last_above(kv, threshold_pct)

    tasks = parse_task_done_timestamps(driver_log)
    steps = parse_step_done_timestamps(driver_log)

    tasks_in = sum(1 for t in tasks if start_ts <= t <= end_ts)
    steps_in = sum(1 for s in steps if start_ts <= s <= end_ts)

    return RunStats(window_s=end_ts - start_ts, tasks_in_window=tasks_in, steps_in_window=steps_in)


def winner_label(tr: float, ct: float) -> str:
    # +/- 5% = "相当"
    if ct == 0:
        return "TR"
    ratio = tr / ct
    if ratio > 1.05:
        return f"TR +{(ratio - 1) * 100:.0f}%"
    if ratio < 0.95:
        return f"CT +{(1 / ratio - 1) * 100:.0f}%"
    return "相当"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=70.0, help="KV cache threshold in percent (default: 70)")
    ap.add_argument("--tr32", type=Path, required=True)
    ap.add_argument("--tr64", type=Path, required=True)
    ap.add_argument("--tr128", type=Path, required=True)
    ap.add_argument("--ct32", type=Path, required=True)
    ap.add_argument("--ct64", type=Path, required=True)
    ap.add_argument("--ct128", type=Path, required=True)
    args = ap.parse_args()

    runs: Dict[str, RunStats] = {}
    for key, p in [
        ("TR32", args.tr32),
        ("TR64", args.tr64),
        ("TR128", args.tr128),
        ("CT32", args.ct32),
        ("CT64", args.ct64),
        ("CT128", args.ct128),
    ]:
        runs[key] = compute_run_stats(p, args.threshold)

    # Table 1
    print("\n### TABLE 1: Tasks done / minute (active window)\n")
    print("| 场景  | TR             | CT             | Winner |")
    print("| :---- | :------------- | :------------- | :----- |")
    for c in ("32", "64", "128"):
        tr = runs[f"TR{c}"].tasks_per_min
        ct = runs[f"CT{c}"].tasks_per_min
        print(f"| C={c}  | {tr:.2f} tasks/min | {ct:.2f} tasks/min | {winner_label(tr, ct)} |")

    # Table 2
    print("\n### TABLE 2: Steps / second (active window)\n")
    print("| 场景  | TR           | CT           | Winner |")
    print("| :---- | :----------- | :----------- | :----- |")
    for c in ("32", "64", "128"):
        tr = runs[f"TR{c}"].steps_per_sec
        ct = runs[f"CT{c}"].steps_per_sec
        print(f"| C={c}  | {tr:.2f} steps/s | {ct:.2f} steps/s | {winner_label(tr, ct)} |")

    # Raw appendix (useful for debugging)
    print("\n### Appendix: raw window stats\n")
    print("| Run | window(min) | tasks_in_window | steps_in_window | tasks/min | steps/s |")
    print("| :-- | ----------: | -------------: | -------------: | --------: | ------: |")
    for key in ("TR32", "CT32", "TR64", "CT64", "TR128", "CT128"):
        r = runs[key]
        print(
            f"| {key} | {r.window_min:.1f} | {r.tasks_in_window} | {r.steps_in_window} | {r.tasks_per_min:.2f} | {r.steps_per_sec:.2f} |"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


