#!/usr/bin/env python3
"""
For each run, compute how many tasks/steps fall:
- before active window (first KV > threshold)
- inside active window (first KV > threshold .. last KV > threshold)
- after active window (after last KV > threshold)
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Allow running as a standalone script (no package install needed).
sys.path.insert(0, os.path.dirname(__file__))

from common import (  # type: ignore
    active_window_first_last_above,
    parse_kv_cache_timeseries,
    parse_step_done_timestamps,
    parse_task_done_timestamps,
)


@dataclass(frozen=True)
class Breakdown:
    first_task_ts: int
    last_task_ts: int
    win_start: int
    win_end: int
    tasks_before: int
    tasks_in: int
    tasks_after: int
    steps_before: int
    steps_in: int
    steps_after: int

    @property
    def window_min(self) -> float:
        return (self.win_end - self.win_start) / 60.0

    @property
    def gap_last_gt70_to_last_task_min(self) -> float:
        return (self.last_task_ts - self.win_end) / 60.0


def compute(run_dir: Path, threshold_pct: float) -> Breakdown:
    kv = parse_kv_cache_timeseries(run_dir / "kv_cache_timeseries.csv")
    win_start, win_end = active_window_first_last_above(kv, threshold_pct)

    tasks = parse_task_done_timestamps(run_dir / "driver.log")
    steps = parse_step_done_timestamps(run_dir / "driver.log")

    first_task = min(tasks) if tasks else 0
    last_task = max(tasks) if tasks else 0

    tasks_before = sum(1 for t in tasks if t < win_start)
    tasks_in = sum(1 for t in tasks if win_start <= t <= win_end)
    tasks_after = sum(1 for t in tasks if t > win_end)

    steps_before = sum(1 for s in steps if s < win_start)
    steps_in = sum(1 for s in steps if win_start <= s <= win_end)
    steps_after = sum(1 for s in steps if s > win_end)

    return Breakdown(
        first_task_ts=first_task,
        last_task_ts=last_task,
        win_start=win_start,
        win_end=win_end,
        tasks_before=tasks_before,
        tasks_in=tasks_in,
        tasks_after=tasks_after,
        steps_before=steps_before,
        steps_in=steps_in,
        steps_after=steps_after,
    )


def fmt_hms(ts: int) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=70.0)
    ap.add_argument("--name", type=str, required=True, help="Label to print (e.g., TR32)")
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()

    b = compute(args.run_dir, args.threshold)

    print(f"\n=== {args.name} (active window: first>70% -> last>70%) ===")
    print(f"first_task={fmt_hms(b.first_task_ts)}  win_start={fmt_hms(b.win_start)}  win_end={fmt_hms(b.win_end)}  last_task={fmt_hms(b.last_task_ts)}")
    print(f"window={b.window_min:.1f} min  gap(win_end->last_task)={b.gap_last_gt70_to_last_task_min:.1f} min")
    print(f"tasks: before/in/after = {b.tasks_before}/{b.tasks_in}/{b.tasks_after}")
    print(f"steps: before/in/after = {b.steps_before}/{b.steps_in}/{b.steps_after}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


