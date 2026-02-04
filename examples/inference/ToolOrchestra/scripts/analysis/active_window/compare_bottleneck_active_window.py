#!/usr/bin/env python3
"""
Compare bottleneck signals within the "active window":
  active window = first KV > threshold  ->  last KV > threshold

This script is meant to explain *why* a run is faster/slower at high concurrency by
looking at:
  - step rates by (from_role -> to_role)
  - LLM call tail behavior (very long vLLM calls)
  - user_sim and expert_call latency distributions

Example (TR/CT/Baseline C=128):
  python3 scripts/analysis/active_window/compare_bottleneck_active_window.py \
    --threshold 70 \
    --run TR128=outputs/.../thunderreact_c128 \
    --run CT128=outputs/.../continuum_c128_... \
    --run BL128=outputs/.../baseline_c128_...
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Allow running as a standalone script (no package install needed).
sys.path.insert(0, os.path.dirname(__file__))

from common import active_window_first_last_above, parse_kv_cache_timeseries  # type: ignore


_TS_ANY = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_DURATION_MS = re.compile(r"\bduration_ms=([0-9]+(?:\.[0-9]+)?)")
_TOTAL_DURATION_MS = re.compile(r"\btotal_duration_ms=([0-9]+(?:\.[0-9]+)?)")
_FROM_ROLE = re.compile(r"\bfrom_role=([^\s]+)")
_TO_ROLE = re.compile(r"\bto_role=([^\s]+)")
_TASK = re.compile(r"\btask=([^\s]+)")
_STEP = re.compile(r"\bstep=([0-9]+)")


def _quantile(sorted_vals: List[float], q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


@dataclass(frozen=True)
class Stats:
    n: int
    mean: float
    p50: float
    p90: float
    p99: float
    min: float
    max: float

    @classmethod
    def from_values(cls, values: List[float]) -> Optional["Stats"]:
        if not values:
            return None
        s = sorted(values)
        n = len(s)
        mean = sum(s) / n
        return cls(
            n=n,
            mean=mean,
            p50=_quantile(s, 0.50) or 0.0,
            p90=_quantile(s, 0.90) or 0.0,
            p99=_quantile(s, 0.99) or 0.0,
            min=float(s[0]),
            max=float(s[-1]),
        )


@dataclass(frozen=True)
class RunBottleneck:
    name: str
    run_dir: Path
    win_start: int
    win_end: int
    window_s: int

    # Step counts (active window)
    step_total: int
    step_by_edge: Dict[str, int]  # e.g., "agent->env": 2627

    # Latencies (ms)
    llm_call_ms: Stats | None
    user_sim_ms: Stats | None
    expert_call_ms: Stats | None

    llm_tail_gt_300s: int
    llm_tail_gt_1200s: int
    llm_top: List[Tuple[float, str, str, str]]  # (dur_s, ts, task, step)


def _parse_ts_unix(line: str) -> Optional[int]:
    m = _TS_ANY.search(line)
    if not m:
        return None
    try:
        return int(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp())
    except Exception:
        return None


def _rate(cnt: int, window_s: int) -> float:
    return cnt / window_s if window_s > 0 else 0.0


def analyze_run(run_name: str, run_dir: Path, threshold_pct: float) -> RunBottleneck:
    kv = parse_kv_cache_timeseries(run_dir / "kv_cache_timeseries.csv")
    win_start, win_end = active_window_first_last_above(kv, threshold_pct)
    window_s = win_end - win_start

    log_path = run_dir / "logs" / "tau2_global.log"
    if not log_path.exists():
        raise FileNotFoundError(f"missing {log_path}")

    step_by_edge: Dict[str, int] = {}
    step_total = 0

    llm_call_ms_vals: List[float] = []
    user_sim_ms_vals: List[float] = []
    expert_call_ms_vals: List[float] = []

    llm_tail_gt_300s = 0
    llm_tail_gt_1200s = 0
    llm_top: List[Tuple[float, str, str, str]] = []

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if (
                "type=step_complete" not in line
                and "type=llm_call" not in line
                and "type=user_sim" not in line
                and "type=expert_call" not in line
            ):
                continue

            ts_unix = _parse_ts_unix(line)
            if ts_unix is None or ts_unix < win_start or ts_unix > win_end:
                continue

            if "type=step_complete" in line:
                step_total += 1
                fr = _FROM_ROLE.search(line)
                to = _TO_ROLE.search(line)
                edge = f"{fr.group(1) if fr else 'unknown'}->{to.group(1) if to else 'unknown'}"
                step_by_edge[edge] = step_by_edge.get(edge, 0) + 1
                continue

            dur_m = _DURATION_MS.search(line)
            if not dur_m:
                continue
            dur_ms = float(dur_m.group(1))

            if "type=llm_call" in line:
                llm_call_ms_vals.append(dur_ms)
                dur_s = dur_ms / 1000.0
                if dur_s > 300:
                    llm_tail_gt_300s += 1
                if dur_s > 1200:
                    llm_tail_gt_1200s += 1
                task = (_TASK.search(line).group(1) if _TASK.search(line) else "unknown")
                step = (_STEP.search(line).group(1) if _STEP.search(line) else "NA")
                llm_top.append((dur_s, datetime.fromtimestamp(ts_unix).strftime("%Y-%m-%d %H:%M:%S"), task, step))
                continue

            if "type=user_sim" in line:
                user_sim_ms_vals.append(dur_ms)
                continue

            if "type=expert_call" in line:
                expert_call_ms_vals.append(dur_ms)
                continue

    llm_top.sort(key=lambda x: x[0], reverse=True)
    llm_top = llm_top[:10]

    return RunBottleneck(
        name=run_name,
        run_dir=run_dir,
        win_start=win_start,
        win_end=win_end,
        window_s=window_s,
        step_total=step_total,
        step_by_edge=step_by_edge,
        llm_call_ms=Stats.from_values(llm_call_ms_vals),
        user_sim_ms=Stats.from_values(user_sim_ms_vals),
        expert_call_ms=Stats.from_values(expert_call_ms_vals),
        llm_tail_gt_300s=llm_tail_gt_300s,
        llm_tail_gt_1200s=llm_tail_gt_1200s,
        llm_top=llm_top,
    )


def _fmt_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def _fmt_stats_ms(s: Stats | None) -> str:
    if not s or s.n == 0:
        return "n=0"
    return f"n={s.n} p50={s.p50/1000:.1f}s p90={s.p90/1000:.1f}s p99={s.p99/1000:.1f}s max={s.max/1000:.1f}s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=70.0, help="KV cache threshold in percent (default: 70)")
    ap.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run spec NAME=DIR (repeatable), e.g. --run BL128=outputs/baseline_c128_... ",
    )
    args = ap.parse_args()

    if not args.run:
        raise SystemExit("Must provide at least one --run NAME=DIR")

    runs: List[RunBottleneck] = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"Invalid --run {spec!r}, expected NAME=DIR")
        name, p = spec.split("=", 1)
        runs.append(analyze_run(name.strip(), Path(p).expanduser(), args.threshold))

    print("\n### Active-window bottleneck summary\n")
    print("| Run | window(min) | win_start | win_end | steps/s (all) | agent->env/s | user->agent/s |")
    print("| :-- | ----------: | :------- | :------ | ------------: | -----------: | ------------: |")
    for r in runs:
        agent_env = r.step_by_edge.get("agent->env", 0)
        user_agent = r.step_by_edge.get("user->agent", 0)
        print(
            f"| {r.name} | {r.window_s/60:.1f} | {_fmt_ts(r.win_start)} | {_fmt_ts(r.win_end)} |"
            f" {_rate(r.step_total, r.window_s):.2f} | {_rate(agent_env, r.window_s):.2f} | {_rate(user_agent, r.window_s):.2f} |"
        )

    print("\n### LLM-call tail (within active window)\n")
    print("| Run | llm_call | >300s | >1200s |")  # keep concise
    print("| :-- | :------ | ----: | -----: |")
    for r in runs:
        n = r.llm_call_ms.n if r.llm_call_ms else 0
        gt300 = r.llm_tail_gt_300s
        gt1200 = r.llm_tail_gt_1200s
        print(f"| {r.name} | {n} | {gt300} | {gt1200} |")

    print("\n### Latency distributions (within active window)\n")
    for r in runs:
        print(f"== {r.name} ==")
        print(f"window: {_fmt_ts(r.win_start)} -> {_fmt_ts(r.win_end)} ({r.window_s/60:.1f} min)")
        print(f"llm_call:   {_fmt_stats_ms(r.llm_call_ms)}")
        print(f"user_sim:   {_fmt_stats_ms(r.user_sim_ms)}")
        print(f"expert_call:{_fmt_stats_ms(r.expert_call_ms)}")

        if r.llm_top:
            print("\nTop llm_call durations:")
            for dur_s, ts, task, step in r.llm_top[:5]:
                print(f"  {dur_s:8.1f}s  {ts}  step={step}  task={task}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

