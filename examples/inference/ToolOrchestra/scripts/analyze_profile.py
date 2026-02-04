#!/usr/bin/env python3
"""Analyze PROFILE logs for latency statistics.

This is a lightweight parser for tau2-bench logs that contain "[PROFILE]" lines.
"""

import json
import re
import statistics
import sys
from collections import defaultdict
from typing import Dict, List


def _p(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    # Nearest-rank method
    idx = min(n - 1, max(0, int(n * q)))
    return sorted_vals[idx]


def parse_profile(log_path: str) -> Dict[str, Dict[str, float]]:
    events: Dict[str, List[float]] = defaultdict(list)
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "[PROFILE]" not in line:
                continue
            # key=value pairs
            matches = dict(re.findall(r"(\w+)=([^\s]+)", line))
            # event type
            event_type = matches.get("type") or matches.get("event") or "unknown"

            # duration field (support multiple names used in codebase)
            dur_s = (
                matches.get("duration_ms")
                or matches.get("total_duration_ms")
                or matches.get("infer_ms")
            )
            if dur_s is None:
                continue
            try:
                events[event_type].append(float(dur_s))
            except Exception:
                continue

    stats: Dict[str, Dict[str, float]] = {}
    for event_type, durations in events.items():
        if not durations:
            continue
        sorted_d = sorted(durations)
        stats[event_type] = {
            "count": float(len(sorted_d)),
            "mean_ms": float(statistics.mean(sorted_d)),
            "median_ms": float(statistics.median(sorted_d)),
            "p90_ms": float(_p(sorted_d, 0.90)),
            "p95_ms": float(_p(sorted_d, 0.95)),
            "p99_ms": float(_p(sorted_d, 0.99)),
        }
    return stats


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: analyze_profile.py <eval.log>", file=sys.stderr)
        return 2
    print(json.dumps(parse_profile(sys.argv[1]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


