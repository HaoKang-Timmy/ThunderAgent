#!/usr/bin/env python3
"""
Check whether KV cache usage ever exceeds a threshold after a cutoff time.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Allow running as a standalone script (no package install needed).
sys.path.insert(0, os.path.dirname(__file__))

from common import parse_kv_cache_timeseries  # type: ignore


def parse_cutoff(s: str) -> int:
    """
    Accept either:
    - unix seconds (int)
    - "YYYY-MM-DD HH:MM:SS" (local time)
    """
    s = s.strip()
    if s.isdigit():
        return int(s)
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    return int(dt.timestamp())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-csv", type=Path, required=True)
    ap.add_argument("--cutoff", type=str, required=True)
    ap.add_argument("--threshold", type=float, default=70.0)
    args = ap.parse_args()

    cutoff_ts = parse_cutoff(args.cutoff)
    kv = parse_kv_cache_timeseries(args.kv_csv)

    after = [s for s in kv if s.ts_unix > cutoff_ts]
    above = [s for s in after if s.usage_pct > args.threshold]

    print("=" * 70)
    print("KV > threshold AFTER cutoff")
    print("=" * 70)
    print(f"kv_csv: {args.kv_csv}")
    print(f"cutoff: {args.cutoff} (ts={cutoff_ts})")
    print(f"threshold: {args.threshold:.1f}%")
    print(f"samples_after_cutoff: {len(after)}")
    print(f"samples_above_threshold: {len(above)}")

    if above:
        last = max(above, key=lambda s: s.ts_unix)
        print(f"last_above_threshold: {datetime.fromtimestamp(last.ts_unix)}  {last.usage_pct:.2f}%")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


