from __future__ import annotations

import csv
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple


KV_METRIC_SUBSTR = "kv_cache_usage_perc"


@dataclass(frozen=True)
class KvSample:
    ts_unix: int
    usage_pct: float  # 0..100


def parse_kv_cache_timeseries(csv_path: str | Path) -> List[KvSample]:
    """Parse `kv_cache_timeseries.csv` and return samples for vllm:kv_cache_usage_perc."""
    csv_path = Path(csv_path)
    out: List[KvSample] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return out
        # expected: ts_iso,ts_unix,port,metric,value
        for row in reader:
            if len(row) < 5:
                continue
            metric = row[3]
            if KV_METRIC_SUBSTR not in metric:
                continue
            try:
                ts_unix = int(row[1])
                value = float(row[4])  # fraction [0,1]
                out.append(KvSample(ts_unix=ts_unix, usage_pct=value * 100.0))
            except Exception:
                continue
    out.sort(key=lambda s: s.ts_unix)
    return out


# `driver.log` occasionally has concatenated lines where `FINISHED SIMULATION`
# appears after some `DEBUG:` text without a leading timestamp. In those cases,
# the INFO timestamp still exists later in the same line, so we search anywhere.
_TS_ANYWHERE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_STEP_TS = re.compile(r"\[PROFILE\] (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*type=step_complete")


def _grep_lines(pattern: str, file_path: str | Path) -> Iterable[str]:
    """Fast line extraction using grep (driver.log can be big)."""
    p = subprocess.run(
        ["grep", "-E", pattern, str(file_path)],
        capture_output=True,
        text=True,
        errors="ignore",
    )
    for line in p.stdout.splitlines():
        if line:
            yield line


def parse_task_done_timestamps(driver_log_path: str | Path) -> List[int]:
    """
    Parse task completion timestamps.

    Definition used: each `FINISHED SIMULATION:` line corresponds to one completed task.
    """
    out: List[int] = []
    for line in _grep_lines("FINISHED SIMULATION", driver_log_path):
        m = _TS_ANYWHERE.search(line)
        if not m:
            continue
        try:
            dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            out.append(int(dt.timestamp()))
        except Exception:
            continue
    out.sort()
    return out


def parse_step_done_timestamps(driver_log_path: str | Path) -> List[int]:
    """Parse step completion timestamps from PROFILE lines with `type=step_complete`."""
    out: List[int] = []
    for line in _grep_lines("type=step_complete", driver_log_path):
        m = _STEP_TS.search(line)
        if not m:
            continue
        try:
            dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            out.append(int(dt.timestamp()))
        except Exception:
            continue
    out.sort()
    return out


def active_window_first_last_above(kv: List[KvSample], threshold_pct: float) -> Tuple[int, int]:
    """
    Active window definition:
    - start = first time kv_usage_pct > threshold_pct
    - end   = last  time kv_usage_pct > threshold_pct
    """
    first = None
    last = None
    for s in kv:
        if s.usage_pct > threshold_pct:
            if first is None:
                first = s.ts_unix
            last = s.ts_unix
    if first is None or last is None:
        raise ValueError(f"No samples above threshold {threshold_pct}%")
    if last <= first:
        raise ValueError("Invalid active window (end <= start)")
    return first, last


