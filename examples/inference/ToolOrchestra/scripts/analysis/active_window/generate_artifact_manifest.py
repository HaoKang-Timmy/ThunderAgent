#!/usr/bin/env python3
"""
Generate a markdown "artifact manifest" for a completed experiment run.

This is intended for open-sourcing: pin exact run directories, key files,
hashes, and the exact throughput numbers under the chosen active-window rule.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Allow running as a standalone script (no package install needed).
import sys

sys.path.insert(0, os.path.dirname(__file__))

from common import (  # type: ignore
    active_window_first_last_above,
    parse_kv_cache_timeseries,
    parse_step_done_timestamps,
    parse_task_done_timestamps,
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_stats(p: Path) -> tuple[int, int, str]:
    """Return (bytes, lines, sha256). Lines counts newline-terminated lines; may be 0 for 1-line files."""
    b = p.stat().st_size
    try:
        lines = sum(1 for _ in p.open("r", errors="ignore"))
    except Exception:
        lines = -1
    return b, lines, sha256_file(p)


def grep_last(pattern: str, p: Path) -> str:
    try:
        out = subprocess.run(
            ["grep", "-E", pattern, str(p)],
            capture_output=True,
            text=True,
            errors="ignore",
        ).stdout.splitlines()
        return out[-1] if out else ""
    except Exception:
        return ""


def count_domain_json(run_dir: Path, domain: str) -> int:
    # tau2-bench writes per-task results under outputs/all_domains/
    d = run_dir / "outputs" / "all_domains"
    if not d.exists():
        return 0
    return len(list(d.glob(f"{domain}__*__trial0.json")))


@dataclass(frozen=True)
class RunRow:
    scenario: str
    scheduler: str
    concurrency: int
    run_dir: Path
    console_log: Optional[Path]
    driver_log: Path
    kv_csv: Path
    scheduler_timestamps: Optional[Path]
    model_config: Optional[Path]
    router_backends: Optional[Path]
    router_programs: Optional[Path]
    global_summary_line: str
    tasks_ok: Optional[int]
    tasks_error: Optional[int]
    tasks_total: Optional[int]
    steps_done: int
    tasks_airline: int
    tasks_retail: int
    tasks_telecom: int
    win_start: int
    win_end: int
    win_minutes: float
    tasks_in_window: int
    steps_in_window: int
    tasks_per_min: float
    steps_per_sec: float


_GLOBAL_RE = re.compile(r"Global evaluation complete: ok=(\d+) error=(\d+) total=(\d+)")


def parse_global_counts(line: str) -> tuple[Optional[int], Optional[int], Optional[int]]:
    m = _GLOBAL_RE.search(line)
    if not m:
        return None, None, None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def compute_run_row(
    scenario: str,
    scheduler: str,
    concurrency: int,
    run_dir: Path,
    threshold: float,
    console_log: Optional[Path],
) -> RunRow:
    driver_log = run_dir / "driver.log"
    kv_csv = run_dir / "kv_cache_timeseries.csv"

    kv = parse_kv_cache_timeseries(kv_csv)
    win_start, win_end = active_window_first_last_above(kv, threshold)
    win_s = win_end - win_start

    tasks_done_ts = parse_task_done_timestamps(driver_log)
    steps_done_ts = parse_step_done_timestamps(driver_log)

    tasks_in_window = sum(1 for t in tasks_done_ts if win_start <= t <= win_end)
    steps_in_window = sum(1 for s in steps_done_ts if win_start <= s <= win_end)

    summary = grep_last(r"Global evaluation complete:", driver_log)
    ok, err, tot = parse_global_counts(summary)

    # domain json counts (in all_domains)
    tasks_airline = count_domain_json(run_dir, "airline")
    tasks_retail = count_domain_json(run_dir, "retail")
    tasks_telecom = count_domain_json(run_dir, "telecom")

    # optional files
    st = run_dir / "scheduler_timestamps"
    scheduler_ts = st if st.exists() else None

    mc_tr = run_dir / "model_config_5090_thunderreact.json"
    mc_ct = run_dir / "model_config_5090_continuum.json"
    model_config = mc_tr if mc_tr.exists() else (mc_ct if mc_ct.exists() else None)

    rb = run_dir / "router_backends.json"
    rp = run_dir / "router_programs.json"
    router_backends = rb if rb.exists() else None
    router_programs = rp if rp.exists() else None

    return RunRow(
        scenario=scenario,
        scheduler=scheduler,
        concurrency=concurrency,
        run_dir=run_dir,
        console_log=console_log if (console_log and console_log.exists()) else None,
        driver_log=driver_log,
        kv_csv=kv_csv,
        scheduler_timestamps=scheduler_ts,
        model_config=model_config,
        router_backends=router_backends,
        router_programs=router_programs,
        global_summary_line=summary,
        tasks_ok=ok,
        tasks_error=err,
        tasks_total=tot,
        steps_done=len(steps_done_ts),
        tasks_airline=tasks_airline,
        tasks_retail=tasks_retail,
        tasks_telecom=tasks_telecom,
        win_start=win_start,
        win_end=win_end,
        win_minutes=win_s / 60.0,
        tasks_in_window=tasks_in_window,
        steps_in_window=steps_in_window,
        tasks_per_min=tasks_in_window / (win_s / 60.0),
        steps_per_sec=steps_in_window / win_s,
    )


def try_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, errors="ignore").stdout.strip()
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=70.0)
    ap.add_argument("--experiment-id", type=str, required=True)
    ap.add_argument("--output-base", type=Path, required=True)

    ap.add_argument("--tr32", type=Path, required=True)
    ap.add_argument("--tr64", type=Path, required=True)
    ap.add_argument("--tr128", type=Path, required=True)
    ap.add_argument("--ct32", type=Path, required=True)
    ap.add_argument("--ct64", type=Path, required=True)
    ap.add_argument("--ct128", type=Path, required=True)

    ap.add_argument("--tr32-console", type=Path, default=None)
    ap.add_argument("--tr64-console", type=Path, default=None)
    ap.add_argument("--tr128-console", type=Path, default=None)
    ap.add_argument("--ct32-console", type=Path, default=None)
    ap.add_argument("--ct64-console", type=Path, default=None)
    ap.add_argument("--ct128-console", type=Path, default=None)

    args = ap.parse_args()

    git_head = try_cmd(["git", "rev-parse", "HEAD"])
    dirty_count = try_cmd(["bash", "-lc", "git status --porcelain | wc -l"]).strip()
    sys_uname = try_cmd(["uname", "-a"])
    nvsmi = try_cmd(["bash", "-lc", "nvidia-smi | head -n 3"])  # best-effort

    rows = [
        compute_run_row("C=32", "ThunderReact", 32, args.tr32, args.threshold, args.tr32_console),
        compute_run_row("C=64", "ThunderReact", 64, args.tr64, args.threshold, args.tr64_console),
        compute_run_row("C=128", "ThunderReact", 128, args.tr128, args.threshold, args.tr128_console),
        compute_run_row("C=32", "Continuum", 32, args.ct32, args.threshold, args.ct32_console),
        compute_run_row("C=64", "Continuum", 64, args.ct64, args.threshold, args.ct64_console),
        compute_run_row("C=128", "Continuum", 128, args.ct128, args.threshold, args.ct128_console),
    ]

    print("### Artifact manifest\n")

    print("#### Repo + system\n")
    print("| Item | Value |")
    print("| :-- | :-- |")
    print(f"| experiment_id | `{args.experiment_id}` |")
    print(f"| output_base | `{args.output_base}` |")
    if git_head:
        print(f"| git_head | `{git_head}` |")
        print(f"| git_dirty_files | `{dirty_count}` |")
    if sys_uname:
        print(f"| uname | `{sys_uname}` |")
    if nvsmi:
        print(f"| nvidia_smi_head | `{nvsmi.replace('`', '')}` |")

    print("\n#### Runs (paths + active-window throughput)\n")
    print("| Scenario | Scheduler | Run dir | Console log | Global summary (tail) | Active window (start→end) | tasks/min | steps/s |")
    print("| :-- | :-- | :-- | :-- | :-- | :-- | --: | --: |")
    for r in rows:
        start = datetime.fromtimestamp(r.win_start).strftime("%Y-%m-%d %H:%M:%S")
        end = datetime.fromtimestamp(r.win_end).strftime("%Y-%m-%d %H:%M:%S")
        console = f"`{r.console_log}`" if r.console_log else ""
        summary = r.global_summary_line.replace("|", "\\|") if r.global_summary_line else ""
        print(
            f"| {r.scenario} | {r.scheduler} | `{r.run_dir}` | {console} | {summary} | `{start}` → `{end}` | {r.tasks_per_min:.2f} | {r.steps_per_sec:.2f} |"
        )

    print("\n#### Task result files (from `outputs/all_domains/`)\n")
    print("| Scenario | Scheduler | retail | telecom | airline | total |")
    print("| :-- | :-- | --: | --: | --: | --: |")
    for r in rows:
        total = r.tasks_retail + r.tasks_telecom + r.tasks_airline
        print(f"| {r.scenario} | {r.scheduler} | {r.tasks_retail} | {r.tasks_telecom} | {r.tasks_airline} | {total} |")

    print("\n#### Key file hashes (sha256)\n")
    print("| Scenario | Scheduler | File | Bytes | Lines | sha256 |")
    print("| :-- | :-- | :-- | --: | --: | :-- |")
    for r in rows:
        key_files = [r.driver_log, r.kv_csv]
        if r.scheduler_timestamps:
            key_files.append(r.scheduler_timestamps)
        if r.model_config:
            key_files.append(r.model_config)
        if r.router_backends:
            key_files.append(r.router_backends)
        if r.router_programs:
            key_files.append(r.router_programs)

        for p in key_files:
            b, ln, sha = file_stats(Path(p))
            print(f"| {r.scenario} | {r.scheduler} | `{Path(p).name}` | {b} | {ln} | `{sha}` |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


