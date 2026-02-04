#!/usr/bin/env python3
"""
Watch 5090 HLE serving outputs for a scheduler and append windowed metrics to markdown tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x != x:  # NaN
            return "n/a"
        return f"{x:.6f}"
    return str(x)


def _ensure_report(report_md: Path, scheduler: str) -> None:
    if report_md.exists():
        return
    report_md.parent.mkdir(parents=True, exist_ok=True)
    title_map = {
        "baseline": "HLE Serving: Baseline vLLM",
        "trnew": "HLE Serving: ThunderReact-new (TR-new)",
        "continuum": "HLE Serving: vLLM-Continuum",
    }
    title = title_map.get(scheduler, f"HLE Serving: {scheduler}")
    report_md.write_text(
        "\n".join(
            [
                f"# {title} (10–130min window)",
                "",
                "- `t0` = first orchestrator response timestamp in `orchestrator_usage.jsonl`.",
                "- Window = `[t0+10min, t0+130min]` = `[600s, 7800s]` after `t0`.",
                "- Rows are appended in completion order (based on `window_summary.json` mtime).",
                "",
                "## Throughput",
                "| C | rep | trials/min | steps/sec |",
                "| --- | --- | --- | --- |",
                "",
                "## Cache & Utilization",
                "| C | rep | server_hit_ratio | request_hit_avg | request_hit_token_weighted | kv_usage_mean_perc | gpu_sm_util_mean |",
                "| --- | --- | --- | --- | --- | --- | --- |",
                "",
                "## Latency",
                "| C | rep | decode_s | e2e_s | decode_over_e2e |",
                "| --- | --- | --- | --- | --- |",
                "",
                "## Preemptions",
                "| C | rep | delta_preemptions_mean_2s | preemptions_total_window | out_dir |",
                "| --- | --- | --- | --- | --- |",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _state_path(report_md: Path) -> Path:
    return report_md.with_suffix(report_md.suffix + ".state.json")


def _load_state(report_md: Path) -> Dict[str, Any]:
    p = _state_path(report_md)
    if not p.exists():
        return {"processed_out_dirs": []}
    try:
        obj = json.loads(_read_text(p))
        if isinstance(obj, dict):
            if "processed_out_dirs" not in obj or not isinstance(obj["processed_out_dirs"], list):
                obj["processed_out_dirs"] = []
            return obj
    except Exception:
        pass
    return {"processed_out_dirs": []}


def _save_state(report_md: Path, state: Dict[str, Any]) -> None:
    p = _state_path(report_md)
    p.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _gpu_sm_mean(gpu_csv: Path, t_start: float, t_end: float) -> Optional[float]:
    if not gpu_csv.exists():
        return None
    sm_sum = 0.0
    sm_n = 0
    try:
        with gpu_csv.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    ts = float(row.get("ts_unix", "nan"))
                    sm = float(row.get("sm_util", "nan"))
                except Exception:
                    continue
                if ts != ts or sm != sm:
                    continue
                if ts < t_start or ts > t_end:
                    continue
                sm_sum += sm
                sm_n += 1
    except Exception:
        return None
    return (sm_sum / sm_n) if sm_n > 0 else None


def _parse_c_rep(out_dir_name: str) -> Tuple[int, int]:
    m = re.search(r"_c(?P<c>\d+)_rep(?P<rep>\d+)_", out_dir_name)
    if not m:
        return 0, 0
    return int(m.group("c")), int(m.group("rep"))


def _extract_one(
    *,
    out_dir: Path,
    window_start_sec: float,
    window_end_sec: float,
) -> Optional[Dict[str, Any]]:
    # Guard against polluted runs where the TR router failed to start and eval traffic
    # hit a stale router (port already in use).
    tr_router_log = out_dir / "logs" / "tr_router.log"
    if tr_router_log.exists():
        txt = _read_text(tr_router_log)
        if "Errno 98" in txt or "address already in use" in txt:
            return None

    window_summary = out_dir / "window_summary.json"
    steps_summary = out_dir / "steps_summary.json"
    gpu_csv = out_dir / "gpu_sm_util_timeseries.csv"

    if not window_summary.exists():
        return None
    try:
        ws = _load_json(window_summary)
    except Exception:
        return None
    if not isinstance(ws, dict) or ws.get("ok") is not True:
        return None
    key = f"{int(window_start_sec)}-{int(window_end_sec)}"
    win = (ws.get("windows") or {}).get(key) or {}
    if not isinstance(win, dict):
        return None

    metrics = win.get("metrics") or {}
    if isinstance(metrics, dict) and metrics.get("available") is False:
        return None
    usage = win.get("response_usage") or {}
    trials = win.get("trials") or {}

    pre = (metrics.get("preemptions") or {}) if isinstance(metrics, dict) else {}
    lat = (metrics.get("latency_means_seconds") or {}) if isinstance(metrics, dict) else {}

    t0_unix = ws.get("t0_first_request_unix")
    if not isinstance(t0_unix, (int, float)):
        return None
    t_start = float(t0_unix) + float(window_start_sec)
    t_end = float(t0_unix) + float(window_end_sec)

    steps_per_sec = None
    if steps_summary.exists():
        try:
            ss = _load_json(steps_summary)
            steps_block = ss.get("steps") if isinstance(ss, dict) else None
            if isinstance(steps_block, dict):
                steps_per_sec = steps_block.get("steps_per_sec")
        except Exception:
            steps_per_sec = None

    c_val, rep_val = _parse_c_rep(out_dir.name)

    trials_per_min = None
    if isinstance(trials, dict) and isinstance(trials.get("trials_per_sec"), (int, float)):
        trials_per_min = float(trials.get("trials_per_sec")) * 60.0

    return {
        "C": c_val,
        "rep": rep_val,
        "trials_per_min": trials_per_min,
        "steps_per_sec": steps_per_sec,
        "server_hit_ratio": metrics.get("hit_ratio") if isinstance(metrics, dict) else None,
        "request_hit_avg": usage.get("cached_tokens_ratio_request_avg") if isinstance(usage, dict) else None,
        "request_hit_token_weighted": usage.get("cached_tokens_ratio_token_weighted") if isinstance(usage, dict) else None,
        "kv_usage_mean_perc": metrics.get("kv_cache_usage_mean_perc") if isinstance(metrics, dict) else None,
        "gpu_sm_util_mean": _gpu_sm_mean(gpu_csv, t_start, t_end),
        "decode_s": lat.get("request_decode_time_seconds") if isinstance(lat, dict) else None,
        "e2e_s": lat.get("e2e_request_latency_seconds") if isinstance(lat, dict) else None,
        "decode_over_e2e": metrics.get("decode_over_e2e_ratio") if isinstance(metrics, dict) else None,
        "delta_preemptions_mean_2s": (pre.get("delta_preemptions_mean_per_interval") if isinstance(pre, dict) else None),
        "preemptions_total_window": (pre.get("sum_delta_preemptions") if isinstance(pre, dict) else None),
        "out_dir": str(out_dir),
    }


def _insert_table_row(lines: List[str], *, header: str, row_line: str) -> List[str]:
    header_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == header:
            header_idx = i
            break
    if header_idx < 0:
        return lines + [row_line]

    start = header_idx + 2
    end = start
    while end < len(lines) and lines[end].lstrip().startswith("|"):
        end += 1
    return lines[:end] + [row_line] + lines[end:]


def _append_rows(*, report_md: Path, row: Dict[str, Any]) -> None:
    def rel(p: Optional[str]) -> str:
        if not p:
            return "n/a"
        try:
            rp = Path(p).resolve()
            return str(rp.relative_to(Path("/workspace")))
        except Exception:
            return p

    thr_header = "| C | rep | trials/min | steps/sec |"
    cache_header = "| C | rep | server_hit_ratio | request_hit_avg | request_hit_token_weighted | kv_usage_mean_perc | gpu_sm_util_mean |"
    lat_header = "| C | rep | decode_s | e2e_s | decode_over_e2e |"
    pre_header = "| C | rep | delta_preemptions_mean_2s | preemptions_total_window | out_dir |"

    thr_row = (
        "| "
        + " | ".join(
            [
                _fmt(row.get("C")),
                _fmt(row.get("rep")),
                _fmt(row.get("trials_per_min")),
                _fmt(row.get("steps_per_sec")),
            ]
        )
        + " |\n"
    )

    cache_row = (
        "| "
        + " | ".join(
            [
                _fmt(row.get("C")),
                _fmt(row.get("rep")),
                _fmt(row.get("server_hit_ratio")),
                _fmt(row.get("request_hit_avg")),
                _fmt(row.get("request_hit_token_weighted")),
                _fmt(row.get("kv_usage_mean_perc")),
                _fmt(row.get("gpu_sm_util_mean")),
            ]
        )
        + " |\n"
    )

    lat_row = (
        "| "
        + " | ".join(
            [
                _fmt(row.get("C")),
                _fmt(row.get("rep")),
                _fmt(row.get("decode_s")),
                _fmt(row.get("e2e_s")),
                _fmt(row.get("decode_over_e2e")),
            ]
        )
        + " |\n"
    )

    pre_row = (
        "| "
        + " | ".join(
            [
                _fmt(row.get("C")),
                _fmt(row.get("rep")),
                _fmt(row.get("delta_preemptions_mean_2s")),
                _fmt(row.get("preemptions_total_window")),
                f"`{rel(row.get('out_dir'))}`",
            ]
        )
        + " |\n"
    )

    lines = report_md.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    for header, row_line in [
        (thr_header, thr_row),
        (cache_header, cache_row),
        (lat_header, lat_row),
        (pre_header, pre_row),
    ]:
        lines = _insert_table_row(lines, header=header, row_line=row_line)
    report_md.write_text("".join(lines), encoding="utf-8")


def _out_dir_mtime(out_dir: Path) -> float:
    ws = out_dir / "window_summary.json"
    if ws.exists():
        try:
            return ws.stat().st_mtime
        except Exception:
            return time.time()
    try:
        return out_dir.stat().st_mtime
    except Exception:
        return time.time()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheduler", required=True, help="baseline | trnew | continuum")
    ap.add_argument("--outputs-dir", default=None, help="Path to outputs/ directory")
    ap.add_argument("--report-md", default=None, help="Path to output markdown report")
    ap.add_argument("--interval-sec", type=float, default=30.0)
    ap.add_argument(
        "--since-unix",
        type=float,
        default=0.0,
        help="Only process out_dirs whose mtime is >= this unix timestamp (prevents older runs from being appended).",
    )
    ap.add_argument("--window-start-sec", type=float, default=600.0)
    ap.add_argument("--window-end-sec", type=float, default=7800.0)
    ap.add_argument("--once", action="store_true", help="Run one scan then exit")
    args = ap.parse_args()

    script_root = Path(__file__).resolve().parents[3]
    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else (script_root / "outputs")
    report_md = (
        Path(args.report_md)
        if args.report_md
        else Path("/workspace/hle-serving-5090") / f"hle_serving_{args.scheduler}_10_130.md"
    )

    _ensure_report(report_md, args.scheduler)
    state = _load_state(report_md)
    processed: set[str] = set(state.get("processed_out_dirs") or [])

    while True:
        out_dirs = sorted(
            outputs_dir.glob(f"hle_serving_5090_{args.scheduler}_c*_rep*_*/"),
            key=_out_dir_mtime,
        )
        scanned_total = len(out_dirs)
        eligible = 0
        skipped_old = 0
        skipped_processed = 0
        pending = 0
        appended = 0
        any_new = False
        for out_dir in out_dirs:
            if _out_dir_mtime(out_dir) < float(args.since_unix):
                skipped_old += 1
                continue
            eligible += 1
            out_dir_s = str(out_dir.resolve())
            if out_dir_s in processed:
                skipped_processed += 1
                continue
            extracted = _extract_one(
                out_dir=out_dir,
                window_start_sec=float(args.window_start_sec),
                window_end_sec=float(args.window_end_sec),
            )
            if extracted is None:
                pending += 1
                continue
            _append_rows(report_md=report_md, row=extracted)
            processed.add(out_dir_s)
            any_new = True
            appended += 1

        if any_new:
            state["processed_out_dirs"] = sorted(processed)
            _save_state(report_md, state)

        print(
            f"[watch] scheduler={args.scheduler} scanned_total={scanned_total} eligible={eligible} "
            f"appended={appended} pending={pending} skipped_old={skipped_old} skipped_processed={skipped_processed} "
            f"state_count={len(processed)} report={report_md}",
            flush=True,
        )

        if args.once:
            break
        time.sleep(float(args.interval_sec))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
