#!/usr/bin/env python3
"""
Analyze TR-new run logs to detect potential router/backend stalls.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TS_UNIX_RE = re.compile(r"\bts_unix=(?P<ts>\d+(?:\.\d+)?)\b")
TS_ISO_RE = re.compile(r"\bts_iso=(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")
PROFILE_TS_RE = re.compile(r"\[PROFILE\]\s+(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:\.\d+)?")

CALL_RE = re.compile(
    r"Calling vLLM at http://(?P<host>[^/]+)/[^ ]+ .* req_id=(?P<id>[\w-]+)"
)
COMPLETE_RE = re.compile(r"vLLM (?:stream complete|request successful).* req_id=(?P<id>[\w-]+)")

ORCH_TS_RE = re.compile(r"\b(?P<md>\d{2}-\d{2}) (?P<hms>\d{2}:\d{2}:\d{2})\b")


def _parse_ts_unix(line: str) -> Optional[float]:
    m = TS_UNIX_RE.search(line)
    if m:
        try:
            return float(m.group("ts"))
        except Exception:
            return None
    m = TS_ISO_RE.search(line)
    if m:
        try:
            return dt.datetime.fromisoformat(m.group("ts")).timestamp()
        except Exception:
            return None
    m = PROFILE_TS_RE.search(line)
    if m:
        try:
            return dt.datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S").timestamp()
        except Exception:
            return None
    return None


def _parse_eval_log(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "trial_ts": [],
        "step_ts": [],
        "tool_ts": [],
        "llm_ts": [],
        "call_events": [],
        "complete_events": [],
        "errors": [],
        "last_ts": None,
    }
    last_ts: Optional[float] = None
    for idx, line in enumerate(path.read_text(errors="ignore").splitlines()):
        ts = _parse_ts_unix(line)
        if ts is not None:
            last_ts = ts
            data["last_ts"] = ts
        if "type=tool_call" in line:
            if ts is not None:
                data["tool_ts"].append(ts)
        if "type=step_complete" in line:
            if ts is not None:
                data["step_ts"].append(ts)
        if "type=llm_call" in line:
            if ts is not None:
                data["llm_ts"].append(ts)
        if "[HLE_TRIAL_COMPLETE]" in line:
            if ts is not None:
                data["trial_ts"].append(ts)

        m = CALL_RE.search(line)
        if m:
            data["call_events"].append(
                {
                    "req_id": m.group("id"),
                    "host": m.group("host"),
                    "ts": last_ts,
                    "line": idx + 1,
                }
            )

        m = COMPLETE_RE.search(line)
        if m:
            data["complete_events"].append({"req_id": m.group("id"), "ts": last_ts, "line": idx + 1})

        if "ERROR" in line or "LLM_ERROR" in line or "Exception" in line or "timeout" in line:
            data["errors"].append(line.strip())

    return data


def _parse_hle_profile(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {"tool_ts": [], "step_ts": [], "llm_ts": []}
    for line in path.read_text(errors="ignore").splitlines():
        ts = _parse_ts_unix(line)
        if ts is None:
            continue
        if "type=tool_call" in line:
            data["tool_ts"].append(ts)
        if "type=step_complete" in line:
            data["step_ts"].append(ts)
        if "type=llm_call" in line:
            data["llm_ts"].append(ts)
    return data


def _parse_orchestrator_log(path: Path, year: int) -> Dict[str, Any]:
    data: Dict[str, Any] = {"last_req_ts": None, "last_engine_ts": None}
    for line in path.read_text(errors="ignore").splitlines():
        if "POST /v1/chat/completions" not in line and "Engine 000" not in line:
            continue
        m = ORCH_TS_RE.search(line)
        if not m:
            continue
        try:
            ts = dt.datetime.strptime(f"{year}-{m.group('md')} {m.group('hms')}", "%Y-%m-%d %H:%M:%S").timestamp()
        except Exception:
            continue
        if "POST /v1/chat/completions" in line:
            data["last_req_ts"] = ts
        if "Engine 000" in line:
            data["last_engine_ts"] = ts
    return data


def _parse_router_log(path: Path) -> Dict[str, Any]:
    text = path.read_text(errors="ignore").splitlines()
    return {
        "paused": sum(1 for l in text if "Paused program" in l),
        "resumed": sum(1 for l in text if "Resumed program" in l),
        "released": sum(1 for l in text if "Released program" in l),
        "must_wait": sum(1 for l in text if "must wait" in l),
        "wait_timeout": sum(1 for l in text if "wait timeout" in l),
        "router_stopped": any("Router stopped" in l for l in text),
        "tail": text[-10:],
    }


def _fmt_ts(ts: Optional[float]) -> str:
    if ts is None:
        return "n/a"
    return dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--router-port", type=int, default=8000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    logs_dir = out_dir / "logs"
    eval_log = logs_dir / "eval_driver.log"
    profile_log = logs_dir / "hle_profile.log"
    router_log = logs_dir / "tr_router.log"
    orch_log = logs_dir / "orchestrator.log"

    if not eval_log.exists():
        raise SystemExit(f"missing {eval_log}")

    eval_data = _parse_eval_log(eval_log)
    profile_data = _parse_hle_profile(profile_log) if profile_log.exists() else {}

    year = dt.datetime.now().year
    if eval_data.get("trial_ts"):
        year = dt.datetime.fromtimestamp(max(eval_data["trial_ts"])).year

    orch_data = _parse_orchestrator_log(orch_log, year) if orch_log.exists() else {}
    router_data = _parse_router_log(router_log) if router_log.exists() else {}

    router_host = f"127.0.0.1:{args.router_port}"
    router_calls = [e for e in eval_data["call_events"] if e.get("host") == router_host]
    call_ids = [e["req_id"] for e in router_calls]
    complete_ids = {e["req_id"] for e in eval_data["complete_events"] if e["req_id"] in call_ids}
    pending = [e for e in router_calls if e["req_id"] not in complete_ids]

    last_complete_ts = None
    if eval_data["complete_events"]:
        last_complete_ts = max(e["ts"] for e in eval_data["complete_events"] if e["ts"] is not None)

    last_call_ts = None
    if router_calls:
        last_call_ts = max(e["ts"] for e in router_calls if e["ts"] is not None)

    print("== TR Run Stall Analysis ==")
    print(f"out_dir: {out_dir}")
    print(f"eval_log: {eval_log}")
    print("")
    print(f"trials: {len(eval_data['trial_ts'])} last={_fmt_ts(max(eval_data['trial_ts']) if eval_data['trial_ts'] else None)}")
    print(f"steps: {len(eval_data['step_ts'])} last={_fmt_ts(max(eval_data['step_ts']) if eval_data['step_ts'] else None)}")
    print(f"llm_calls: {len(eval_data['llm_ts'])} last={_fmt_ts(max(eval_data['llm_ts']) if eval_data['llm_ts'] else None)}")
    print("")
    print(f"orchestrator_calls_to_router: {len(call_ids)}")
    print(f"router_completions_seen: {len(complete_ids)}")
    print(f"pending_calls: {len(pending)}")
    print(f"last_call_ts: {_fmt_ts(last_call_ts)}")
    print(f"last_completion_ts: {_fmt_ts(last_complete_ts)}")
    if last_call_ts and last_complete_ts and last_call_ts > last_complete_ts:
        print(f"gap_last_call_minus_completion_sec: {last_call_ts - last_complete_ts:.1f}")
    print("")
    if orch_data:
        print(f"orchestrator.log last /v1/chat/completions: {_fmt_ts(orch_data.get('last_req_ts'))}")
        print(f"orchestrator.log last Engine log: {_fmt_ts(orch_data.get('last_engine_ts'))}")
    else:
        print("orchestrator.log: n/a")
    print("")
    if profile_data:
        print(f"profile last tool_call: {_fmt_ts(max(profile_data.get('tool_ts', []) or [None]))}")
        print(f"profile last step_complete: {_fmt_ts(max(profile_data.get('step_ts', []) or [None]))}")
        print(f"profile last llm_call: {_fmt_ts(max(profile_data.get('llm_ts', []) or [None]))}")
    print("")
    if router_data:
        print(f"router paused/resumed/released: {router_data.get('paused')} / {router_data.get('resumed')} / {router_data.get('released')}")
        print(f"router must_wait: {router_data.get('must_wait')} wait_timeout: {router_data.get('wait_timeout')}")
        print(f"router stopped: {router_data.get('router_stopped')}")
        print("router tail:")
        for line in router_data.get("tail", []):
            print(f"  {line}")
    print("")
    if eval_data["errors"]:
        print("errors in eval_driver.log:")
        for line in eval_data["errors"][-10:]:
            print(f"  {line}")

    if pending:
        last_pending = max((p["ts"] for p in pending if p["ts"] is not None), default=None)
        print("")
        print("pending summary:")
        print(f"  last_pending_ts: {_fmt_ts(last_pending)}")
        print(f"  pending_req_ids_tail: {[p['req_id'] for p in pending[-5:]]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
