#!/usr/bin/env python3
import argparse
import os
import re
import time
from pathlib import Path


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


def _get_ppid(pid: int) -> int | None:
    status = Path(f"/proc/{pid}/status")
    try:
        for line in status.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("PPid:"):
                return int(line.split(":", 1)[1].strip())
    except Exception:
        return None
    return None


def _get_cmdline(pid: int) -> str:
    cmdline = Path(f"/proc/{pid}/cmdline")
    try:
        raw = cmdline.read_bytes()
    except Exception:
        return ""
    parts = [p.decode(errors="ignore") for p in raw.split(b"\x00") if p]
    return " ".join(parts)


def _find_ancestor_cmd(pid: int, needle: str, *, max_hops: int = 30) -> str:
    cur = pid
    for _ in range(max_hops):
        cmd = _get_cmdline(cur)
        if needle in cmd:
            return cmd
        ppid = _get_ppid(cur)
        if not ppid or ppid <= 1:
            break
        cur = ppid
    return ""


def _find_open_fd_path(pid: int, *, suffix: str) -> str | None:
    fd_dir = Path(f"/proc/{pid}/fd")
    try:
        for entry in fd_dir.iterdir():
            try:
                target = os.readlink(entry)
            except Exception:
                continue
            if target.endswith(suffix):
                return target
    except Exception:
        return None
    return None


def _lsof_listen_pids(port: int) -> list[int]:
    import subprocess

    try:
        res = subprocess.run(
            ["lsof", f"-tiTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return []
    if res.returncode != 0:
        return []
    pids: list[int] = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line))
        except Exception:
            continue
    return sorted(set(pids))


RUN_CMD_RE = re.compile(r"run_hle_serving_one_w130\.sh\s+trnew\s+(?P<c>\d+)\s+(?P<rep>\d+)")


def _infer_task_from_pid(pid: int) -> tuple[int | None, int | None, str]:
    ancestor = _find_ancestor_cmd(pid, "run_hle_serving_one_w130.sh")
    m = RUN_CMD_RE.search(ancestor)
    if not m:
        return None, None, ancestor
    return int(m.group("c")), int(m.group("rep")), ancestor


def _latest_parallel_dir(outputs_dir: Path) -> Path | None:
    cands = list(outputs_dir.glob("hle_serving_5090_trnew_parallel_*"))
    if not cands:
        return None
    # Prefer name-based sorting (timestamp is embedded in the directory name) so we
    # don't get confused by an older sweep dir whose mtime got bumped.
    cands.sort(key=lambda p: p.name, reverse=True)
    return cands[0]


def _count_lines(path: Path) -> int | None:
    try:
        return sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-md", default="/workspace/hle-serving-5090/hle_serving_trnew_10_130_clean.md")
    ap.add_argument("--outputs-dir", default="/workspace/ToolOrchestra/outputs")
    ap.add_argument("--orch-port-base", type=int, default=1900)
    ap.add_argument("--tr-router-port-base", type=int, default=28000)
    ap.add_argument("--tr-backend-port-base", type=int, default=28100)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--interval-sec", type=float, default=60.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    report_md = Path(args.report_md)
    state_json = report_md.with_suffix(report_md.suffix + ".state.json")

    while True:
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{now}] trnew status panel")

        processed = None
        if state_json.exists():
            try:
                import json

                obj = json.loads(state_json.read_text(encoding="utf-8"))
                processed = len(obj.get("processed_out_dirs") or [])
            except Exception:
                processed = None
        if processed is not None:
            print(f"  report_processed_out_dirs={processed}  state={state_json}")
        else:
            print(f"  report_state=missing_or_unreadable  state={state_json}")

        parallel_dir = _latest_parallel_dir(outputs_dir)
        if parallel_dir is not None:
            q = parallel_dir / "task_queue.txt"
            qn = _count_lines(q) if q.exists() else None
            print(f"  sweep_dir={parallel_dir}  queue_remaining_lines={qn if qn is not None else 'n/a'}")
        else:
            print("  sweep_dir=n/a")

        for i in range(args.workers):
            orch_port = args.orch_port_base + i
            router_port = args.tr_router_port_base + i
            backend_port = args.tr_backend_port_base + i

            backend_pids = _lsof_listen_pids(backend_port)
            router_pids = _lsof_listen_pids(router_port)

            backend_pid = backend_pids[0] if backend_pids else None
            router_pid = router_pids[0] if router_pids else None

            c = rep = None
            out_dir = None
            if backend_pid is not None:
                c, rep, _ = _infer_task_from_pid(backend_pid)
                orch_log = _find_open_fd_path(backend_pid, suffix="/logs/orchestrator.log")
                if orch_log:
                    out_dir = str(Path(orch_log).parent.parent)

            print(
                "  worker{idx}: gpu={gpu} orch_port={orch} router_port={router} backend_port={backend} "
                "c={c} rep={rep} backend_pid={bpid} router_pid={rpid} out_dir={out}".format(
                    idx=i,
                    gpu=i,
                    orch=orch_port,
                    router=router_port,
                    backend=backend_port,
                    c=c if c is not None else "n/a",
                    rep=rep if rep is not None else "n/a",
                    bpid=backend_pid if backend_pid is not None else "n/a",
                    rpid=router_pid if router_pid is not None else "n/a",
                    out=out_dir if out_dir is not None else "n/a",
                )
            )

        print("", flush=True)

        if args.once:
            break
        time.sleep(float(args.interval_sec))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
