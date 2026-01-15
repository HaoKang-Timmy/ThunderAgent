#!/usr/bin/env python3
"""Run mini-swe-agent SWE-bench batch with the router wired in via MSWEA_ROUTER_ADMIN_URL."""

import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

ROUTER_URL = os.getenv("MSWEA_ROUTER_ADMIN_URL", "http://127.0.0.1:8300")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    env = os.environ.copy()
    env["MSWEA_ROUTER_ADMIN_URL"] = ROUTER_URL
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_fp = (log_dir / "swebench.log").open("a", buffering=1)

    cmd = [
        sys.executable,
        "-m",
        "minisweagent.run.extra.swebench",
        "--model-class",
        "vllm",
        "--model",
        "/mnt/shared/models/SWE-agent-LM-32B",
        "--subset",
        "lite",
        "--split",
        "test",
        "--workers",
        "128",
        "--output",
        "./swebench_output",
    ]

    logging.info("Running SWE-bench with MSWEA_ROUTER_ADMIN_URL=%s", env["MSWEA_ROUTER_ADMIN_URL"])
    proc = subprocess.Popen(
        cmd,
        cwd="mini-swe-agent",
        env=env,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
    )

    def handle_term(_signum, _frame):
        logging.warning("Received termination signal, stopping SWE-bench run...")
        proc.terminate()

    signal.signal(signal.SIGTERM, handle_term)

    try:
        returncode = proc.wait()
    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt received, terminating SWE-bench run...")
        proc.terminate()
        try:
            returncode = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            returncode = proc.wait()

    if returncode != 0:
        logging.error("SWE-bench run failed with status %s", returncode)
        sys.exit(returncode)
    log_fp.close()


if __name__ == "__main__":
    main()
