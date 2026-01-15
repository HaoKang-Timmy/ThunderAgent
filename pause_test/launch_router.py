#!/usr/bin/env python3
"""Start the multi-backend router for the four local vLLM servers."""

import argparse
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

DEFAULT_PORTS = [8100, 8101, 8102, 8103]
ROUTER_PORT = 8300


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch router with configurable control parameters.")
    parser.add_argument("--thrashing-distance", type=float, default=0.05, help="Thrashing distance (default: 0.05)")
    parser.add_argument("--control-window", type=float, default=4.5, help="Control time window seconds (default: 4.5)")
    parser.add_argument("--control-step", type=float, default=0.4, help="Control step (default: 0.4)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    backends = [f"http://127.0.0.1:{p}" for p in DEFAULT_PORTS]
    env = os.environ.copy()
    env["VLLM_BACKENDS"] = ",".join(backends)
    env["THRASHING_DISTANCE"] = str(args.thrashing_distance)
    env["CONTROL_TIME_WINDOW_S"] = str(args.control_window)
    env["CONTROL_STEP"] = str(args.control_step)
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_fp = (log_dir / "router.log").open("a", buffering=1)

    cmd = [
        "uvicorn",
        "multi_backend_router:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(ROUTER_PORT),
        "--log-level",
        "info",
    ]
    logging.info(
        "Starting router on port %s with backends: %s | thrashing=%.3f window=%.2f step=%.3f",
        ROUTER_PORT,
        env["VLLM_BACKENDS"],
        args.thrashing_distance,
        args.control_window,
        args.control_step,
    )
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
    )

    def handle_term(_signum, _frame):
        logging.warning("Received termination signal, shutting down router...")
        proc.terminate()

    signal.signal(signal.SIGTERM, handle_term)

    try:
        returncode = proc.wait()
    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt received, terminating router...")
        proc.terminate()
        try:
            returncode = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            returncode = proc.wait()

    if returncode != 0:
        logging.error("Router exited with status %s", returncode)
        sys.exit(returncode)
    log_fp.close()


if __name__ == "__main__":
    main()
