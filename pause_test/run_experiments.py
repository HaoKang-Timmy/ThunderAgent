#!/usr/bin/env python3
"""Batch runner for multiple router control parameter sweeps."""

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DOCKER_CONFIG = {
    "auths": {
        "https://index.docker.io/v1/": {
            "auth": "ZXJndDEwOmRja3JfcGF0X25mcDJCNFlMWjdDVHFTbnNZQjF2QXREMDUySQ=="
        }
    }
}

ROUTER_PORT = 8300
ROUTER_URL = f"http://127.0.0.1:{ROUTER_PORT}"

RUNS = [
    {"thrashing": 0.05, "window": 2.0, "step": 0.2},
    {"thrashing": 0.05, "window": 3.0, "step": 0.2},
    {"thrashing": 0.05, "window": 3.5, "step": 0.2},
    {"thrashing": 0.05, "window": 4.0, "step": 0.2},
    {"thrashing": 0.05, "window": 4.5, "step": 0.2},
    {"thrashing": 0.05, "window": 5.0, "step": 0.2},
]


def write_docker_config():
    """Overwrite ~/.docker/config.json with required auth."""
    docker_dir = Path.home() / ".docker"
    docker_dir.mkdir(parents=True, exist_ok=True)
    config_path = docker_dir / "config.json"
    config_path.write_text(json.dumps(DOCKER_CONFIG, indent=4))
    return config_path


def verify_docker_config(config_path: Path) -> bool:
    try:
        current = json.loads(config_path.read_text())
        return current == DOCKER_CONFIG
    except Exception:
        return False


def wait_for_router(timeout_s: float = 60.0) -> None:
    """Poll /programs until router responds or timeout."""
    url = f"{ROUTER_URL}/programs"
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(1.0)
    raise RuntimeError(f"Router did not become ready in {timeout_s}s (last_error={last_error})")


def start_router(thrashing: float, window: float, step: float):
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_fp = (log_dir / f"router_{thrashing}_{window}_{step}.log").open("a", buffering=1)
    env = os.environ.copy()
    env["VLLM_BACKENDS"] = env.get(
        "VLLM_BACKENDS",
        "http://127.0.0.1:8100,http://127.0.0.1:8101,http://127.0.0.1:8102,http://127.0.0.1:8103",
    )
    cmd = [
        sys.executable,
        "launch_router.py",
        "--thrashing-distance",
        str(thrashing),
        "--control-window",
        str(window),
        "--control-step",
        str(step),
    ]
    logging.info("Starting router: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
    )
    return proc, log_fp


def stop_process(proc: subprocess.Popen, timeout: float = 10.0):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def run_swebench(output_dir: str):
    env = os.environ.copy()
    env["MSWEA_ROUTER_ADMIN_URL"] = ROUTER_URL
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
        f"./{output_dir}",
    ]
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_fp = (log_dir / f"swebench_{output_dir}.log").open("a", buffering=1)
    logging.info("Running swebench: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd="mini-swe-agent",
        env=env,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
    )
    return proc, log_fp


def move_output(output_dir: str):
    src = Path("mini-swe-agent") / output_dir
    dst_root = Path("/mnt/shared/data")
    dst_root.mkdir(parents=True, exist_ok=True)
    dst = dst_root / output_dir
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))
    logging.info("Moved %s to %s", src, dst)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = write_docker_config()
    logging.info("Wrote docker config to %s", config_path)

    for run in RUNS:
        thrashing = run["thrashing"]
        window = run["window"]
        step = run["step"]
        output_dir = f"{thrashing}_{window}_{step}"
        logging.info("=== Starting run %s ===", output_dir)

        # Ensure config is fresh before each run
        write_docker_config()

        router_proc, router_log = start_router(thrashing, window, step)
        try:
            wait_for_router()
            swe_proc, swe_log = run_swebench(output_dir)
            try:
                swe_return = swe_proc.wait()
                swe_log.close()
                if swe_return != 0:
                    raise RuntimeError(f"swebench returned non-zero exit {swe_return}")
                move_output(output_dir)
            finally:
                stop_process(swe_proc)
        finally:
            stop_process(router_proc)
            router_log.close()

        if not verify_docker_config(config_path):
            raise RuntimeError("Docker config changed unexpectedly after run.")

        logging.info("=== Finished run %s ===", output_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted by user, exiting.")
