#!/usr/bin/env python3
"""Launch four vLLM servers sequentially, waiting for each to become healthy."""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL_PATH = "/mnt/shared/models/SWE-agent-LM-32B"
PORTS = [8100, 8101, 8102, 8103]
DEFAULT_GPU_PAIRS = ["0,1", "2,3", "4,5", "6,7"]


def parse_gpu_pairs(raw: str) -> list[str]:
    pairs = [part.strip() for part in raw.split(";") if part.strip()]
    if len(pairs) != 4:
        raise ValueError("GPU pair list must contain four entries like '0,1;2,3;4,5;6,7'")
    return pairs


def wait_for_health(port: int, timeout_s: int = 900) -> None:
    """Poll the vLLM /health endpoint until it returns 200 or timeout."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(2.0)
    raise RuntimeError(f"Timed out waiting for vLLM server on port {port} to become healthy")


def terminate_processes(procs: list[subprocess.Popen], timeout: float = 10.0) -> None:
    """Terminate a list of subprocesses gracefully, then force kill if needed."""
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(p.poll() is not None for p in procs):
            return
        time.sleep(0.5)
    for proc in procs:
        if proc.poll() is None:
            proc.kill()


def launch_servers(gpu_pairs: list[str]) -> None:
    procs: list[subprocess.Popen] = []
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_files: list = []
    try:
        for port, gpu_pair in zip(PORTS, gpu_pairs):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_pair
            log_fp = (log_dir / f"vllm_{port}.log").open("a", buffering=1)
            log_files.append(log_fp)
            cmd = [
                "vllm",
                "serve",
                MODEL_PATH,
                "--quantization",
                "fp8",
                "--kv-cache-dtype",
                "fp8",
                "--tensor-parallel-size",
                "2",
                "--port",
                str(port),
                "--host",
                "0.0.0.0",
            ]
            logging.info("Starting vLLM on port %s with CUDA_VISIBLE_DEVICES=%s", port, gpu_pair)
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
            )
            procs.append(proc)
            wait_for_health(port)
            logging.info("vLLM on port %s is healthy; proceeding to next server.", port)
        logging.info("All four vLLM servers started and reported healthy. Waiting for exit (Ctrl+C to stop).")
        for proc in procs:
            proc.wait()
    except KeyboardInterrupt:
        logging.warning("Received KeyboardInterrupt, terminating vLLM servers...")
        terminate_processes(procs)
    except Exception as exc:
        logging.error("Error while launching servers: %s", exc, exc_info=True)
        terminate_processes(procs)
        sys.exit(1)
    finally:
        for fp in log_files:
            try:
                fp.close()
            except Exception:
                pass


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Launch four vLLM servers sequentially.")
    parser.add_argument(
        "--gpu-pairs",
        default=";".join(DEFAULT_GPU_PAIRS),
        help="Semicolon-separated CUDA_VISIBLE_DEVICES pairs for each server (four entries). "
        "Example: '0,1;2,3;4,5;6,7'",
    )
    args = parser.parse_args()
    gpu_pairs = parse_gpu_pairs(args.gpu_pairs)

    # Ensure subprocesses die if parent receives SIGTERM.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    launch_servers(gpu_pairs)


if __name__ == "__main__":
    main()
