#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Local runner for HLE evaluation (no SLURM).
# Starts:
#   - Retrieval service (FastAPI) on GPU 1 (conda env: retriever)
#   - Orchestrator-8B vLLM OpenAI server on GPU 0 (conda env: vllm1)
# Then runs eval via eval_hle_local.py with a model_config JSON compatible with eval_hle.py.

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import requests
import selectors


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _tail_last_line(path: str) -> str:
    try:
        if not path or not os.path.isfile(path):
            return ""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        return ""
    except Exception:
        return ""


def _tail_last_n(path: str, n: int = 40) -> str:
    try:
        if not path or not os.path.isfile(path):
            return ""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        tail = lines[-n:] if len(lines) > n else lines
        return "\n".join(tail).rstrip()
    except Exception:
        return ""


def find_conda_exe() -> str:
    """
    Resolve the conda executable path robustly.
    This avoids failures when conda isn't on PATH (common in non-interactive shells).
    """
    candidates = []
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        candidates.append(conda_exe)
    which = shutil.which("conda")
    if which:
        candidates.append(which)
    candidates.extend(
        [
            "/root/miniconda3/bin/conda",
            "/opt/conda/bin/conda",
            str(Path.home() / "miniconda3" / "bin" / "conda"),
            str(Path.home() / "anaconda3" / "bin" / "conda"),
        ]
    )
    for c in candidates:
        try:
            if c and os.path.exists(c):
                return c
        except Exception:
            pass
    raise RuntimeError(
        "conda executable not found. Ensure conda is installed and on PATH, "
        "or set CONDA_EXE to the full path of the conda binary."
    )

def find_conda_sh(conda_exe: Optional[str] = None) -> str:
    """
    Resolve the conda.sh path for `source conda.sh && conda activate <env>`.
    """
    candidates = []
    conda_sh_env = os.environ.get("CONDA_SH")
    if conda_sh_env:
        candidates.append(conda_sh_env)

    try:
        conda_exe = conda_exe or os.environ.get("CONDA_EXE") or shutil.which("conda")
        if conda_exe:
            base = Path(conda_exe).resolve().parent.parent
            candidates.append(str(base / "etc" / "profile.d" / "conda.sh"))
    except Exception:
        pass

    candidates.extend(
        [
            "/root/miniconda3/etc/profile.d/conda.sh",
            "/opt/conda/etc/profile.d/conda.sh",
            str(Path.home() / "miniconda3" / "etc" / "profile.d" / "conda.sh"),
            str(Path.home() / "anaconda3" / "etc" / "profile.d" / "conda.sh"),
        ]
    )
    for c in candidates:
        try:
            if c and os.path.isfile(c):
                return c
        except Exception:
            pass
    raise RuntimeError(
        "conda.sh not found. Set CONDA_SH to the full path of conda.sh, "
        "or install conda in a standard location (e.g., /root/miniconda3)."
    )


def conda_activate_cmd(conda_sh: str, conda_env: str, inner_cmd: list) -> list:
    """
    Build a command that runs `inner_cmd` inside `conda_env` without using `conda run`.
    """
    inner = " ".join(shlex.quote(str(x)) for x in inner_cmd)
    script = f"source {shlex.quote(conda_sh)} && conda activate {shlex.quote(conda_env)} && exec {inner}"
    return ["bash", "-lc", script]


@dataclass
class ManagedProcess:
    name: str
    cmd: list
    env: Dict[str, str]
    cwd: str
    stdout_path: str
    stderr_path: str
    process: Optional[subprocess.Popen] = None
    _stdout_fh: Optional[Any] = None
    _stderr_fh: Optional[Any] = None

    def start(self) -> None:
        os.makedirs(os.path.dirname(self.stdout_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.stderr_path), exist_ok=True)
        self._stdout_fh = open(self.stdout_path, "w", encoding="utf-8")
        self._stderr_fh = open(self.stderr_path, "w", encoding="utf-8")
        log(f"Starting {self.name}: {' '.join(self.cmd)}")
        log(f"  cwd={self.cwd}")
        log(f"  stdout={self.stdout_path}")
        log(f"  stderr={self.stderr_path}")
        try:
            cvd = self.env.get("CUDA_VISIBLE_DEVICES")
            if cvd is not None:
                log(f"  CUDA_VISIBLE_DEVICES={cvd}")
        except Exception:
            pass
        self.process = subprocess.Popen(
            self.cmd,
            env=self.env,
            cwd=self.cwd,
            stdout=self._stdout_fh,
            stderr=self._stderr_fh,
            text=True,
            start_new_session=True,
        )

    def poll(self) -> Optional[int]:
        if not self.process:
            return None
        return self.process.poll()

    def stop(self) -> None:
        if not self.process:
            return
        if self.process.poll() is not None:
            return
        log(f"Stopping {self.name} (pid={self.process.pid})...")
        # Best-effort: terminate the whole process group (important when a wrapper shell spawns children).
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except Exception:
            self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            log(f"Force killing {self.name}...")
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except Exception:
                self.process.kill()
            self.process.wait()
        finally:
            try:
                if self._stdout_fh:
                    self._stdout_fh.close()
                if self._stderr_fh:
                    self._stderr_fh.close()
            except Exception:
                pass


class VLLMServer(ManagedProcess):
    def __init__(
        self,
        name: str,
        model_path: str,
        port: int,
        gpu_id: int,
        conda_env: Optional[str] = None,
        conda_sh: Optional[str] = None,
        tool_parser: str = "hermes",
        log_dir: str = "logs/hle_local",
        cwd: str = ".",
        extra_args: Optional[list] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stdout_path = os.path.join(log_dir, f"{name}_port_{port}_{timestamp}.out")
        stderr_path = os.path.join(log_dir, f"{name}_port_{port}_{timestamp}.err")
        base_cmd = [
            "vllm",
            "serve",
            model_path,
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            tool_parser,
            "--port",
            str(port),
        ]
        if extra_args:
            base_cmd.extend(list(extra_args))
        cmd = (
            conda_activate_cmd(conda_sh or find_conda_sh(), conda_env, base_cmd)
            if conda_env
            else base_cmd
        )
        proc_env = (env or os.environ.copy()).copy()
        proc_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        proc_env.setdefault("PYTHONUNBUFFERED", "1")
        super().__init__(
            name=name,
            cmd=cmd,
            env=proc_env,
            cwd=cwd,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        self.port = port
        self.ip_addr = "127.0.0.1"

    def is_ready(self, timeout_s: int = 5) -> bool:
        try:
            r = requests.get(f"http://{self.ip_addr}:{self.port}/health", timeout=timeout_s)
            return r.status_code == 200
        except Exception:
            return False

    def wait_until_ready(self, max_wait_s: int = 900, check_interval_s: int = 10) -> bool:
        log(f"Waiting for {self.name} vLLM server on port {self.port}...")
        start = time.time()
        last_stdout = ""
        last_stderr = ""
        while time.time() - start < max_wait_s:
            if self.process and self.process.poll() is not None:
                log(f"ERROR: {self.name} exited early (code={self.process.returncode}).")
                err_tail = _tail_last_n(self.stderr_path, n=120)
                out_tail = _tail_last_n(self.stdout_path, n=80)
                if err_tail:
                    log(f"{self.name} stderr tail:\n{err_tail}")
                if out_tail:
                    log(f"{self.name} stdout tail:\n{out_tail}")
                return False
            if self.is_ready():
                log(f"✓ {self.name} is ready (port {self.port})")
                return True
            elapsed = int(time.time() - start)
            cur_out = _tail_last_line(self.stdout_path)
            cur_err = _tail_last_line(self.stderr_path)
            if cur_out and cur_out != last_stdout:
                log(f"{self.name} stdout: {cur_out}")
                last_stdout = cur_out
            if cur_err and cur_err != last_stderr:
                log(f"{self.name} stderr: {cur_err}")
                last_stderr = cur_err
            log(f"{self.name} still starting... (elapsed={elapsed}s)")
            time.sleep(check_interval_s)
        log(f"ERROR: {self.name} did not become ready within {max_wait_s}s")
        err_tail = _tail_last_n(self.stderr_path, n=120)
        out_tail = _tail_last_n(self.stdout_path, n=80)
        if err_tail:
            log(f"{self.name} stderr tail:\n{err_tail}")
        if out_tail:
            log(f"{self.name} stdout tail:\n{out_tail}")
        return False


def wait_fastapi_ready(
    url: str,
    proc: Optional[ManagedProcess] = None,
    max_wait_s: int = 1200,
    check_interval_s: int = 10,
) -> bool:
    log(f"Waiting for retrieval server: {url}")
    start = time.time()
    last_stdout = ""
    last_stderr = ""
    while time.time() - start < max_wait_s:
        if proc and proc.process and proc.process.poll() is not None:
            log(f"ERROR: retrieval process exited early (code={proc.process.returncode}).")
            err_tail = _tail_last_n(proc.stderr_path, n=200)
            out_tail = _tail_last_n(proc.stdout_path, n=120)
            if err_tail:
                log(f"retrieval stderr tail:\n{err_tail}")
            if out_tail:
                log(f"retrieval stdout tail:\n{out_tail}")
            return False
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                log("✓ retrieval server is ready")
                return True
        except Exception:
            pass
        elapsed = int(time.time() - start)
        if proc:
            cur_out = _tail_last_line(proc.stdout_path)
            cur_err = _tail_last_line(proc.stderr_path)
            if cur_out and cur_out != last_stdout:
                log(f"retrieval stdout: {cur_out}")
                last_stdout = cur_out
            if cur_err and cur_err != last_stderr:
                log(f"retrieval stderr: {cur_err}")
                last_stderr = cur_err
        log(f"retrieval still starting... (elapsed={elapsed}s)")
        time.sleep(check_interval_s)
    log(f"ERROR: retrieval server did not become ready within {max_wait_s}s")
    if proc:
        err_tail = _tail_last_n(proc.stderr_path, n=200)
        out_tail = _tail_last_n(proc.stdout_path, n=120)
        if err_tail:
            log(f"retrieval stderr tail:\n{err_tail}")
        if out_tail:
            log(f"retrieval stdout tail:\n{out_tail}")
    return False


def write_model_config(
    model_config_path: str,
    retrieval_ip: str,
    retrieval_port: int,
    agent_model_name: str,
    agent_ip: str,
    agent_port: int,
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(model_config_path), exist_ok=True)
    cfg: Dict[str, Any] = {
        "retrieval": [{"ip_addr": retrieval_ip, "port": str(retrieval_port)}],
        # Keep the original keys used by run_hle.py so eval scripts can be swapped easily.
        "Qwen/Qwen2.5-Math-72B-Instruct": [],
        "Qwen/Qwen3-32B": [],
        "Qwen/Qwen2.5-Math-7B-Instruct": [],
        "meta-llama/Llama-3.3-70B-Instruct": [],
        agent_model_name: [{"ip_addr": agent_ip, "port": str(agent_port)}],
        "Qwen/Qwen2.5-Coder-32B-Instruct": [],
        "vllm_model_config_path": model_config_path,
    }
    with open(model_config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    log(f"Wrote model_config to {model_config_path}")
    return cfg


def _retrieval_smoke_call(port: int, eid: str, query: str) -> None:
    payload = {"queries": [query], "topk": 100, "return_scores": True, "eid": eid}
    log(f"Smoke testing /retrieve (eid={eid})...")
    r = requests.post(f"http://127.0.0.1:{port}/retrieve", json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    try:
        n = len(data[0]) if isinstance(data, list) and data else 0
        log(f"✓ /retrieve ok (returned {n} results)")
    except Exception:
        log("✓ /retrieve ok (response received)")

def _http_ok(url: str, timeout_s: int = 2) -> bool:
    try:
        r = requests.get(url, timeout=timeout_s)
        return r.status_code == 200
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HLE evaluation locally (no SLURM).")
    parser.add_argument(
        "--agent-model",
        type=str,
        default=os.getenv("CKPT_DIR", ""),
        help="Path to Orchestrator-8B checkpoint (default: $CKPT_DIR)",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=os.getenv("INDEX_DIR", "/workspace/dataset/multi-train/index"),
        help="Directory containing eval.index and eval.jsonl (default: $INDEX_DIR or /workspace/dataset/multi-train/index)",
    )
    parser.add_argument(
        "--example-path",
        type=str,
        default="hle.jsonl",
        help="HLE eval jsonl path (default: evaluation/hle.jsonl). If you have the real file elsewhere, pass an absolute path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/hle_local",
        help="Output directory (default: outputs/hle_local)",
    )
    parser.add_argument(
        "--model-config-path",
        type=str,
        default="model_configs/hle_local.json",
        help="Where to write model config JSON (default: model_configs/hle_local.json)",
    )
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Max number of concurrent HLE questions (default: 2). For smoke test use 25; for full run try 512.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "PROFILE", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="HLE structured log level (default: INFO). Use PROFILE to enable timing logs.",
    )

    parser.add_argument("--orchestrator-gpu", type=int, default=0)
    parser.add_argument("--retrieval-gpu", type=int, default=1)
    parser.add_argument("--orchestrator-port", type=int, default=1406)
    parser.add_argument("--retrieval-port", type=int, default=1401)
    parser.add_argument(
        "--retriever-conda-env",
        type=str,
        default="retriever",
        help="Conda env name for retrieval_hle.py (default: retriever)",
    )
    parser.add_argument(
        "--vllm-conda-env",
        type=str,
        default="vllm1",
        help="Conda env name for vLLM server + eval_hle_local.py (default: vllm1)",
    )

    parser.add_argument("--skip-server-start", action="store_true", help="Skip starting local servers")
    parser.add_argument("--skip-retrieval-start", action="store_true", help="Skip starting retrieval server")
    parser.add_argument("--skip-orchestrator-start", action="store_true", help="Skip starting orchestrator vLLM server")
    parser.add_argument("--skip-eval", action="store_true", help="Start servers but do not run eval; keep running until Ctrl-C")
    parser.add_argument("--smoke-retrieval", action="store_true", help="Start retrieval server and run one /retrieve request, then exit")
    parser.add_argument(
        "--no-reuse-running",
        action="store_false",
        dest="reuse_running",
        help="Do not reuse already-running servers on the target ports (default: reuse if reachable)",
    )
    parser.set_defaults(reuse_running=True)
    parser.add_argument(
        "--conda-sh",
        type=str,
        default=os.environ.get("CONDA_SH", ""),
        help="Path to conda.sh for `source conda.sh && conda activate ...` (default: auto-detect or $CONDA_SH)",
    )

    parser.add_argument("--log-dir", type=str, default="logs/hle_local")
    parser.add_argument("--retrieval-cache-dir", type=str, default="cache/hle")
    parser.add_argument(
        "--example-id-file",
        type=str,
        default="examples.json",
        help="Path to examples.json (default: evaluation/examples.json)",
    )
    parser.add_argument(
        "--smoke-eid",
        type=str,
        default="673eb1cfadce15d9254eb2ac",
        help="EID to use for --smoke-retrieval (must exist in examples.json)",
    )
    parser.add_argument(
        "--smoke-query",
        type=str,
        default="Peter Sloterdijk considers that the State is a metaphor for which anthroposphere?",
        help="Query to use for --smoke-retrieval",
    )

    args = parser.parse_args()

    # Resolve repo paths
    repo_path = os.getenv("REPO_PATH")
    if not repo_path:
        repo_path = str(Path(__file__).resolve().parents[1])
        os.environ["REPO_PATH"] = repo_path
        log(f"REPO_PATH not set; inferred: {repo_path}")
    else:
        log(f"Using REPO_PATH={repo_path}")

    eval_dir = str(Path(repo_path) / "evaluation")
    if not os.path.isdir(eval_dir):
        log(f"ERROR: evaluation directory not found: {eval_dir}")
        return 1

    # Ensure INDEX_DIR is set for retrieval_hle.py
    os.environ["INDEX_DIR"] = args.index_dir
    if not os.path.isdir(args.index_dir):
        log(f"ERROR: index dir does not exist: {args.index_dir}")
        return 1
    for fn in ("eval.index", "eval.jsonl"):
        p = os.path.join(args.index_dir, fn)
        if not os.path.exists(p):
            log(f"ERROR: missing index file: {p}")
            return 1

    # Resolve example path (if relative, interpret relative to evaluation/)
    example_path = args.example_path
    if not os.path.isabs(example_path):
        example_path = os.path.join(eval_dir, example_path)
    if not os.path.exists(example_path):
        log(f"WARNING: example file not found: {example_path}")
        log("         If evaluation/hle.jsonl is a git-lfs pointer in your checkout, pass the real dataset path via --example-path")

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(eval_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    model_config_path = args.model_config_path
    if not os.path.isabs(model_config_path):
        model_config_path = os.path.join(eval_dir, model_config_path)

    log_dir = args.log_dir
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(eval_dir, log_dir)
    os.makedirs(log_dir, exist_ok=True)

    example_id_file = args.example_id_file
    if not os.path.isabs(example_id_file):
        example_id_file = os.path.join(eval_dir, example_id_file)
    if not os.path.exists(example_id_file):
        log(f"ERROR: examples.json not found: {example_id_file}")
        return 1

    tavily_key = os.getenv("TAVILY_KEY", "")
    if not tavily_key:
        log("WARNING: TAVILY_KEY not set. Retrieval fallback to Tavily will not work.")

    def _count_jsonl(path: str) -> int:
        try:
            if not path or not os.path.isfile(path):
                return 0
            n = 0
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.strip():
                        n += 1
            return n
        except Exception:
            return 0

    def _fmt_hms(seconds: float) -> str:
        if seconds < 0:
            seconds = 0
        s = int(seconds)
        h = s // 3600
        m = (s % 3600) // 60
        s = s % 60
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    conda_exe = find_conda_exe()
    conda_sh = args.conda_sh.strip() or find_conda_sh(conda_exe)
    log(f"Using conda: {conda_exe}")
    log(f"Using conda.sh: {conda_sh}")

    orchestrator: Optional[VLLMServer] = None
    retrieval: Optional[ManagedProcess] = None

    def _cleanup(*_a):
        nonlocal orchestrator, retrieval
        log("Cleaning up processes...")
        try:
            if retrieval:
                retrieval.stop()
        finally:
            if orchestrator:
                orchestrator.stop()

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    # Only require agent model path if we may start orchestrator / run eval.
    agent_model = args.agent_model
    if not args.smoke_retrieval:
        if not agent_model:
            log("ERROR: --agent-model is required (or set CKPT_DIR)")
            return 1
        if not os.path.exists(agent_model):
            log(f"ERROR: agent model path does not exist: {agent_model}")
            return 1

    try:
        retrieval_openapi = f"http://127.0.0.1:{args.retrieval_port}/openapi.json"
        if (
            args.reuse_running
            and not args.skip_retrieval_start
            and not args.skip_server_start
            and _http_ok(retrieval_openapi)
        ):
            log(f"Reusing existing retrieval server on port {args.retrieval_port} (--reuse-running)")
        elif not args.skip_server_start and not args.skip_retrieval_start:
            retrieval_env = os.environ.copy()
            retrieval_env["CUDA_VISIBLE_DEVICES"] = str(args.retrieval_gpu)
            retrieval_env.setdefault("PYTHONUNBUFFERED", "1")
            retrieval_cmd = conda_activate_cmd(
                conda_sh,
                args.retriever_conda_env,
                [
                    "python",
                    os.path.join(eval_dir, "retrieval_hle.py"),
                    "--port",
                    str(args.retrieval_port),
                    "--new_cache_dir",
                    args.retrieval_cache_dir,
                    "--example_id_file",
                    example_id_file,
                    "--tavily_key",
                    tavily_key,
                ],
            )
            retrieval = ManagedProcess(
                name="retrieval_hle",
                cmd=retrieval_cmd,
                env=retrieval_env,
                cwd=eval_dir,
                stdout_path=os.path.join(log_dir, "retrieval_hle.out"),
                stderr_path=os.path.join(log_dir, "retrieval_hle.err"),
            )
            retrieval.start()
            if not wait_fastapi_ready(
                retrieval_openapi,
                proc=retrieval,
            ):
                return 1

        if args.smoke_retrieval:
            _retrieval_smoke_call(port=args.retrieval_port, eid=args.smoke_eid, query=args.smoke_query)
            return 0

        orchestrator_health = f"http://127.0.0.1:{args.orchestrator_port}/health"
        if (
            args.reuse_running
            and not args.skip_orchestrator_start
            and not args.skip_server_start
            and _http_ok(orchestrator_health)
        ):
            log(f"Reusing existing orchestrator server on port {args.orchestrator_port} (--reuse-running)")
        elif not args.skip_server_start and not args.skip_orchestrator_start:
            orchestrator = VLLMServer(
                name="orchestrator",
                model_path=agent_model,
                port=args.orchestrator_port,
                gpu_id=args.orchestrator_gpu,
                conda_env=args.vllm_conda_env,
                conda_sh=conda_sh,
                tool_parser="hermes",
                log_dir=log_dir,
                cwd=eval_dir,
            )
            orchestrator.start()
            if not orchestrator.wait_until_ready():
                return 1

        # Always write config (even if servers are pre-started), so eval script has a stable path.
        write_model_config(
            model_config_path=model_config_path,
            retrieval_ip="127.0.0.1",
            retrieval_port=args.retrieval_port,
            agent_model_name=agent_model,
            agent_ip="127.0.0.1",
            agent_port=args.orchestrator_port,
        )

        if args.skip_eval:
            log("Servers started; skipping eval. Press Ctrl-C to stop.")
            try:
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                return 0

        eval_cmd = conda_activate_cmd(
            conda_sh,
            args.vllm_conda_env,
            [
                "python",
                os.path.join(eval_dir, "eval_hle_local.py"),
                "--model_name",
                agent_model,
                "--output_dir",
                output_dir,
                "--model_config",
                model_config_path,
                "--max_rounds",
                str(args.max_rounds),
                "--example_path",
                example_path,
                "--concurrency",
                str(args.concurrency),
                "--log_level",
                str(args.log_level),
                "--log_file",
                os.path.join(log_dir, "hle.log"),
            ],
        )
        log(f"Running eval: {' '.join(eval_cmd)}")
        eval_env = os.environ.copy()
        eval_env.setdefault("PYTHONUNBUFFERED", "1")

        # Also export env vars for eval_hle_local.py (so users can run it standalone).
        eval_env["HLE_LOG_LEVEL"] = str(args.log_level)
        eval_env["HLE_LOG_FILE"] = os.path.join(log_dir, "hle.log")
        # Keep stdout structured logs off by default to avoid spamming the runner output.
        eval_env.setdefault("HLE_LOG_STREAM", "0")

        total = _count_jsonl(example_path)
        done = 0
        start_ts = time.time()

        eval_log_path = os.path.join(log_dir, "eval_hle_local.log")
        with open(eval_log_path, "a", encoding="utf-8", buffering=1) as eval_log:
            process = subprocess.Popen(
                eval_cmd,
                cwd=eval_dir,
                env=eval_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            selector = selectors.DefaultSelector()
            if process.stdout is not None:
                selector.register(process.stdout, selectors.EVENT_READ, data="STDOUT")
            if process.stderr is not None:
                selector.register(process.stderr, selectors.EVENT_READ, data="STDERR")

            last_output_ts = time.time()
            last_heartbeat_ts = last_output_ts

            while True:
                retcode = process.poll()
                events = selector.select(timeout=0.2)
                for key, _mask in events:
                    stream_name = key.data
                    fileobj = key.fileobj
                    try:
                        line = fileobj.readline()
                    except Exception as e:
                        log(f"WARNING: failed reading eval {stream_name}: {e}")
                        try:
                            selector.unregister(fileobj)
                        except Exception:
                            pass
                        continue

                    if line:
                        last_output_ts = time.time()
                        raw = line.rstrip("\n")

                        # Persist eval output (survives cancellation)
                        try:
                            eval_log.write(f"[eval_hle_local.py {stream_name}] {raw}\n")
                            eval_log.flush()
                        except Exception:
                            pass

                        # Update ETA on per-task completion markers.
                        # NOTE: stdout from many threads may interleave; use substring match and count.
                        if "[HLE_TASK_COMPLETE]" in raw:
                            n_marks = raw.count("[HLE_TASK_COMPLETE]") or 1
                            for _ in range(n_marks):
                                done += 1
                                if total > 0 and done > 0:
                                    elapsed = time.time() - start_ts
                                    remaining = max(0, total - done)
                                    per_task = elapsed / done
                                    eta_s = per_task * remaining
                                    eta_line = f"[OVERALL_ETA] done={done}/{total} elapsed={_fmt_hms(elapsed)} eta={_fmt_hms(eta_s)}"
                                    # Red ANSI line for terminal
                                    print("\033[31m" + eta_line + "\033[0m", flush=True)
                                    # Also persist ETA to eval log (plain text; no ANSI)
                                    try:
                                        eval_log.write(f"[run_hle_local.py ETA] {eta_line}\n")
                                        eval_log.flush()
                                    except Exception:
                                        pass

                        print(f"[eval_hle_local.py {stream_name}] {raw}", flush=True)
                    else:
                        # EOF on this stream
                        try:
                            selector.unregister(fileobj)
                        except Exception:
                            pass

                # Heartbeat so users can distinguish \"quiet\" from \"stuck\"
                now = time.time()
                if retcode is None and (now - last_heartbeat_ts) > 30 and (now - last_output_ts) > 30:
                    last_heartbeat_ts = now
                    log(f"HLE eval still running (pid={process.pid}); no output for {int(now - last_output_ts)}s...")

                if retcode is not None and not selector.get_map():
                    return retcode
    finally:
        _cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
