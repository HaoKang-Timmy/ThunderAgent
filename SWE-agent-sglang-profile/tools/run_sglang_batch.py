#!/usr/bin/env python3
"""Launch a local sglang OpenAI-compatible server and run a SWE-agent batch.

All parameters for the sglang server and the downstream `sweagent run-batch`
invocation are specified in a YAML configuration file so that different
setups can be captured declaratively.
"""

from __future__ import annotations

import argparse
import http.client
import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from sweagent.run.common import _parse_args_to_nested_dict
from sweagent.utils.serialization import merge_nested_dicts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start an sglang OpenAI-compatible server and run `sweagent run-batch` using settings from a YAML file.",
    )
    parser.add_argument(
        "--config-file",
        default="config/sglang_batch.yaml",
        help="Path to YAML config describing the sglang server and run-batch options (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--only-runs",
        nargs="+",
        help="Only execute the runs whose label (from batch.runs[].label) or numeric index matches one of the provided values.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Configuration at {path} must be a mapping")
    return dict(data)


def ensure_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Section '{name}' must be a mapping")
    return dict(value)


def _normalize_sequence(value: Any) -> list[str]:
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    raise ValueError("Expected command to be a sequence or string")


def build_sglang_command(config: Mapping[str, Any]) -> list[str]:
    if "command" in config and config["command"]:
        return _normalize_sequence(config["command"])

    def _first_present(*keys: str) -> Any:
        for key in keys:
            if key in config and config[key] is not None:
                return config[key]
        return None

    model_path = _first_present("model_path", "model-path")
    if not model_path:
        raise ValueError("sglang config must contain either 'command' or 'model_path'")

    cmd: list[str] = [
        sys.executable,
        "-m",
        "sglang.launch_server",
    ]

    config_path = _first_present("config", "config_path")
    if config_path:
        cmd.extend(["--config", str(config_path)])

    cmd.extend(["--model-path", str(model_path)])

    host = _first_present("host")
    if host is None:
        host = "0.0.0.0"
    cmd.extend(["--host", str(host)])

    port = _first_present("port")
    if port is None:
        port = 30000
    cmd.extend(["--port", str(port)])

    value_options: dict[str, str] = {
        "tensor_parallel": "tp-size",
        "tp": "tp-size",
        "pipeline_parallel": "pp-size",
        "pp": "pp-size",
        "data_parallel": "dp-size",
        "dp": "dp-size",
        "nnodes": "nnodes",
        "node-rank": "node-rank",
        "dist-init-addr": "dist-init-addr",
        "tokenizer-path": "tokenizer-path",
        "tokenizer-mode": "tokenizer-mode",
        "context-length": "context-length",
        "mem-fraction-static": "mem-fraction-static",
        "max-total-tokens": "max-total-tokens",
        "max-prefill-tokens": "max-prefill-tokens",
        "chunked-prefill-size": "chunked-prefill-size",
        "schedule-policy": "schedule-policy",
        "schedule-conservativeness": "schedule-conservativeness",
        "kv-cache-dtype": "kv-cache-dtype",
        "quantization": "quantization",
        "model-impl": "model-impl",
        "dtype": "dtype",
        "revision": "revision",
        "load-format": "load-format",
        "tool-call-parser": "tool-call-parser",
        "max-running-requests": "max-running-requests",
    }

    for cfg_key, cli_flag in value_options.items():
        value = _first_present(cfg_key)
        if value is None:
            continue
        cmd.extend([f"--{cli_flag}", str(value)])

    bool_switches: dict[str, str] = {
        "enable-metrics": "enable-metrics",
        "log-requests": "log-requests",
        "enable-torch-compile": "enable-torch-compile",
        "skip-tokenizer-init": "skip-tokenizer-init",
        "skip-server-warmup": "skip-server-warmup",
        "trust-remote-code": "trust-remote-code",
        "enable-cache-report": "enable-cache-report",
    }

    for cfg_key, cli_flag in bool_switches.items():
        present_value = _first_present(cli_flag)
        if isinstance(present_value, str):
            lowered = present_value.strip().lower()
            if lowered in {"true", "1"}:
                present_value = True
            elif lowered in {"false", "0"}:
                present_value = False
        if not present_value:
            continue
        cmd.append(f"--{cli_flag}")

    extra_args = config.get("extra_args")
    if extra_args:
        cmd.extend(_normalize_sequence(extra_args))

    return cmd


def wait_for_server(host: str, port: int, timeout: int, *, path: str = "/v1/models") -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=5)
            conn.request("GET", path)
            response = conn.getresponse()
            if response.status in {200, 401, 403}:
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2)
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
    raise RuntimeError(f"sglang server did not become ready within {timeout}s: {last_error}")


def _merge_env(extra_env: Mapping[str, Any] | None) -> Mapping[str, str]:
    base = os.environ.copy()
    if not extra_env:
        return base
    base.update({str(k): str(v) for k, v in extra_env.items()})
    return base


def _merge_two_envs(base: Mapping[str, Any] | None, override: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not base and not override:
        return {}
    merged: dict[str, Any] = {}
    if base:
        merged.update({str(k): v for k, v in base.items()})
    if override:
        merged.update({str(k): v for k, v in override.items()})
    return merged


def _normalize_server_configs(config: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    """Yield labelled server configurations for multi-stage launches."""
    stages = config.get("stages")
    if stages is None:
        label = str(config.get("label", "sglang"))
        yield label, dict(config)
        return
    if isinstance(stages, Mapping):
        for label, entry in stages.items():
            if not isinstance(entry, Mapping):
                raise ValueError("Each entry in sglang.stages must be a mapping")
            stage_cfg = dict(config)
            stage_cfg.update(entry)
            stage_cfg.setdefault("label", label)
            yield str(label), stage_cfg
        return
    if isinstance(stages, Sequence):
        for idx, entry in enumerate(stages):
            if not isinstance(entry, Mapping):
                raise ValueError("Each entry in sglang.stages must be a mapping")
            label = entry.get("label") or f"stage-{idx}"
            stage_cfg = dict(config)
            stage_cfg.update(entry)
            stage_cfg.setdefault("label", label)
            yield str(label), stage_cfg
        return
    raise ValueError("sglang.stages must be a mapping or sequence if provided")


def _resolve_server_value(
    stage_cfg: Mapping[str, Any],
    base_cfg: Mapping[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    if key in stage_cfg:
        return stage_cfg[key]
    if key in base_cfg:
        return base_cfg[key]
    return default


def _extract_run(run_cfg: Any, idx: int) -> tuple[list[str], Mapping[str, Any] | None, str]:
    if isinstance(run_cfg, Mapping):
        args = run_cfg.get("args", [])
        if args and not isinstance(args, Sequence):
            raise ValueError("batch.runs[].args must be a sequence")
        env = run_cfg.get("env")
        label = run_cfg.get("label") or f"run-{idx}"
        return [str(arg) for arg in args], env, label
    if isinstance(run_cfg, Sequence):
        return [str(arg) for arg in run_cfg], None, f"run-{idx}"
    raise ValueError("Each entry in batch.runs must be a mapping or sequence")


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"none", "null"}:
            return None
    return value


def _to_plain_dict(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: _to_plain_dict(value) for key, value in data.items()}
    return _normalize_value(data)


def _apply_cli_overrides(
    base_cfg: Mapping[str, Any],
    args_list: Sequence[str],
) -> tuple[dict[str, Any], str | None]:
    applied_cfg = deepcopy(dict(base_cfg))
    if not args_list:
        return applied_cfg, None

    overrides_raw = _parse_args_to_nested_dict(list(args_list))
    overrides = _to_plain_dict(overrides_raw)

    suffix_override = None
    if isinstance(overrides, dict) and "suffix" in overrides:
        suffix_override = overrides.pop("suffix")
        if suffix_override is not None:
            suffix_override = str(suffix_override)

    merge_nested_dicts(applied_cfg, overrides if isinstance(overrides, dict) else {})
    return applied_cfg, suffix_override


def _prepare_run_cleanup(
    config_path: str,
    extra_args: Sequence[Any] | None,
    *,
    dry_run: bool,
    label: str | None = None,
) -> list[str] | None:
    cmd = [
        sys.executable,
        "-m",
        "sweagent.run.run",
        "run-batch",
        "--config",
        config_path,
    ]
    if extra_args:
        cmd.extend(str(arg) for arg in extra_args)

    if dry_run:
        prefix = "[dry-run] "
        if label:
            prefix += f"[{label}] "
        print(prefix + " ".join(cmd))
        return None

    if label:
        print(f"[run] Starting batch '{label}'")
    return cmd


def run_batch(
    config_path: str,
    extra_args: Sequence[Any] | None,
    *,
    env: Mapping[str, Any] | None,
    dry_run: bool,
    label: str | None = None,
) -> int:
    cmd = _prepare_run_cleanup(config_path, extra_args, dry_run=dry_run, label=label)
    if dry_run or cmd is None:
        return 0
    completed = subprocess.run(cmd, env=_merge_env(env))
    return completed.returncode


def terminate_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=15)
    except Exception:  # noqa: BLE001
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()


def main() -> int:
    args = parse_args()
    raw_config = load_config(args.config_file)

    server_cfg = ensure_mapping(raw_config.get("sglang"), "sglang")
    batch_cfg = ensure_mapping(raw_config.get("batch"), "batch")

    sweagent_config = str(batch_cfg.get("sweagent_config", "config/sglang_local.yaml"))
    batch_args = batch_cfg.get("args", [])
    runs = batch_cfg.get("runs")
    allowed_runs: set[str] | None = None
    if batch_args and not isinstance(batch_args, Sequence):
        raise ValueError("batch.args must be a sequence if provided")
    if runs is not None and not isinstance(runs, Sequence):
        raise ValueError("batch.runs must be a sequence if provided")
    if args.only_runs:
        allowed_runs = {str(item) for item in args.only_runs}
        if not allowed_runs:
            allowed_runs = None

    server_cmd = build_sglang_command(server_cfg)
    print("Starting sglang server:", " ".join(server_cmd))

    if args.dry_run:
        print("[dry-run] would wait for server and run sweagent batch")
        if runs:
            matched = False
            for idx, run_cfg in enumerate(runs):
                run_args, run_env, label = _extract_run(run_cfg, idx)
                if allowed_runs and label not in allowed_runs and str(idx) not in allowed_runs and str(idx + 1) not in allowed_runs:
                    continue
                matched = True
                combined_args = list(batch_args) + run_args
                combined_env = _merge_two_envs(batch_cfg.get("env"), run_env)
                run_batch(sweagent_config, combined_args, env=combined_env, dry_run=True, label=label)
            if allowed_runs and not matched:
                print(f"[dry-run] No runs matched --only-runs {sorted(allowed_runs)}")
        else:
            run_batch(sweagent_config, batch_args, env=batch_cfg.get("env"), dry_run=True)
        return 0

    server_proc = subprocess.Popen(server_cmd, env=_merge_env(server_cfg.get("env")))
    try:
        connect_host = str(server_cfg.get("connect_host", server_cfg.get("host", "127.0.0.1")))
        port = int(server_cfg.get("port", 30000))
        timeout = int(server_cfg.get("wait_timeout", raw_config.get("wait_timeout", 180)))
        health_path = str(server_cfg.get("health_path", "/v1/models"))
        wait_for_server(connect_host, port, timeout, path=health_path)
        print("sglang server is ready. Running sweagent batch...")
        exit_code = 0
        if runs:
            matched = False
            for idx, run_cfg in enumerate(runs):
                run_args, run_env, label = _extract_run(run_cfg, idx)
                if allowed_runs and label not in allowed_runs and str(idx) not in allowed_runs and str(idx + 1) not in allowed_runs:
                    continue
                matched = True
                combined_args = list(batch_args) + run_args
                combined_env = _merge_two_envs(batch_cfg.get("env"), run_env)
                exit_code = run_batch(
                    sweagent_config,
                    combined_args,
                    env=combined_env,
                    dry_run=False,
                    label=label,
                )
                if exit_code != 0:
                    print(f"Batch '{label}' failed with exit code {exit_code}; stopping remaining runs")
                    break
            if allowed_runs and not matched:
                print(f"No runs matched --only-runs {sorted(allowed_runs)}")
                exit_code = 0
        else:
            exit_code = run_batch(
                sweagent_config,
                batch_args,
                env=batch_cfg.get("env"),
                dry_run=False,
            )
        return exit_code
    finally:
        print("Stopping sglang server...")
        terminate_process(server_proc)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
