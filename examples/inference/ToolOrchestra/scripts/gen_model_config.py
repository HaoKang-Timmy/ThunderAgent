#!/usr/bin/env python3
"""Generate model config for rollout experiments with 4 separate nodes.

Usage:
    # For ThunderReact (single router endpoint)
    python gen_model_config.py <orch_ip> <expert_1_ip> <expert_2_ip> <expert_3_ip> --scheduler thunderreact

    # For Baseline/Continuum (single router endpoint via vllm-router)
    python gen_model_config.py <orch_ip> <expert_1_ip> <expert_2_ip> <expert_3_ip> --scheduler baseline

    # Override orchestrator endpoints (useful for 5090 single-GPU runs):
    python gen_model_config.py 127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1 --orch-endpoints 127.0.0.1:1900
"""
import json
import os
import sys
import argparse


def _parse_orch_endpoints(spec: str):
    """
    Parse a comma-separated list of endpoints in the form:
      ip:port,ip:port,...
    Returns: list[dict] with keys ip_addr, port (as str)
    """
    endpoints = []
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid --orch-endpoints entry (expected ip:port): {item}")
        ip, port = item.rsplit(":", 1)
        ip = ip.strip()
        port = port.strip()
        if not ip or not port:
            raise ValueError(f"Invalid --orch-endpoints entry: {item}")
        endpoints.append({"ip_addr": ip, "port": port})
    if not endpoints:
        raise ValueError("No valid endpoints parsed from --orch-endpoints")
    return endpoints


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate model config for rollout experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Node Configuration:
  Node 1 (Orchestrator): Nemotron-Orchestrator-8B (DP=8)
    - ThunderReact: ThunderReact router on port 8000
    - Baseline/Continuum: vllm-router on port 8000

  Node 2 (Expert-1): openai/gpt-oss-20b (TP=2, DP=4)
    - Ports: 1910-1913

  Node 3 (Expert-2): Qwen/Qwen3-32B-FP8 (TP=4, DP=2)
    - Ports: 1904-1905

  Node 4 (Expert-3): Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 (TP=4, DP=2)
    - Ports: 1920-1921
"""
    )
    parser.add_argument("orch_ip", help="Orchestrator node IP")
    parser.add_argument("expert_1_ip", help="Expert-1 (gpt-oss-20b) node IP")
    parser.add_argument("expert_2_ip", help="Expert-2 (Qwen3-32B) node IP")
    parser.add_argument("expert_3_ip", help="Expert-3 (Qwen3-Next-80B) node IP")
    parser.add_argument(
        "--scheduler",
        choices=["baseline", "thunderreact", "continuum"],
        default="baseline",
        help="Scheduler type for orchestrator (default: baseline)"
    )
    parser.add_argument(
        "--orch-endpoints",
        default=None,
        help=(
            "Override orchestrator endpoints with a comma-separated list of ip:port entries "
            "(e.g., 127.0.0.1:1900 or 127.0.0.1:8000). "
            "If set, this takes precedence over --scheduler endpoint defaults."
        ),
    )
    parser.add_argument(
        "--orch-port",
        default=None,
        help=(
            "Convenience shorthand for --orch-endpoints <orch_ip>:<orch_port>. "
            "Example: --orch-port 1900 (equivalent to --orch-endpoints <orch_ip>:1900)."
        ),
    )
    parser.add_argument(
        "--output",
        default="model_config_rollout.json",
        help="Output file path (default: model_config_rollout.json)"
    )
    args = parser.parse_args()

    ckpt_dir = os.getenv("CKPT_DIR")
    if not ckpt_dir:
        print("ERROR: CKPT_DIR env var is required (path to Orchestrator-8B checkpoint)", file=sys.stderr)
        return 2

    # Orchestrator endpoints depend on scheduler type
    if args.orch_endpoints or args.orch_port:
        spec = args.orch_endpoints
        if args.orch_port:
            spec = f"{args.orch_ip}:{args.orch_port}"
        orch_endpoints = _parse_orch_endpoints(spec)
    else:
        # Default: route ALL orchestrator requests through a router on port 8000.
        # - ThunderReact: ThunderReact router
        # - Baseline: vllm-router (session routing)
        # - Continuum: vllm-router (session routing) + continuum backends
        orch_endpoints = [{"ip_addr": args.orch_ip, "port": "8000"}]

    config = {
        # Orchestrator-8B endpoints
        ckpt_dir: orch_endpoints,

        # Expert-1: gpt-oss-20b (TP=2, DP=4) → 4 endpoints on ports 1910-1913
        "openai/gpt-oss-20b": [
            {"ip_addr": args.expert_1_ip, "port": str(1910 + i)} for i in range(4)
        ],

        # Expert-2: Qwen3-32B-FP8 (TP=4, DP=2) → 2 endpoints on ports 1904-1905
        "Qwen/Qwen3-32B-FP8": [
            {"ip_addr": args.expert_2_ip, "port": str(1904 + i)} for i in range(2)
        ],

        # Expert-3: Qwen3-Next-80B (TP=4, DP=2) → 2 endpoints on ports 1920-1921
        "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8": [
            {"ip_addr": args.expert_3_ip, "port": str(1920 + i)} for i in range(2)
        ],

        "vllm_model_config_path": args.output,

        # OSS expert mapping for llm_utils.py
        "oss_expert_mapping": {
            "expert-1": "openai/gpt-oss-20b",
            "expert-2": "Qwen/Qwen3-32B-FP8",
            "expert-3": "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Generated {args.output}")
    print(f"  Scheduler: {args.scheduler}")
    print(f"  Orchestrator ({args.orch_ip}): {len(orch_endpoints)} endpoint(s)")
    print(f"  Expert-1 ({args.expert_1_ip}): gpt-oss-20b (ports 1910-1913)")
    print(f"  Expert-2 ({args.expert_2_ip}): Qwen3-32B-FP8 (ports 1904-1905)")
    print(f"  Expert-3 ({args.expert_3_ip}): Qwen3-Next-80B (ports 1920-1921)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
