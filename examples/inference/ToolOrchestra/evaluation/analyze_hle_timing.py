#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Analyze HLE structured logs (HLE_LOG_FILE) and generate timing/length distributions.

Outputs, for each metric:
- stats JSON (n/mean/std/min/max)
- 10-bin histogram with equal-width (linear) bins
- 10-bin histogram with log-scale (log-spaced) bins

Example:
  python analyze_hle_timing.py --log-dir logs/hle_smoke_c25 --out-prefix hle_smoke_c25 --bins 10
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Any

import numpy as np
import matplotlib.pyplot as plt


def _extract_float(regex: str, s: str) -> Optional[float]:
    m = re.search(regex, s)
    return float(m.group(1)) if m else None


def _extract_str(regex: str, s: str) -> Optional[str]:
    m = re.search(regex, s)
    return str(m.group(1)) if m else None


def _stats(values: list[float]) -> Optional[dict[str, Any]]:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return None
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _hist_linear(values: list[float], bins: int) -> Optional[dict[str, Any]]:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return None
    lo = float(a.min())
    hi = float(a.max())
    if hi == lo:
        hi = lo + 1e-9
    edges = np.linspace(lo, hi, int(bins) + 1)
    counts, _ = np.histogram(a, bins=edges)
    perc = (counts / a.size) * 100.0
    return {"edges": edges.tolist(), "counts": counts.tolist(), "percent": perc.tolist()}


def _hist_log(values: list[float], bins: int) -> Optional[dict[str, Any]]:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return None

    # Log bins require strictly positive edges; clamp non-positive values into the first bin.
    pos = a[a > 0]
    if pos.size == 0:
        return None
    lo = float(pos.min())
    hi = float(pos.max())
    if hi == lo:
        hi = lo * 10.0
    # Clamp to lo so zeros land in the first bin.
    a = np.clip(a, lo, None)
    edges = np.logspace(np.log10(lo), np.log10(hi), int(bins) + 1)
    counts, _ = np.histogram(a, bins=edges)
    perc = (counts / a.size) * 100.0
    return {"edges": edges.tolist(), "counts": counts.tolist(), "percent": perc.tolist()}


def _plot_hist_linear(values: list[float], bins: int, title: str, xlabel: str, out_png: Path) -> None:
    h = _hist_linear(values, bins)
    s = _stats(values)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)
    if not h or not s:
        ax.set_title(title)
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        return

    edges = np.asarray(h["edges"], dtype=float)
    perc = np.asarray(h["percent"], dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2.0
    width = (edges[1] - edges[0]) * 0.9
    bars = ax.bar(centers, perc, width=width, align="center")
    for bar, pct in zip(bars, perc):
        if pct <= 0:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"{title} (n={s['n']}, mean={s['mean']:.2f}, std={s['std']:.2f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("%")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_hist_log(values: list[float], bins: int, title: str, xlabel: str, out_png: Path) -> None:
    h = _hist_log(values, bins)
    s = _stats(values)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)
    if not h or not s:
        ax.set_title(title)
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        return

    edges = np.asarray(h["edges"], dtype=float)
    perc = np.asarray(h["percent"], dtype=float)
    widths = np.diff(edges)
    # Draw exact bin spans.
    bars = ax.bar(edges[:-1], perc, width=widths, align="edge")

    # Label each bar at geometric center of its bin (looks right on log x-scale).
    centers = np.sqrt(edges[:-1] * edges[1:])
    for center, pct in zip(centers, perc):
        if pct <= 0:
            continue
        ax.text(center, pct, f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xscale("log")
    ax.set_title(f"{title} (n={s['n']}, mean={s['mean']:.2f}, std={s['std']:.2f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("%")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--log-path", type=str, help="Path to a single HLE log file (e.g. hle.log)")
    src.add_argument("--log-dir", type=str, help="Directory containing HLE log file(s)")
    p.add_argument("--log-glob", type=str, default="hle*.log", help="Glob inside --log-dir (default: hle*.log)")
    p.add_argument(
        "--all-runs",
        action="store_true",
        help=(
            "Analyze all runs found in the provided log file(s). "
            "By default, if a run_start marker exists, we only analyze the latest run segment."
        ),
    )
    p.add_argument("--out-dir", type=str, default=None, help="Output directory (default: same as log file/dir)")
    p.add_argument("--out-prefix", type=str, default="hle_profile", help="Output filename prefix")
    p.add_argument("--bins", type=int, default=10, help="Number of bins for histograms (default: 10)")
    args = p.parse_args()

    log_paths: list[Path] = []
    if args.log_path:
        lp = Path(args.log_path).expanduser().resolve()
        if not lp.exists():
            raise SystemExit(f"log path not found: {lp}")
        log_paths = [lp]
        default_out = lp.parent
        source_label = lp.name
    else:
        ld = Path(args.log_dir).expanduser().resolve()
        if not ld.exists():
            raise SystemExit(f"log dir not found: {ld}")
        log_paths = sorted([p for p in ld.glob(args.log_glob) if p.is_file()])
        if not log_paths:
            raise SystemExit(f"no logs matched in {ld} with glob={args.log_glob}")
        default_out = ld
        source_label = ld.name

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read all logs (aggregate)
    lines: list[str] = []
    for lp in log_paths:
        lines.extend(lp.read_text(encoding="utf-8", errors="replace").splitlines())

    if not args.all_runs:
        # If multiple runs are appended into the same log file, only analyze the latest run.
        # (We emit a type=run_start marker at the beginning of each eval_hle_local.py execution.)
        last_run_start_idx: Optional[int] = None
        for i, line in enumerate(lines):
            if "type=run_start" in line:
                last_run_start_idx = i
        if last_run_start_idx is not None:
            lines = lines[last_run_start_idx:]

    # Tool call metrics
    tool_all_ms: list[float] = []
    tool_by_name_ms: dict[str, list[float]] = {
        "enhance_reasoning": [],
        "search": [],
        "answer": [],
    }

    # vLLM metrics (ToolOrchestra orchestrator calls)
    vllm_infer_ms: list[float] = []
    vllm_prefill_ms: list[float] = []
    vllm_decode_ms: list[float] = []
    vllm_prefill_len: list[float] = []
    vllm_decode_len: list[float] = []
    vllm_total_len: list[float] = []

    for line in lines:
        if "type=tool_call" in line:
            dur = _extract_float(r"duration_ms=([0-9]+(?:\.[0-9]+)?)", line)
            if dur is not None:
                tool_all_ms.append(dur)
            tool_name = _extract_str(r"tool=([^\s]+)", line)
            if tool_name and dur is not None and tool_name in tool_by_name_ms:
                tool_by_name_ms[tool_name].append(dur)

        if "type=llm_call" in line and "toolorchestra_vllm_infer_ms=" in line:
            v = _extract_float(r"toolorchestra_vllm_infer_ms=([0-9]+(?:\.[0-9]+)?)", line)
            if v is not None:
                vllm_infer_ms.append(v)
            v = _extract_float(r"toolorchestra_vllm_prefill_ms=([0-9]+(?:\.[0-9]+)?)", line)
            if v is not None:
                vllm_prefill_ms.append(v)
            v = _extract_float(r"toolorchestra_vllm_decode_ms=([0-9]+(?:\.[0-9]+)?)", line)
            if v is not None:
                vllm_decode_ms.append(v)
            prefill = _extract_float(r"toolorchestra_vllm_prefill_len=([0-9]+(?:\.[0-9]+)?)", line)
            if prefill is not None:
                vllm_prefill_len.append(prefill)
            decode = _extract_float(r"toolorchestra_vllm_decode_len=([0-9]+(?:\.[0-9]+)?)", line)
            if decode is not None:
                vllm_decode_len.append(decode)
            if prefill is not None and decode is not None:
                vllm_total_len.append(float(prefill) + float(decode))

    summary: dict[str, Any] = {}

    def _add_metric(key: str, values: list[float], unit: str, title: str) -> None:
        summary[key] = {
            "unit": unit,
            "stats": _stats(values),
            "hist_linear": _hist_linear(values, args.bins),
            "hist_log": _hist_log(values, args.bins),
        }
        _plot_hist_linear(values, args.bins, title=title, xlabel=unit, out_png=out_dir / f"{args.out_prefix}_{key}_linear.png")
        _plot_hist_log(values, args.bins, title=title, xlabel=unit, out_png=out_dir / f"{args.out_prefix}_{key}_log.png")

    _add_metric("tool_all", tool_all_ms, "ms", "HLE tool calls (all) duration_ms")
    _add_metric("tool_enhance_reasoning", tool_by_name_ms["enhance_reasoning"], "ms", "HLE tool enhance_reasoning duration_ms")
    _add_metric("tool_search", tool_by_name_ms["search"], "ms", "HLE tool search duration_ms")
    _add_metric("tool_answer", tool_by_name_ms["answer"], "ms", "HLE tool answer duration_ms")

    _add_metric("vllm_infer_ms", vllm_infer_ms, "ms", "ToolOrchestra vLLM infer_ms (end-to-end)")
    _add_metric("vllm_prefill_ms", vllm_prefill_ms, "ms", "ToolOrchestra vLLM prefill_ms (TTFT)")
    _add_metric("vllm_decode_ms", vllm_decode_ms, "ms", "ToolOrchestra vLLM decode_ms")
    _add_metric("vllm_prefill_len", vllm_prefill_len, "tok", "ToolOrchestra vLLM prefill_len (prompt tokens)")
    _add_metric("vllm_decode_len", vllm_decode_len, "tok", "ToolOrchestra vLLM decode_len (completion tokens)")
    _add_metric("vllm_total_len", vllm_total_len, "tok", "ToolOrchestra vLLM total_len (prompt+completion tokens)")

    json_path = out_dir / f"{args.out_prefix}_stats.json"
    json_path.write_text(json.dumps({"source": source_label, "metrics": summary}, indent=2), encoding="utf-8")

    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


