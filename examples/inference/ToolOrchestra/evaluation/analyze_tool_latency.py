#!/usr/bin/env python3
"""
Analyze tool latency statistics from HLE/FRAMES evaluation outputs.

Usage:
    python analyze_tool_latency.py <output_dir>
    python analyze_tool_latency.py outputs_oss_hle_full_new
    python analyze_tool_latency.py outputs_oss_frames_full_new

With plots:
    python analyze_tool_latency.py outputs_oss_hle_full_new --plots --out-dir logs_oss/hle_full_new_latency_plots
"""

import json
import sys
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt  # type: ignore[import-not-found]


def collect_latencies(output_dir: Path) -> dict:
    """Collect latency data from all completed task JSON files."""
    latencies = {
        "search": {"e2e": [], "expert": []},
        "answer": {"e2e": [], "expert": []},
        "enhance_reasoning": {"e2e": [], "expert": []}
    }

    for f in output_dir.glob("*.json"):
        try:
            d = json.load(open(f))
            for turn, responses in d.get("all_tool_responses", {}).items():
                for r in responses:
                    tool = r.get("tool")
                    if tool == "search":
                        query_ms = r.get("_query_llm_ms", 0)
                        retrieval_ms = r.get("_retrieval_ms", 0)
                        if query_ms and retrieval_ms:
                            latencies["search"]["expert"].append(query_ms)
                            latencies["search"]["e2e"].append(query_ms + retrieval_ms)
                    elif tool == "answer":
                        expert_ms = r.get("_expert_ms", 0)
                        judge_ms = r.get("_judge_ms", 0)
                        if expert_ms:
                            latencies["answer"]["expert"].append(expert_ms)
                            latencies["answer"]["e2e"].append(expert_ms + (judge_ms or 0))
                    elif tool == "enhance_reasoning":
                        llm_ms = r.get("_llm_ms", 0)
                        exec_ms = r.get("_exec_ms", 0)
                        if llm_ms:
                            latencies["enhance_reasoning"]["expert"].append(llm_ms)
                            latencies["enhance_reasoning"]["e2e"].append(llm_ms + (exec_ms or 0))
        except Exception:
            pass

    return latencies


def compute_stats(data: list) -> dict | None:
    """Compute statistics for a list of latency values (in ms)."""
    if not data:
        return None
    arr = np.array(data) / 1000  # Convert to seconds
    return {
        "count": len(arr),
        "mean": np.mean(arr),
        "std": np.std(arr),
        "min": np.min(arr),
        "max": np.max(arr),
        "p10": np.percentile(arr, 10),
        "p25": np.percentile(arr, 25),
        "p40": np.percentile(arr, 40),
        "p50": np.percentile(arr, 50),
        "p90": np.percentile(arr, 90),
        "p95": np.percentile(arr, 95),
        "p99": np.percentile(arr, 99),
        "lt5s": np.mean(arr < 5) * 100
    }


def print_table(title: str, latencies: dict, data_type: str):
    """Print a markdown table for the given latency data."""
    print(f"### {title}\n")
    print("| Tool | Count | Mean | Std | Min | Max | P10 | P25 | P40 | P50 | P90 | P95 | P99 | <5s% |")
    print("|------|-------|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|")
    for tool in ["search", "answer", "enhance_reasoning"]:
        stats = compute_stats(latencies[tool][data_type])
        if stats:
            print(f"| {tool} | {stats['count']} | {stats['mean']:.2f}s | {stats['std']:.2f}s | "
                  f"{stats['min']:.2f}s | {stats['max']:.2f}s | {stats['p10']:.2f}s | {stats['p25']:.2f}s | "
                  f"{stats['p40']:.2f}s | {stats['p50']:.2f}s | {stats['p90']:.2f}s | {stats['p95']:.2f}s | "
                  f"{stats['p99']:.2f}s | {stats['lt5s']:.1f}% |")
        else:
            print(f"| {tool} | 0 | - | - | - | - | - | - | - | - | - | - | - | - |")

def _plot_hist_percent(values_s: np.ndarray, *, bins: int, title: str, xlabel: str, out_png: Path) -> None:
    """
    Plot a linear histogram with y-axis as percent.

    Requirements (per user feedback):
    - Bars must align to bin boundaries (use align='edge' and full bin widths).
    - Denote bin size in the title.
    """
    if values_s.size == 0:
        return
    counts, edges = np.histogram(values_s, bins=bins)
    perc = counts / max(values_s.size, 1) * 100.0

    fig, ax = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)
    widths = np.diff(edges)
    bars = ax.bar(edges[:-1], perc, width=widths, align="edge", edgecolor="black", linewidth=0.3, alpha=0.85)
    for bar, p in zip(bars, perc):
        if p <= 0:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{p:.1f}%", ha="center", va="bottom", fontsize=8)

    # Bin size (constant for numpy's equal-width bins), in seconds
    bin_w = float(widths[0]) if widths.size else 0.0
    ax.set_title(f"{title}  |  bins={bins}, bin_width={bin_w:.2f}s")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("%")
    ax.grid(axis="y", alpha=0.25)
    # Draw subtle vertical grid lines at bin edges to emphasize boundary alignment.
    for e in edges:
        ax.axvline(float(e), color="gray", linewidth=0.5, alpha=0.12, zorder=0)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def write_plots(output_dir: Path, latencies: dict, *, out_dir: Path, prefix: str, bins: int) -> None:
    """Write overall + per-tool latency distribution plots (end-to-end and expert LLM)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def as_seconds(tool: str, kind: str) -> np.ndarray:
        a = np.asarray(latencies[tool][kind], dtype=float)
        if a.size == 0:
            return a
        return a / 1000.0

    def stats_line(a_s: np.ndarray) -> str:
        if a_s.size == 0:
            return "n=0"
        mean = float(a_s.mean())
        med = float(np.percentile(a_s, 50))
        p90 = float(np.percentile(a_s, 90))
        return f"n={a_s.size}, mean={mean:.2f}s, median={med:.2f}s, p90={p90:.2f}s"

    # overall
    for kind in ["e2e", "expert"]:
        all_s = np.concatenate([as_seconds(t, kind) for t in ["search", "answer", "enhance_reasoning"] if as_seconds(t, kind).size > 0])
        _plot_hist_percent(
            all_s,
            bins=bins,
            title=f"{prefix}: ALL tools ({kind})  |  {stats_line(all_s)}",
            xlabel="seconds",
            out_png=out_dir / f"{prefix}_{kind}_all_tools.png",
        )

        for tool in ["search", "answer", "enhance_reasoning"]:
            a_s = as_seconds(tool, kind)
            _plot_hist_percent(
                a_s,
                bins=bins,
                title=f"{prefix}: tool={tool} ({kind})  |  {stats_line(a_s)}",
                xlabel="seconds",
                out_png=out_dir / f"{prefix}_{kind}_{tool}.png",
            )


def main():
    p = argparse.ArgumentParser(description="Analyze tool latency statistics from eval output JSONs")
    p.add_argument("output_dir", type=str, help="Output directory containing per-task JSON files")
    p.add_argument("--plots", action="store_true", help="Also generate histogram PNGs (overall + per-tool)")
    p.add_argument("--out-dir", type=str, default=None, help="Directory to write plots into (default: <output_dir>/latency_plots)")
    p.add_argument("--prefix", type=str, default=None, help="Filename/title prefix (default: output_dir name)")
    p.add_argument("--bins", type=int, default=10, help="Histogram bin count (default: 10)")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: Directory {output_dir} does not exist")
        sys.exit(1)

    completed = len(list(output_dir.glob("*.json")))
    print(f"**Completed tasks: {completed}**\n")

    latencies = collect_latencies(output_dir)

    print_table("End-to-End Latency (seconds)", latencies, "e2e")
    print()
    print_table("Expert LLM Call Latency (seconds)", latencies, "expert")

    if args.plots:
        out_dir = Path(args.out_dir) if args.out_dir else (output_dir / "latency_plots")
        prefix = args.prefix if args.prefix else output_dir.name
        write_plots(output_dir, latencies, out_dir=out_dir, prefix=prefix, bins=int(args.bins))
        print(f"\nWrote plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
