#!/usr/bin/env python3
"""Plot per-step average prefill/decode token usage for multiple run directories."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate prefix_cache_metrics.jsonl files for multiple runs and "
            "plot per-step average prefill/decode token counts (stacked) up to a "
            "given number of steps."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing run subdirectories (e.g. bs4, bs32, ...).",
    )
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        default=["bs4", "bs32"],
        help="Run directory names to include (default: %(default)s).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Only aggregate the first N steps (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory to write output figures (default: %(default)s).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
        help="DPI for saved figures (default: %(default)s).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-run aggregation details.",
    )
    return parser.parse_args()


def _iter_metrics_files(run_dir: Path) -> Iterable[Path]:
    for path in sorted(run_dir.rglob("prefix_cache_metrics.jsonl")):
        if path.is_file():
            yield path


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_instance_steps(metrics_path: Path, *, max_steps: int) -> Dict[int, Tuple[float, float]]:
    """Return step -> (prefill_tokens, decode_tokens) for a single instance."""
    step_tokens: Dict[int, Tuple[float, float]] = {}
    previous_input_tokens: float | None = None

    with metrics_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            raw_step = record.get("step")
            try:
                step = int(raw_step)
            except (TypeError, ValueError):
                continue
            if step <= 0 or step > max_steps:
                previous_input_tokens = (
                    _safe_float(record.get("input_tokens")) if step == 0 else previous_input_tokens
                )
                continue

            input_tokens = _safe_float(record.get("input_tokens"))
            output_tokens = _safe_float(record.get("output_tokens")) or 0.0
            if input_tokens is None:
                # Without input token counts we cannot derive per-step prefill usage.
                previous_input_tokens = None
                continue

            if previous_input_tokens is None:
                prefill_tokens = max(input_tokens, 0.0)
            else:
                prefill_tokens = max(input_tokens - previous_input_tokens, 0.0)

            step_tokens[step] = (prefill_tokens, max(output_tokens, 0.0))
            previous_input_tokens = input_tokens

    return step_tokens


def _aggregate_run(run_dir: Path, *, max_steps: int, verbose: bool = False) -> dict[int, dict[str, float]]:
    """Aggregate per-step totals and counts for a run."""
    totals: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    instance_counts: dict[int, dict[str, int]] = defaultdict(lambda: {"prefill": 0, "decode": 0})
    instance_total = 0

    for metrics_file in _iter_metrics_files(run_dir):
        instance_total += 1
        per_step = _load_instance_steps(metrics_file, max_steps=max_steps)
        if not per_step:
            continue
        for step, (prefill_tokens, decode_tokens) in per_step.items():
            totals[step]["prefill"] += prefill_tokens
            totals[step]["decode"] += decode_tokens
            instance_counts[step]["prefill"] += 1
            instance_counts[step]["decode"] += 1

    averages: dict[int, dict[str, float]] = {}
    for step in sorted(totals):
        prefill_count = max(instance_counts[step]["prefill"], 1)
        decode_count = max(instance_counts[step]["decode"], 1)
        averages[step] = {
            "prefill_avg": totals[step]["prefill"] / prefill_count,
            "decode_avg": totals[step]["decode"] / decode_count,
            "prefill_count": instance_counts[step]["prefill"],
            "decode_count": instance_counts[step]["decode"],
        }

    if verbose:
        observed_steps = len(averages)
        print(
            f"[run] {run_dir}: {instance_total} instances scanned, "
            f"{observed_steps} steps with data (max step {max(averages, default=0)})",
        )
    return averages


def _plot_run(
    run_name: str,
    per_step: dict[int, dict[str, float]],
    *,
    output_dir: Path,
    dpi: int,
) -> None:
    if not per_step:
        print(f"[warn] No data available for run '{run_name}', skipping plot.")
        return

    steps = sorted(per_step)
    prefill = [per_step[step]["prefill_avg"] for step in steps]
    decode = [per_step[step]["decode_avg"] for step in steps]

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f"prefill_decode_tokens_{run_name}.png"

    plt.figure(figsize=(max(8, len(steps) * 0.35), 5))
    plt.bar(steps, prefill, color="#5DADE2", label="Prefill tokens", width=0.9, align="edge")
    plt.bar(steps, decode, bottom=prefill, color="#58D68D", label="Decode tokens", width=0.9, align="edge")
    plt.xlabel("Step")
    plt.ylabel("Average tokens per step")
    plt.title(f"Average Prefill/Decode Tokens per Step ({run_name})")
    plt.xlim(min(steps) - 0.1, max(steps) + 0.9)
    if len(steps) > 40:
        plt.xticks(steps[:: max(1, len(steps) // 40)])
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=dpi)
    plt.close()
    print(f"[info] Saved plot for '{run_name}' to {figure_path}")


def main() -> None:
    args = parse_args()
    root = args.root
    if not root.is_dir():
        raise SystemExit(f"Root directory '{root}' does not exist or is not a directory.")

    for run_name in args.runs:
        run_dir = root / run_name
        if not run_dir.is_dir():
            print(f"[warn] Run directory '{run_dir}' missing; skipping.")
            continue
        per_step = _aggregate_run(run_dir, max_steps=args.max_steps, verbose=args.verbose)
        _plot_run(run_name, per_step, output_dir=args.output_dir, dpi=args.dpi)


if __name__ == "__main__":
    main()
