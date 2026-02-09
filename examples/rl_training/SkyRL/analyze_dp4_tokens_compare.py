#!/usr/bin/env python3
"""Compare per-step rollout tokens/sec for DP=4 default vs TR(atw=0.1).

Inputs are SLURM job IDs. The script reads slurm-<JOBID>.err and prints an
aligned 3-column terminal table:
  Step | <default-label> tokens/sec | <tr-label> tokens/sec
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

TRAJS_PER_STEP_DEFAULT = 396  # 99 prompts x 4 samples
ANSI_RE = re.compile(r"\x1b\[\d+(?:;\d+)*m")
STEP_START_RE = re.compile(r"Started: 'step'")
GEN_FINISH_RE = re.compile(r"Finished: 'generate', time cost: ([\d.]+)s")
REWARDS_RE = re.compile(
    r"avg_final_rewards:\s*([\d.eE+-]+),\s*avg_response_length:\s*([\d.eE+-]+)"
)


@dataclass
class StepInfo:
    step_num: int
    gen_duration: Optional[float] = None
    avg_resp_len: Optional[float] = None

    def tokens_per_sec(self, trajs_per_step: int) -> Optional[float]:
        if self.gen_duration is None or self.avg_resp_len is None or self.gen_duration <= 0:
            return None
        return (trajs_per_step * self.avg_resp_len) / self.gen_duration


def strip_ansi(line: str) -> str:
    return ANSI_RE.sub("", line)


def parse_slurm_err(path: str) -> List[StepInfo]:
    steps: List[StepInfo] = []
    current: Optional[StepInfo] = None
    step_num = 0

    with open(path, "r", errors="replace") as f:
        for raw in f:
            line = strip_ansi(raw)

            if STEP_START_RE.search(line):
                step_num += 1
                current = StepInfo(step_num=step_num)
                steps.append(current)
                continue

            m = GEN_FINISH_RE.search(line)
            if m and current:
                current.gen_duration = float(m.group(1))
                continue

            m = REWARDS_RE.search(line)
            if m and current:
                current.avg_resp_len = float(m.group(2))
                continue

    return steps


def steps_to_tps(steps: List[StepInfo], trajs_per_step: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for s in steps:
        tps = s.tokens_per_sec(trajs_per_step)
        if tps is not None:
            out[s.step_num] = tps
    return out


def fmt(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def print_aligned_table(headers: List[str], rows: List[List[str]]) -> None:
    if not headers:
        return
    n_cols = len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for i in range(min(n_cols, len(row))):
            widths[i] = max(widths[i], len(str(row[i])))

    # Step column left-aligned, numeric columns right-aligned.
    aligns = ["left"] + ["right"] * (n_cols - 1)

    def _pad(text: str, width: int, align: str) -> str:
        return text.ljust(width) if align == "left" else text.rjust(width)

    print(" | ".join(_pad(headers[i], widths[i], "left") for i in range(n_cols)))
    print("-+-".join("-" * widths[i] for i in range(n_cols)))
    for row in rows:
        print(
            " | ".join(
                _pad(str(row[i]) if i < len(row) else "", widths[i], aligns[i]) for i in range(n_cols)
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--default-job", required=True, help="SLURM job ID for DP=4 default run")
    parser.add_argument("--tr-job", required=True, help="SLURM job ID for DP=4 TR(atw=0.1) run")
    parser.add_argument(
        "--base-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory containing slurm-<JOBID>.err (default: script directory)",
    )
    parser.add_argument(
        "--trajs-per-step",
        type=int,
        default=TRAJS_PER_STEP_DEFAULT,
        help=f"Trajectories per rollout step (default: {TRAJS_PER_STEP_DEFAULT})",
    )
    parser.add_argument("--default-label", default="dp4_default")
    parser.add_argument("--tr-label", default="dp4_tr_atw01")
    args = parser.parse_args()

    default_path = os.path.join(args.base_dir, f"slurm-{args.default_job}.err")
    tr_path = os.path.join(args.base_dir, f"slurm-{args.tr_job}.err")

    missing = [p for p in [default_path, tr_path] if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"ERROR: missing log file: {p}")
        raise SystemExit(1)

    default_steps = parse_slurm_err(default_path)
    tr_steps = parse_slurm_err(tr_path)
    default_tps = steps_to_tps(default_steps, args.trajs_per_step)
    tr_tps = steps_to_tps(tr_steps, args.trajs_per_step)

    all_step_ids = sorted(set(default_tps.keys()) | set(tr_tps.keys()))

    headers = [
        "Step",
        f"{args.default_label} tokens/sec",
        f"{args.tr_label} tokens/sec",
    ]
    rows: List[List[str]] = []
    for step_id in all_step_ids:
        rows.append(
            [
                str(step_id),
                fmt(default_tps.get(step_id)),
                fmt(tr_tps.get(step_id)),
            ]
        )

    print_aligned_table(headers, rows)


if __name__ == "__main__":
    main()
