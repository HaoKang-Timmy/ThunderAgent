#!/usr/bin/env python3
"""
Check whether an HLE jsonl file contains duplicate example IDs.

Usage:
  python check_hle_duplicate_ids.py --path evaluation/hle.jsonl

Notes:
- Streams line-by-line (safe for large jsonl files).
- Expects each line to be a JSON object containing an "id" field.
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Tuple


def main() -> int:
    parser = argparse.ArgumentParser(description="Check duplicate `id` values in an HLE jsonl file.")
    parser.add_argument(
        "--path",
        type=str,
        default="hle.jsonl",
        help="Path to HLE jsonl file (default: hle.jsonl; typically evaluation/hle.jsonl when run from evaluation/).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Max line-number samples to show per duplicated id (default: 5).",
    )
    args = parser.parse_args()

    counts: Dict[str, int] = defaultdict(int)
    samples: Dict[str, List[int]] = defaultdict(list)
    missing_id_lines: List[int] = []
    json_error_lines: List[Tuple[int, str]] = []

    total_lines = 0
    with open(args.path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                json_error_lines.append((lineno, str(e)))
                continue
            ex_id = obj.get("id")
            if not ex_id:
                missing_id_lines.append(lineno)
                continue
            ex_id = str(ex_id)
            counts[ex_id] += 1
            if len(samples[ex_id]) < max(1, args.max_samples):
                samples[ex_id].append(lineno)

    duplicated = {k: v for k, v in counts.items() if v > 1}
    unique_ids = len(counts)

    print("=" * 72)
    print("HLE jsonl duplicate-id check")
    print("=" * 72)
    print(f"File: {args.path}")
    print(f"Total non-empty JSONL lines read: {total_lines}")
    print(f"Unique ids: {unique_ids}")
    print(f"Duplicate ids: {len(duplicated)}")
    print(f"Missing/empty id lines: {len(missing_id_lines)}")
    print(f"JSON parse errors: {len(json_error_lines)}")

    if json_error_lines:
        print("\nJSON parse error samples (line -> error):")
        for lineno, err in json_error_lines[: min(10, len(json_error_lines))]:
            print(f"  - L{lineno}: {err}")
        if len(json_error_lines) > 10:
            print(f"  ... (+{len(json_error_lines) - 10} more)")

    if missing_id_lines:
        print("\nMissing/empty `id` line samples:")
        for lineno in missing_id_lines[: min(20, len(missing_id_lines))]:
            print(f"  - L{lineno}")
        if len(missing_id_lines) > 20:
            print(f"  ... (+{len(missing_id_lines) - 20} more)")

    if duplicated:
        print("\nDuplicate ids (id -> count, sample line numbers):")
        # Show largest duplicates first
        for ex_id, cnt in sorted(duplicated.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  - {ex_id}: {cnt}  lines={samples.get(ex_id, [])}")
        return 2

    print("\n✓ No duplicate ids found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


