#!/usr/bin/env python3
"""Compare results across experiment runs.

Given one or more tau2-bench output directories, summarize success rate by domain.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any


def load_results(output_dir: str) -> Dict[str, Any]:
    results: Dict[str, Any] = {"domains": {}}
    p = Path(output_dir)
    for f in p.glob("*.json"):
        if f.stem not in {"retail", "telecom", "airline"}:
            continue
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        sims = data.get("simulations", [])
        success = sum(1 for s in sims if (s.get("reward_info", {}) or {}).get("reward", 0) > 0)
        results["domains"][f.stem] = {
            "tasks": len(sims),
            "success": success,
            "success_rate": success / max(1, len(sims)),
        }
    return results


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: compare_experiments.py <output_dir1> [output_dir2 ...]", file=sys.stderr)
        return 2
    comparison: Dict[str, Any] = {}
    for d in sys.argv[1:]:
        comparison[Path(d).name] = load_results(d)
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


