#!/usr/bin/env python3
"""Run a fixed sequence of mini-extra swebench runs, then move outputs.

This script runs the following steps sequentially:
- mini-extra swebench ... --workers {N} --output ./continuum_{N}
- move ./continuum_{N} to /mnt/shared/data

It stops immediately on any failure.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


MODEL_PATH = Path("/mnt/shared/models/GLM-4.6-FP8")
DEST_DIR = Path("/mnt/shared/data")
OUTPUT_SIZES = [32, 48, 64, 96, 128]


def run_cmd(argv: list[str]) -> None:
    printable = " ".join(argv)
    print(f"\n$ {printable}", flush=True)
    subprocess.run(argv, check=True)


def main() -> int:
    if not DEST_DIR.exists() or not DEST_DIR.is_dir():
        print(f"ERROR: destination dir not found: {DEST_DIR}", file=sys.stderr)
        return 2

    if not MODEL_PATH.exists():
        print(f"ERROR: model path not found: {MODEL_PATH}", file=sys.stderr)
        return 2

    for n in OUTPUT_SIZES:
        out_dir = Path(f"./continuum_{n}").resolve()

        # Clean up leftover output if present to avoid confusing partial results.
        if out_dir.exists():
            print(f"ERROR: output already exists: {out_dir}", file=sys.stderr)
            print("Refusing to overwrite. Please remove it first.", file=sys.stderr)
            return 2

        run_cmd(
            [
                "mini-extra",
                "swebench",
                "--model-class",
                "vllm",
                "--model",
                str(MODEL_PATH),
                "--subset",
                "lite",
                "--split",
                "test",
                "--workers",
                str(n),
                "--output",
                str(out_dir),
            ]
        )

        print(f"\n$ mv {out_dir} {DEST_DIR}", flush=True)
        # Move into DEST_DIR keeping the same basename.
        shutil.move(str(out_dir), str(DEST_DIR))

    print("\nAll runs completed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: command failed with exit code {e.returncode}", file=sys.stderr)
        return_code = e.returncode if isinstance(e.returncode, int) else 1
        raise SystemExit(return_code)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
