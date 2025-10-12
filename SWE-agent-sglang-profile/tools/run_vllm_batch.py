#!/usr/bin/env python3
"""Deprecated wrapper redirecting to tools.run_sglang_batch."""

from __future__ import annotations

import warnings

from tools.run_sglang_batch import main as _main


def main() -> int:
    warnings.warn(
        "tools/run_vllm_batch.py is deprecated; use tools/run_sglang_batch.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
