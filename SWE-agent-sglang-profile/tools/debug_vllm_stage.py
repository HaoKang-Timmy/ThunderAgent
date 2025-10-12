#!/usr/bin/env python3
"""Deprecated wrapper redirecting to tools.debug_sglang_stage."""

from __future__ import annotations

import warnings

from tools.debug_sglang_stage import main as _main


def main() -> int:
    warnings.warn(
        "tools/debug_vllm_stage.py is deprecated; use tools/debug_sglang_stage.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
