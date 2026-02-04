#!/usr/bin/env python3
"""
Probe vLLM streaming usage fields with repeated identical requests.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Optional

import requests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:1900/v1/chat/completions")
    ap.add_argument("--model", default="/workspace/ckpt/nvidia/Nemotron-Orchestrator-8B")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--prompt-words", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    words = ["ping"] * args.prompt_words
    prompt = " ".join(words)
    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": prompt},
    ]

    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    with open(args.out, "a", encoding="utf-8") as out_f:
        for i in range(args.reps):
            out_f.write(f"=== Request {i+1} ===\n")
            out_f.flush()
            try:
                with requests.post(args.url, json=payload, stream=True, timeout=(10, 3600)) as resp:
                    out_f.write(f"status {resp.status_code}\n")
                    usage: Optional[dict] = None
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data = line[len("data: "):].strip()
                            if data == "[DONE]":
                                break
                            try:
                                obj = json.loads(data)
                            except Exception:
                                continue
                            if isinstance(obj, dict) and obj.get("usage") is not None:
                                usage = obj["usage"]
                    out_f.write("usage: " + json.dumps(usage, ensure_ascii=False) + "\n")
            except Exception as exc:
                out_f.write(f"error: {exc}\n")
            out_f.flush()
            time.sleep(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
