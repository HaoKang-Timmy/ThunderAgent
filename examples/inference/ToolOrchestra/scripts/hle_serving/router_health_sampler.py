#!/usr/bin/env python3
"""
Sample ThunderReact /health and /metrics endpoints on a fixed interval.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from typing import Optional, Tuple
from urllib import error, request


def _now() -> Tuple[float, str]:
    ts = time.time()
    iso = dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")
    return ts, iso


def _fetch(url: str, timeout: float) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    try:
        with request.urlopen(url, timeout=timeout) as resp:
            data = resp.read().decode("utf-8", errors="replace")
            return resp.status, data, None
    except Exception as exc:  # pragma: no cover - best effort sampler
        return None, None, str(exc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--health-url", required=True)
    ap.add_argument("--metrics-url")
    ap.add_argument("--health-out", required=True)
    ap.add_argument("--metrics-out")
    ap.add_argument("--interval-sec", type=float, default=2.0)
    ap.add_argument("--timeout-sec", type=float, default=2.0)
    args = ap.parse_args()

    health_path = args.health_out
    metrics_path = args.metrics_out

    with open(health_path, "a", encoding="utf-8") as health_f, (
        open(metrics_path, "a", encoding="utf-8") if metrics_path else None
    ) as metrics_f:
        while True:
            ts, iso = _now()

            status, text, err = _fetch(args.health_url, args.timeout_sec)
            entry = {
                "ts_unix": ts,
                "ts_iso": iso,
                "status_code": status,
                "ok": err is None and status == 200,
            }
            if err is None and text:
                try:
                    entry["health"] = json.loads(text)
                except Exception:
                    entry["health_raw"] = text
            else:
                entry["error"] = err
            health_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            health_f.flush()

            if metrics_f and args.metrics_url:
                m_status, m_text, m_err = _fetch(args.metrics_url, args.timeout_sec)
                metrics_f.write(
                    f"# ts_unix={ts} ts_iso={iso} status_code={m_status} ok={m_err is None and m_status == 200}\n"
                )
                if m_err is None and m_text:
                    metrics_f.write(m_text.rstrip() + "\n")
                else:
                    metrics_f.write(f"# error={m_err}\n")
                metrics_f.flush()

            time.sleep(args.interval_sec)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
