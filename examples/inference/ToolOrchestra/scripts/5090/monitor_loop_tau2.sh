#!/bin/bash
set -euo pipefail

INTERVAL_SEC="${INTERVAL_SEC:-300}"

while true; do
  bash "$(dirname "${BASH_SOURCE[0]}")/quick_check_tau2.sh" || true
  sleep "${INTERVAL_SEC}"
done

