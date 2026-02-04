#!/usr/bin/env bash
set -euo pipefail

# Wait for the baseline sweep to finish (through C=24), then run extra C values.

# setup_envs.sh prints key prefixes; avoid unbound var errors under set -u.
# export NEBIUS_API_KEY="${NEBIUS_API_KEY:-}"
source /workspace/oss-ToolOrchestra/setup_envs.sh

echo "[baseline-extra] waiting for baseline sweep to finish..."
while pgrep -f "run_hle_serving_sweep.sh baseline" >/dev/null; do
  sleep 30
done

echo "[baseline-extra] sweep done, running extra C=32,40 (rep=1)"
for c in 32 40; do
  echo "[baseline-extra] starting baseline C=${c} rep=1"
  bash /workspace/oss-ToolOrchestra/scripts/5090/hle_serving/run_hle_serving_one_w130.sh baseline "${c}" 1
done

echo "[baseline-extra] done"
