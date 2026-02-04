#!/usr/bin/env bash
set -euo pipefail

# Wait for the current custom queue to finish (ends with TR-new C=24),
# then run extra continuum settings: C=48 (repeat) and C=64.

export NEBIUS_API_KEY="${NEBIUS_API_KEY:-}"
source /workspace/oss-ToolOrchestra/setup_envs.sh

echo "[post-trnew] waiting for custom queue to finish..."
while pgrep -f "run_custom_queue_20260115.sh" >/dev/null; do
  sleep 30
done

echo "[post-trnew] custom queue done, running continuum C=48 rep=1 (repeat)"
bash /workspace/oss-ToolOrchestra/scripts/5090/hle_serving/run_hle_serving_one_w130.sh continuum 48 1

echo "[post-trnew] running continuum C=64 rep=1"
bash /workspace/oss-ToolOrchestra/scripts/5090/hle_serving/run_hle_serving_one_w130.sh continuum 64 1

echo "[post-trnew] done"
