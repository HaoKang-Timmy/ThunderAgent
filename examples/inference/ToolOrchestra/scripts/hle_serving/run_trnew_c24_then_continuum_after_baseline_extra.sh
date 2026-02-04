#!/usr/bin/env bash
set -euo pipefail

# Wait for baseline extra sweep (C=32,40) to finish, then run TR-new C=24,
# followed by a continuum sweep (rep=1).

export NEBIUS_API_KEY="${NEBIUS_API_KEY:-}"
source /workspace/oss-ToolOrchestra/setup_envs.sh

echo "[post-baseline] waiting for baseline extra (C=32,40) to finish..."
while pgrep -f "run_baseline_extra_after_sweep.sh" >/dev/null; do
  sleep 30
done

echo "[post-baseline] baseline extra finished, running TR-new C=24 rep=1"
bash /workspace/oss-ToolOrchestra/scripts/5090/hle_serving/run_hle_serving_one_w130.sh trnew 24 1

echo "[post-baseline] TR-new C=24 done, starting continuum sweep (rep=1)"
bash /workspace/oss-ToolOrchestra/scripts/5090/hle_serving/run_hle_serving_sweep.sh continuum 1

echo "[post-baseline] done"
