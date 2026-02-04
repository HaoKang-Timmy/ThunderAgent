#!/usr/bin/env bash
set -euo pipefail

# Custom eval queue (requested order):
# 1) continuum: C=24,32,40,48,56,72
# 2) baseline:  C=24,32,40,56
# 3) trnew:     C=24

export NEBIUS_API_KEY="${NEBIUS_API_KEY:-}"
source /workspace/oss-ToolOrchestra/setup_envs.sh

CONTINUUM_C_LIST=(24 32 40 48 56 72)
BASELINE_C_LIST=(24 32 40 56)

echo "[queue] continuum C_LIST=${CONTINUUM_C_LIST[*]}"
for c in "${CONTINUUM_C_LIST[@]}"; do
  echo "[queue] continuum C=${c} rep=1"
  bash /workspace/oss-ToolOrchestra/scripts/5090/hle_serving/run_hle_serving_one_w130.sh continuum "${c}" 1
done

echo "[queue] baseline C_LIST=${BASELINE_C_LIST[*]}"
for c in "${BASELINE_C_LIST[@]}"; do
  echo "[queue] baseline C=${c} rep=1"
  bash /workspace/oss-ToolOrchestra/scripts/5090/hle_serving/run_hle_serving_one_w130.sh baseline "${c}" 1
done

echo "[queue] trnew C=24 rep=1"
bash /workspace/oss-ToolOrchestra/scripts/5090/hle_serving/run_hle_serving_one_w130.sh trnew 24 1

echo "[queue] done"
