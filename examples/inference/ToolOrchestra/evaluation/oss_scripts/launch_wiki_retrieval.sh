#!/usr/bin/env bash
# Launch Wikipedia retrieval service (FRAMES) on a GPU node.
#
# Requires:
# - conda env: retriever
# - GPU node
# - INDEX_DIR pointing to multi-train index dir (wiki.index/wiki.jsonl)
#
# Optional env vars:
# - INDEX_DIR (default: /data/haokang/dataset/multi-train/index/)
# - HF_HOME  (default: /home/${USER}/haokang/huggingface)
# - WIKI_RETRIEVAL_PORT (default: 8766)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "[$(date)] Starting wiki retrieval service on host=$(hostname) user=${USER}" | tee -a "${LOG_DIR}/retrieval_wiki_launcher_$(date +%Y%m%d).log"

source "/home/${USER}/miniconda3/etc/profile.d/conda.sh"
conda activate retriever

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

export INDEX_DIR="${INDEX_DIR:-/data/haokang/dataset/multi-train/index/}"
export HF_HOME="${HF_HOME:-/home/${USER}/haokang/huggingface}"

echo "[$(date)] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG_DIR}/retrieval_wiki_launcher_$(date +%Y%m%d).log"
echo "[$(date)] INDEX_DIR=${INDEX_DIR}" | tee -a "${LOG_DIR}/retrieval_wiki_launcher_$(date +%Y%m%d).log"
echo "[$(date)] Checking index files..." | tee -a "${LOG_DIR}/retrieval_wiki_launcher_$(date +%Y%m%d).log"
ls -lh "${INDEX_DIR}/wiki.index" "${INDEX_DIR}/wiki.jsonl" | tee -a "${LOG_DIR}/retrieval_wiki_launcher_$(date +%Y%m%d).log"

PORT="${WIKI_RETRIEVAL_PORT:-8766}"
cd "${EVAL_DIR}"

echo "[$(date)] Launching retrieval_wiki.py on port=${PORT} ..." | tee -a "${LOG_DIR}/retrieval_wiki_launcher_$(date +%Y%m%d).log"
python retrieval_wiki.py --port "${PORT}" 2>&1 | tee "${LOG_DIR}/retrieval_wiki_port${PORT}_$(date +%Y%m%d).log"


