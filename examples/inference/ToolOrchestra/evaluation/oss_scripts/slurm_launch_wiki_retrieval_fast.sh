#!/usr/bin/env bash
# Fast Wikipedia retrieval service (FRAMES) on SLURM.
#
# - Enables corpus preloading (RAM cache) + optional request batching
# - Uses CPU FAISS by default (safer); set WIKI_FAISS_GPU=1 to shard index across visible GPUs
#
# Usage:
#   sbatch /home/junxiong/haokang/ToolOrchestra/evaluation/oss_scripts/slurm_launch_wiki_retrieval_fast.sh
#
# Then find node:
#   squeue -j <JOBID> -h -o %N
#
# Health:
#   curl -s http://<NODE>:8766/health

#SBATCH --partition=batch
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --exclusive
#SBATCH --job-name=frames_wiki_fast
#SBATCH --output=/home/junxiong/haokang/ToolOrchestra/evaluation/oss_scripts/logs/slurm_frames_wiki_fast_%j.out
#SBATCH --error=/home/junxiong/haokang/ToolOrchestra/evaluation/oss_scripts/logs/slurm_frames_wiki_fast_%j.err
#SBATCH --exclude=research-secure-02,research-secure-03,research-secure-07,research-secure-09,research-secure-18,research-secure-20

set -euo pipefail

echo "[$(date)] job_id=${SLURM_JOB_ID} host=$(hostname) ip=$(hostname -i)" | tee -a /home/junxiong/haokang/ToolOrchestra/evaluation/oss_scripts/logs/slurm_frames_wiki_fast_${SLURM_JOB_ID}.log

# We'll pick a usable GPU below based on current GPU memory usage.

export INDEX_DIR="${INDEX_DIR:-/data/haokang/dataset/multi-train/index/}"
export HF_HOME="${HF_HOME:-/home/${USER}/haokang/huggingface}"
export WIKI_RETRIEVAL_PORT="${WIKI_RETRIEVAL_PORT:-8766}"

# --- Performance knobs ---
# Phase 1: corpus preloading (huge win; default in code is ON, keep explicit here)
export WIKI_PRELOAD_CORPUS="${WIKI_PRELOAD_CORPUS:-1}"

# Phase 3: request batching (helps when FRAMES concurrency is high)
export WIKI_REQUEST_BATCHING="${WIKI_REQUEST_BATCHING:-1}"
export WIKI_BATCH_TIMEOUT_S="${WIKI_BATCH_TIMEOUT_S:-0.05}"
export WIKI_MAX_BATCH_REQUESTS="${WIKI_MAX_BATCH_REQUESTS:-16}"
export WIKI_MAX_BATCH_QUERIES="${WIKI_MAX_BATCH_QUERIES:-16}"

# Internal retrieval depth; FRAMES filters down to request.topk
export WIKI_RETRIEVAL_INTERNAL_TOPK="${WIKI_RETRIEVAL_INTERNAL_TOPK:-1000}"

# Important: `torch.cuda.empty_cache()` adds significant per-request latency; keep it OFF.
export WIKI_TORCH_EMPTY_CACHE="${WIKI_TORCH_EMPTY_CACHE:-0}"

# Phase 2 (optional): GPU FAISS index sharding
export WIKI_FAISS_GPU="${WIKI_FAISS_GPU:-1}"
export WIKI_FAISS_GPU_MIN_GPUS="${WIKI_FAISS_GPU_MIN_GPUS:-3}"

# Avoid noisy tqdm in server logs unless debugging
export WIKI_RETRIEVAL_TQDM="${WIKI_RETRIEVAL_TQDM:-0}"

echo "[$(date)] CUDA_VISIBLE_DEVICES(before_pick)=${CUDA_VISIBLE_DEVICES:-<unset>}" | tee -a /home/junxiong/haokang/ToolOrchestra/evaluation/oss_scripts/logs/slurm_frames_wiki_fast_${SLURM_JOB_ID}.log
echo "[$(date)] INDEX_DIR=${INDEX_DIR}" | tee -a /home/junxiong/haokang/ToolOrchestra/evaluation/oss_scripts/logs/slurm_frames_wiki_fast_${SLURM_JOB_ID}.log
echo "[$(date)] WIKI_RETRIEVAL_PORT=${WIKI_RETRIEVAL_PORT}" | tee -a /home/junxiong/haokang/ToolOrchestra/evaluation/oss_scripts/logs/slurm_frames_wiki_fast_${SLURM_JOB_ID}.log
echo "[$(date)] WIKI_PRELOAD_CORPUS=${WIKI_PRELOAD_CORPUS} WIKI_REQUEST_BATCHING=${WIKI_REQUEST_BATCHING} WIKI_FAISS_GPU=${WIKI_FAISS_GPU}" | tee -a /home/junxiong/haokang/ToolOrchestra/evaluation/oss_scripts/logs/slurm_frames_wiki_fast_${SLURM_JOB_ID}.log

# Pick a usable GPU (some nodes are "dirty" with leftover processes on GPU0).
echo "[$(date)] GPU snapshot (before launch):" | tee -a /home/junxiong/haokang/ToolOrchestra/evaluation/oss_scripts/logs/slurm_frames_wiki_fast_${SLURM_JOB_ID}.log
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | tee -a /home/junxiong/haokang/ToolOrchestra/evaluation/oss_scripts/logs/slurm_frames_wiki_fast_${SLURM_JOB_ID}.log || true

# Choose free GPU(s) by lowest memory.used (< 1024 MiB threshold).
NEED_GPUS=1
if [[ "${WIKI_FAISS_GPU}" == "1" || "${WIKI_FAISS_GPU}" == "true" ]]; then
  NEED_GPUS=3
fi

FREE_GPU_LIST="$(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1" "$2}' \
    | sort -k2n \
    | awk '$2 < 1024 {print $1}'
)"
FREE_GPUS="$(echo "${FREE_GPU_LIST}" | head -n "${NEED_GPUS}" | paste -sd, -)"
FREE_GPU_COUNT="$(echo "${FREE_GPUS}" | awk -F',' '{print NF}')"
if [[ -z "${FREE_GPUS}" || "${FREE_GPU_COUNT}" -lt "${NEED_GPUS}" ]]; then
  echo "[$(date)] ERROR: Not enough free GPUs found (<1024MiB). need=${NEED_GPUS} got=${FREE_GPUS:-<none>} Aborting." | tee -a /home/junxiong/haokang/ToolOrchestra/evaluation/oss_scripts/logs/slurm_frames_wiki_fast_${SLURM_JOB_ID}.log
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tee -a /home/junxiong/haokang/ToolOrchestra/evaluation/oss_scripts/logs/slurm_frames_wiki_fast_${SLURM_JOB_ID}.log || true
  exit 3
fi

export CUDA_VISIBLE_DEVICES="${FREE_GPUS}"
echo "[$(date)] Picked CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (need_gpus=${NEED_GPUS} for WIKI_FAISS_GPU=${WIKI_FAISS_GPU})" | tee -a /home/junxiong/haokang/ToolOrchestra/evaluation/oss_scripts/logs/slurm_frames_wiki_fast_${SLURM_JOB_ID}.log

bash /home/junxiong/haokang/ToolOrchestra/evaluation/oss_scripts/launch_wiki_retrieval.sh


