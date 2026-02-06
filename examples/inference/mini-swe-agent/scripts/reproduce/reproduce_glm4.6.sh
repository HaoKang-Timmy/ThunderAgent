#!/usr/bin/env bash
set -euo pipefail

# Reproduce GLM-4.6-FP8 + vLLM + ThunderAgent + mini-swe-agent SWE-bench run.
# Run this script from repository root.

# =========================
# User-facing configuration
# =========================
# Required: fill this path before running.
HF_HOME=""

# Model
MODEL_REPO="zai-org/GLM-4.6-FP8"
MODEL_DIR="${HF_HOME}/models/glm-4.6-fp8"

# Runtime services
VLLM_PORT="8100"
TA_PORT="8000"
VLLM_TP_SIZE="8"
HEALTH_TIMEOUT_S="1800"

# SWE-bench run
SWEBENCH_SUBSET="lite"
SWEBENCH_SPLIT="test"
SWEBENCH_WORKERS="128"
SWEBENCH_OUTPUT="./swebench_output"

# All logs are stored under repo-root ./logs.
LOG_DIR="./logs"
ROUTER_DIR="${LOG_DIR}/router_metrics"

# =========================
# Internal wiring
# =========================
VLLM_LOG="${LOG_DIR}/vllm_glm46_${VLLM_PORT}.log"
TA_LOG="${LOG_DIR}/thunderagent_${TA_PORT}.log"

VLLM_HEALTH_URL="http://localhost:${VLLM_PORT}/health"
TA_HEALTH_URL="http://localhost:${TA_PORT}/health"

VLLM_PID=""
TA_PID=""

log_info() {
  echo "[INFO] $*"
}

log_error() {
  echo "[ERROR] $*" >&2
}

die() {
  log_error "$*"
  exit 1
}

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    die "Missing required command: ${cmd}"
  fi
}

wait_for_health() {
  local url="$1"
  local name="$2"
  local timeout_s="${3:-600}"

  local start now out
  start="$(date +%s)"

  while true; do
    out=""
    if out="$(curl -sS -m 2 "${url}" 2>&1)"; then
      return 0
    fi

    now="$(date +%s)"
    if (( now - start >= timeout_s )); then
      log_error "${name} health check failed: timeout after ${timeout_s}s"
      log_error "last curl output: ${out}"
      return 1
    fi

    sleep 2
  done
}

cleanup() {
  set +e
  if [[ -n "${TA_PID}" ]] && kill -0 "${TA_PID}" 2>/dev/null; then
    kill "${TA_PID}" 2>/dev/null
  fi
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null
  fi
}
trap cleanup EXIT

prepare_model_dir() {
  log_info "Model repo: ${MODEL_REPO}"
  log_info "Model cache directory: ${MODEL_DIR}"
  mkdir -p "${MODEL_DIR}"
}

download_model() {
  log_info "Downloading model snapshot to: ${MODEL_DIR}"
  HF_HOME="${HF_HOME}" MODEL_REPO="${MODEL_REPO}" MODEL_DIR="${MODEL_DIR}" python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["MODEL_REPO"]
local_dir = os.environ["MODEL_DIR"]

snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)
PY
  log_info "Model download completed."
}

start_vllm() {
  log_info "Starting vLLM on port ${VLLM_PORT} (log: ${VLLM_LOG})"
  nohup vllm serve "${MODEL_REPO}" \
    --download-dir "${MODEL_DIR}" \
    --tensor-parallel-size "${VLLM_TP_SIZE}" \
    --kv_cache_dtype fp8 \
    --port "${VLLM_PORT}" \
    >"${VLLM_LOG}" 2>&1 &
  VLLM_PID="$!"

  log_info "Waiting for vLLM health: ${VLLM_HEALTH_URL}"
  wait_for_health "${VLLM_HEALTH_URL}" "vLLM" "${HEALTH_TIMEOUT_S}"
}

start_thunderagent() {
  log_info "Starting ThunderAgent on port ${TA_PORT} (log: ${TA_LOG})"
  log_info "ThunderAgent profile directory: ${ROUTER_DIR}"
  nohup python -m ThunderAgent \
    --backends "http://localhost:${VLLM_PORT}" \
    --port "${TA_PORT}" \
    --metrics \
    --profile \
    --profile-dir "${ROUTER_DIR}" \
    >"${TA_LOG}" 2>&1 &
  TA_PID="$!"

  log_info "Waiting for ThunderAgent health: ${TA_HEALTH_URL}"
  wait_for_health "${TA_HEALTH_URL}" "ThunderAgent" "${HEALTH_TIMEOUT_S}"
}

run_swebench() {
  log_info "Running mini-extra swebench"
  mini-extra swebench \
    --subset "${SWEBENCH_SUBSET}" \
    --split "${SWEBENCH_SPLIT}" \
    --workers "${SWEBENCH_WORKERS}" \
    --output "${SWEBENCH_OUTPUT}"
}

main() {
  [[ -n "${HF_HOME}" ]] || die "HF_HOME is required. Edit HF_HOME at the top of this script."
  [[ -d "./examples/inference/mini-swe-agent" ]] || die "Please run from repository root."

  require_cmd vllm
  require_cmd curl
  require_cmd mini-extra
  require_cmd python

  mkdir -p "${LOG_DIR}"
  mkdir -p "${ROUTER_DIR}"

  prepare_model_dir
  download_model
  start_vllm
  start_thunderagent
  run_swebench

  log_info "Done. Logs: ${VLLM_LOG}, ${TA_LOG}"
}

main "$@"
