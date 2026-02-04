#!/bin/bash
set -euo pipefail

ORCH_PORT="${ORCH_PORT:-1900}"
CHECK_EXPERT_HEALTH="${CHECK_EXPERT_HEALTH:-0}"

echo "=== Quick Check $(date '+%Y-%m-%d %H:%M:%S %Z') ==="

echo ""
echo "[vLLM]"
VLLM_PID="$(pgrep -f "vllm serve .*--port ${ORCH_PORT}" | head -1 || true)"
if [[ -n "${VLLM_PID}" ]]; then
  echo "  pid: ${VLLM_PID}"
else
  echo "  pid: (not running)"
fi

if curl -sf --max-time 3 "http://127.0.0.1:${ORCH_PORT}/health" >/dev/null 2>&1; then
  echo "  /health: OK"
else
  echo "  /health: FAIL"
fi

if curl -sf --max-time 5 "http://127.0.0.1:${ORCH_PORT}/v1/models" >/dev/null 2>&1; then
  echo "  /v1/models: OK"
else
  echo "  /v1/models: FAIL"
fi

METRICS="$(curl -sf --max-time 5 "http://127.0.0.1:${ORCH_PORT}/metrics" 2>/dev/null || true)"
MODELS_JSON="$(curl -sf --max-time 5 "http://127.0.0.1:${ORCH_PORT}/v1/models" 2>/dev/null || true)"
if [[ -n "${MODELS_JSON}" ]]; then
  MODEL_ID="$(python3 -c 'import json,sys; data=json.load(sys.stdin); items=(data.get("data") or []); mid=(items[0].get("id") if isinstance(items, list) and items else None); print(mid or "")' <<<"${MODELS_JSON}" 2>/dev/null || true)"
  if [[ -n "${MODEL_ID}" ]]; then
    echo "  model_id: ${MODEL_ID}"
  fi
fi

KV_LINE="$(echo "${METRICS}" | grep -E "^vllm:kv_cache_usage_perc" | head -1 || true)"
GPU_LINE="$(echo "${METRICS}" | grep -E "^vllm:gpu_cache_usage_perc" | head -1 || true)"
KV_FRAC="$(echo "${KV_LINE}" | awk '{print $2}' || true)"
GPU_FRAC="$(echo "${GPU_LINE}" | awk '{print $2}' || true)"
REQ_RUNNING="$(echo "${METRICS}" | awk '/^vllm:num_requests_running/{sum+=$2} END{printf "%.0f", sum+0}' || true)"
REQ_WAITING="$(echo "${METRICS}" | awk '/^vllm:num_requests_waiting/{sum+=$2} END{printf "%.0f", sum+0}' || true)"
if [[ -n "${KV_FRAC}" ]]; then
  KV_PCT="$(echo "${KV_FRAC}" | awk '{printf "%.1f", $1*100.0}')"
  echo "  kv_cache_usage: ${KV_PCT}%"
else
  echo "  kv_cache_usage: (unavailable)"
fi
if [[ -n "${GPU_FRAC}" ]]; then
  GPU_PCT="$(echo "${GPU_FRAC}" | awk '{printf "%.1f", $1*100.0}')"
  echo "  gpu_cache_usage: ${GPU_PCT}%"
fi
if [[ -n "${REQ_RUNNING}" ]]; then
  echo "  requests_running: ${REQ_RUNNING}"
fi
if [[ -n "${REQ_WAITING}" ]]; then
  echo "  requests_waiting: ${REQ_WAITING}"
fi

echo ""
echo "[Tunnels]"
TUNNEL_COUNT="$(ss -lntp 2>/dev/null | egrep ':(1910|1911|1912|1913|1904|1905|1906|1920) ' | wc -l | tr -d ' ' || true)"
echo "  forwarded_ports_listen: ${TUNNEL_COUNT} (expect 8)"

if [[ "${CHECK_EXPERT_HEALTH}" == "1" ]]; then
  for p in 1910 1904 1920; do
    if curl -sf --max-time 5 "http://127.0.0.1:${p}/health" >/dev/null 2>&1; then
      echo "  expert ${p}/health: OK"
    else
      echo "  expert ${p}/health: FAIL"
    fi
  done
fi

echo ""
echo "[GPU]"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader || true
else
  echo "  nvidia-smi: (not found)"
fi

echo ""
echo "[tau2-bench]"
RUN_PID="$(pgrep -f "python .*run_oss.py" | head -1 || true)"
if [[ -z "${RUN_PID}" ]]; then
  echo "  run_oss.py: (not running)"
  exit 0
fi

CMDLINE="$(ps -ww -p "${RUN_PID}" -o args= 2>/dev/null | tr '\n' ' ' || true)"
OUT_DIR="$(echo "${CMDLINE}" | awk '{for (i=1; i<NF; i++) if ($i=="--output-dir") {print $(i+1); exit}}')"
LOG_DIR="$(echo "${CMDLINE}" | awk '{for (i=1; i<NF; i++) if ($i=="--log-dir") {print $(i+1); exit}}')"
if [[ -n "${OUT_DIR}" ]]; then
  RUN_DIR="$(dirname "${OUT_DIR}")"
elif [[ -n "${LOG_DIR}" ]]; then
  RUN_DIR="$(dirname "${LOG_DIR}")"
else
  RUN_DIR=""
fi

echo "  pid: ${RUN_PID}"
echo "  run_dir: ${RUN_DIR}"

DRIVER_LOG="${RUN_DIR}/driver.log"
if [[ ! -f "${DRIVER_LOG}" ]]; then
  echo "  driver.log: (missing)"
  exit 0
fi

DONE="$(grep -c "FINISHED SIMULATION" "${DRIVER_LOG}" 2>/dev/null || true)"
COMPLETE="$(grep -c "Global evaluation complete" "${DRIVER_LOG}" 2>/dev/null || true)"
echo "  tasks_finished: ${DONE}/278"
echo "  global_complete_lines: ${COMPLETE}"

LAST_TASK_LINE="$(grep "FINISHED SIMULATION" "${DRIVER_LOG}" 2>/dev/null | tail -1 || true)"
if [[ -n "${LAST_TASK_LINE}" ]]; then
  LAST_TASK_TS="$(echo "${LAST_TASK_LINE}" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}' | tail -1 || true)"
  if [[ -n "${LAST_TASK_TS}" ]]; then
    LAST_TASK_EPOCH="$(date -d "${LAST_TASK_TS}" +%s 2>/dev/null || true)"
    NOW_EPOCH="$(date +%s)"
    if [[ -n "${LAST_TASK_EPOCH}" ]]; then
      echo "  last_task_ts: ${LAST_TASK_TS} (${NOW_EPOCH}-${LAST_TASK_EPOCH}=$((NOW_EPOCH - LAST_TASK_EPOCH))s ago)"
    fi
  fi
fi

LAST_STEP_LINE="$(grep "type=step_complete" "${DRIVER_LOG}" 2>/dev/null | tail -1 || true)"
if [[ -n "${LAST_STEP_LINE}" ]]; then
  LAST_STEP_TS="$(echo "${LAST_STEP_LINE}" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}' | tail -1 || true)"
  if [[ -n "${LAST_STEP_TS}" ]]; then
    LAST_STEP_EPOCH="$(date -d "${LAST_STEP_TS}" +%s 2>/dev/null || true)"
    NOW_EPOCH="$(date +%s)"
    if [[ -n "${LAST_STEP_EPOCH}" ]]; then
      echo "  last_step_ts: ${LAST_STEP_TS} (${NOW_EPOCH}-${LAST_STEP_EPOCH}=$((NOW_EPOCH - LAST_STEP_EPOCH))s ago)"
    fi
  fi
fi

SUMMARY_LINE="$(grep "Global evaluation complete" "${DRIVER_LOG}" 2>/dev/null | tail -1 || true)"
if [[ -n "${SUMMARY_LINE}" ]]; then
  echo "  summary: ${SUMMARY_LINE}"
fi
