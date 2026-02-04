#!/usr/bin/env bash
set -euo pipefail

# HLE profiling launcher:
# - Creates timestamped output/log dirs
# - Launches run_hle_local.py in the background (nohup)
# - Writes periodic progress to monitor.out
# - Runs analyze_hle_timing.py after completion

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${EVAL_DIR}/../setup_envs.sh" >/dev/null 2>&1

TS="$(date +%Y%m%d_%H%M%S)"

# Defaults (override via env vars when calling this script)
AGENT_MODEL="${AGENT_MODEL:-${CKPT_DIR:-}}"
INDEX_DIR_ARG="${INDEX_DIR_ARG:-${INDEX_DIR:-/workspace/dataset/multi-train/index}}"
EXAMPLE_PATH="${EXAMPLE_PATH:-${EVAL_DIR}/hle.jsonl}"
CONCURRENCY="${CONCURRENCY:-512}"
MAX_ROUNDS="${MAX_ROUNDS:-50}"

RUN_NAME="${RUN_NAME:-hle_profile_${TS}}"
LOG_DIR="${LOG_DIR:-${EVAL_DIR}/logs/${RUN_NAME}}"
OUT_DIR="${OUT_DIR:-${EVAL_DIR}/outputs/${RUN_NAME}}"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

if command -v pgrep >/dev/null 2>&1; then
  if pgrep -f "ToolOrchestra/evaluation/run_hle_local.py" >/dev/null 2>&1; then
    echo "ERROR: run_hle_local.py already running (refusing to start a second one)." | tee -a "${LOG_DIR}/runner.out"
    exit 1
  fi
fi

if [[ -z "${AGENT_MODEL}" ]]; then
  echo "ERROR: AGENT_MODEL/CKPT_DIR not set. Export CKPT_DIR or AGENT_MODEL." | tee -a "${LOG_DIR}/runner.out"
  exit 1
fi

echo "[${TS}] Starting HLE profile run" | tee -a "${LOG_DIR}/runner.out"
echo "  AGENT_MODEL=${AGENT_MODEL}" | tee -a "${LOG_DIR}/runner.out"
echo "  INDEX_DIR=${INDEX_DIR_ARG}" | tee -a "${LOG_DIR}/runner.out"
echo "  EXAMPLE_PATH=${EXAMPLE_PATH}" | tee -a "${LOG_DIR}/runner.out"
echo "  CONCURRENCY=${CONCURRENCY}" | tee -a "${LOG_DIR}/runner.out"
echo "  OUT_DIR=${OUT_DIR}" | tee -a "${LOG_DIR}/runner.out"
echo "  LOG_DIR=${LOG_DIR}" | tee -a "${LOG_DIR}/runner.out"

DRIVER_OUT="${LOG_DIR}/driver.out"
MONITOR_OUT="${LOG_DIR}/monitor.out"
ANALYSIS_OUT="${LOG_DIR}/analysis.out"

# Always restart orchestrator (GPU0) + retrieval (GPU1) for each run, and ensure
# their logs land under this run's LOG_DIR.
kill_listen_port() {
  local port="$1"
  local desc="$2"
  if ! command -v lsof >/dev/null 2>&1; then
    echo "[runner] WARNING: lsof not found; cannot kill by port for ${desc} (port=${port})." | tee -a "${LOG_DIR}/runner.out"
    return 0
  fi
  local pids
  pids="$(lsof -t -iTCP:${port} -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  echo "[runner] stopping ${desc} listeners on port=${port}: pids=${pids}" | tee -a "${LOG_DIR}/runner.out"
  kill ${pids} >/dev/null 2>&1 || true
  sleep 3
  pids="$(lsof -t -iTCP:${port} -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    echo "[runner] force-killing ${desc} listeners on port=${port}: pids=${pids}" | tee -a "${LOG_DIR}/runner.out"
    kill -9 ${pids} >/dev/null 2>&1 || true
  fi
}

kill_listen_port 1406 "orchestrator vLLM"
kill_listen_port 1401 "retrieval"

# Launch run_hle_local.py
# Note: We pass --no-reuse-running so both GPU0 vLLM and GPU1 retrieval are restarted each run.
# We also source setup_envs.sh above, so CKPT_DIR/API keys are available.
nohup python "${EVAL_DIR}/run_hle_local.py" \
  --agent-model "${AGENT_MODEL}" \
  --index-dir "${INDEX_DIR_ARG}" \
  --example-path "${EXAMPLE_PATH}" \
  --output-dir "${OUT_DIR}" \
  --log-dir "${LOG_DIR}" \
  --max-rounds "${MAX_ROUNDS}" \
  --concurrency "${CONCURRENCY}" \
  --log-level PROFILE \
  --no-reuse-running \
  > "${DRIVER_OUT}" 2>&1 &

PID=$!
echo "${PID}" > "${LOG_DIR}/driver.pid"
echo "[${TS}] driver.pid=${PID}" | tee -a "${LOG_DIR}/runner.out"

(
  while kill -0 "${PID}" >/dev/null 2>&1; do
    now="$(date +%Y-%m-%d_%H:%M:%S)"
    done_cnt="0"
    if [[ -f "${LOG_DIR}/eval_hle_local.log" ]]; then
      # Fixed-string match (faster + avoids regex escaping pitfalls)
      done_cnt="$(grep -Fc "[HLE_TASK_COMPLETE]" "${LOG_DIR}/eval_hle_local.log" 2>/dev/null || true)"
    fi
    total_cnt="0"
    if [[ -f "${EXAMPLE_PATH}" ]]; then
      total_cnt="$(grep -cve '^[[:space:]]*$' "${EXAMPLE_PATH}" 2>/dev/null || true)"
    fi
    echo "[${now}] done=${done_cnt}/${total_cnt} pid=${PID}" >> "${MONITOR_OUT}"
    sleep 60
  done
) &
MON_PID=$!
echo "${MON_PID}" > "${LOG_DIR}/monitor.pid"

set +e
wait "${PID}"
RET=$?
set -e
echo "[$(date +%Y-%m-%d_%H:%M:%S)] run_hle_local.py exited code=${RET}" | tee -a "${LOG_DIR}/runner.out"

if kill -0 "${MON_PID}" >/dev/null 2>&1; then
  kill "${MON_PID}" >/dev/null 2>&1 || true
fi

# Run analysis inside the vllm1 env
bash -lc "source /root/miniconda3/etc/profile.d/conda.sh && conda activate vllm1 && python \"${EVAL_DIR}/analyze_hle_timing.py\" --log-dir \"${LOG_DIR}\" --log-glob \"hle.log\" --out-prefix \"${RUN_NAME}\" --bins 10" \
  > "${ANALYSIS_OUT}" 2>&1 || true

echo "Done. See:" | tee -a "${LOG_DIR}/runner.out"
echo "  ${DRIVER_OUT}" | tee -a "${LOG_DIR}/runner.out"
echo "  ${LOG_DIR}/hle.log" | tee -a "${LOG_DIR}/runner.out"
echo "  ${ANALYSIS_OUT}" | tee -a "${LOG_DIR}/runner.out"

exit "${RET}"
