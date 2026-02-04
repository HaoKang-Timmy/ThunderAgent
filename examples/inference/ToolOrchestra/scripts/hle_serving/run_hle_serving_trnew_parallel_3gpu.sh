#!/usr/bin/env bash
set -euo pipefail

# Run TR-new serving sweep with 3 parallel workers (GPUs 0/1/2) and a shared retriever on GPU3.
#
# Usage:
#   export CKPT_DIR=...
#   export INDEX_DIR=...
#   bash scripts/5090/hle_serving/run_hle_serving_trnew_parallel_3gpu.sh
#
# Notes:
# - Experts must already be reachable via localhost tunnels on ports:
#   1840-1843, 1810-1811, 1820-1821
# - Shared retriever runs at http://127.0.0.1:${RETRIEVAL_PORT} (default 1401).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EVAL_DIR="${REPO_DIR}/evaluation"

CKPT_DIR="${CKPT_DIR:?ERROR: CKPT_DIR not set}"
INDEX_DIR="${INDEX_DIR:?ERROR: INDEX_DIR not set (needed by retrieval_hle.py)}"

RETRIEVAL_PORT="${RETRIEVAL_PORT:-1401}"
RET_GPU="${RET_GPU:-3}"
RETRIEVER_CONDA_ENV="${RETRIEVER_CONDA_ENV:-retriever-clean}"
RETRIEVAL_CACHE_DIR="${RETRIEVAL_CACHE_DIR:-/workspace/ToolOrchestra/outputs/hle_serving_5090_retrieval_cache_shared}"

# Worker port bases (override if needed).
# Note: this machine has nginx listening on :8001, so avoid the 8000/8001/8002 range by default.
ORCH_PORT_BASE="${ORCH_PORT_BASE:-1900}"
TR_ROUTER_PORT_BASE="${TR_ROUTER_PORT_BASE:-28000}"
TR_BACKEND_PORT_BASE="${TR_BACKEND_PORT_BASE:-28100}"

START_RETRIEVAL_SHARED="${START_RETRIEVAL_SHARED:-1}"
START_WATCHER="${START_WATCHER:-1}"

# Sweep configuration.
# - Default: full sweep of C_LIST_STR × REP_LIST with a deterministic first wave (24/32/40 rep=1).
# - If CUSTOM_TASKS_FILE is set, tasks are read from that file (format: "C rep" per line) and
#   the first wave is disabled by default (set FORCE_FIRST_WAVE=1 to re-enable).
C_LIST_STR="${C_LIST_STR:-24,32,40,48,56,64,72,96,120,144}"
REP_LIST="${REP_LIST:-1,2,3}"
CUSTOM_TASKS_FILE="${CUSTOM_TASKS_FILE:-}"

FORCE_FIRST_WAVE="${FORCE_FIRST_WAVE:-}"
if [[ -n "${CUSTOM_TASKS_FILE}" && -z "${FORCE_FIRST_WAVE}" ]]; then
  FORCE_FIRST_WAVE=0
else
  FORCE_FIRST_WAVE="${FORCE_FIRST_WAVE:-1}"
fi

IFS=',' read -r -a C_LIST <<< "${C_LIST_STR}"

REPORT_MD="${REPORT_MD:-/workspace/hle-serving-5090/hle_serving_trnew_10_130_clean.md}"

TS="$(date +%Y%m%d_%H%M%S)"
SHARED_OUT_DIR="${REPO_DIR}/outputs/hle_serving_5090_trnew_parallel_${TS}"
mkdir -p "${SHARED_OUT_DIR}/logs"

RETRIEVAL_LOG="${SHARED_OUT_DIR}/logs/retrieval_shared.log"
WATCHER_LOG="${SHARED_OUT_DIR}/logs/watcher_trnew.log"

RET_PID=""
WATCH_PID=""

cleanup() {
  set +e
  if [[ -n "${WATCH_PID}" ]] && kill -0 "${WATCH_PID}" >/dev/null 2>&1; then
    kill "${WATCH_PID}" >/dev/null 2>&1 || true
  fi
  if [[ "${START_RETRIEVAL_SHARED}" == "1" ]] && [[ -n "${RET_PID}" ]] && kill -0 "${RET_PID}" >/dev/null 2>&1; then
    kill "${RET_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "[preflight] checking expert tunnels..."
for p in 1840 1841 1842 1843 1810 1811 1820 1821; do
  if ! curl -sf --max-time 2 "http://127.0.0.1:${p}/health" >/dev/null 2>&1; then
    echo "ERROR: expert tunnel not ready on 127.0.0.1:${p} (/health failed)" >&2
    exit 1
  fi
done
echo "[preflight] expert tunnels OK."

if [[ "${START_RETRIEVAL_SHARED}" == "1" ]]; then
  echo "[retrieval] starting shared retriever on GPU ${RET_GPU} port ${RETRIEVAL_PORT} (env=${RETRIEVER_CONDA_ENV})..."
  source ~/miniconda3/etc/profile.d/conda.sh
  ( \
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate "${RETRIEVER_CONDA_ENV}" && \
    export CUDA_VISIBLE_DEVICES="${RET_GPU}" && \
    export INDEX_DIR="${INDEX_DIR}" && \
	  python "${EVAL_DIR}/retrieval_hle.py" \
	      --port "${RETRIEVAL_PORT}" \
	      --new_cache_dir "${RETRIEVAL_CACHE_DIR}" \
	      --example_id_file "${EVAL_DIR}/examples.json" \
	      --tavily_key "${TAVILY_KEY:-}" \
	      > "${RETRIEVAL_LOG}" 2>&1 \
  ) &
  RET_PID=$!

  echo "[retrieval] waiting /openapi.json..."
  for _ in $(seq 1 600); do
    if ! kill -0 "${RET_PID}" >/dev/null 2>&1; then
      echo "ERROR: shared retrieval exited early. See ${RETRIEVAL_LOG}" >&2
      exit 1
    fi
    if curl -sf --max-time 2 "http://127.0.0.1:${RETRIEVAL_PORT}/openapi.json" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  echo "[retrieval] shared retriever ready."
else
  echo "[retrieval] using existing server on port ${RETRIEVAL_PORT} (START_RETRIEVAL_SHARED=0)"
  if ! curl -sf --max-time 2 "http://127.0.0.1:${RETRIEVAL_PORT}/openapi.json" >/dev/null 2>&1; then
    echo "ERROR: expected existing retrieval server on port ${RETRIEVAL_PORT} but /openapi.json failed" >&2
    exit 1
  fi
fi

SINCE_UNIX="${SINCE_UNIX:-$(date +%s)}"

if [[ "${START_WATCHER}" == "1" ]]; then
  echo "[watcher] starting serving metrics watcher for trnew..."
  python "${REPO_DIR}/scripts/5090/hle_serving/watch_hle_serving_metrics.py" \
    --scheduler trnew \
    --report-md "${REPORT_MD}" \
    --interval-sec 60 \
    --since-unix "${SINCE_UNIX}" \
    > "${WATCHER_LOG}" 2>&1 &
  WATCH_PID=$!
else
  echo "[watcher] disabled (START_WATCHER=0)"
fi

IFS=',' read -r -a REPS <<< "${REP_LIST}"

run_worker() {
  local worker_id="${1:?worker_id}"
  local orch_gpu="${2:?orch_gpu}"
  local orch_port="${3:?orch_port}"
  local tr_router_port="${4:?tr_router_port}"
  local tr_backend_port="${5:?tr_backend_port}"
  local tasks_file="${6:?tasks_file}"
  local lock_file="${7:?lock_file}"
  shift 7
  local first_task_c="${1:-}"
  local first_task_rep="${2:-}"

  echo "[worker${worker_id}] GPU=${orch_gpu} ports orch=${orch_port} router=${tr_router_port} backend=${tr_backend_port}"

  run_one() {
    local c="${1:?c}"
    local rep="${2:?rep}"
    echo "[worker${worker_id}] starting C=${c} rep=${rep}"
    START_RETRIEVAL=0 \
      RETRIEVAL_PORT="${RETRIEVAL_PORT}" \
      ORCH_GPU="${orch_gpu}" \
      ORCH_PORT="${orch_port}" \
      TR_ROUTER_PORT="${tr_router_port}" \
      TR_BACKEND_PORT="${tr_backend_port}" \
      bash "${SCRIPT_DIR}/run_hle_serving_one_w130.sh" trnew "${c}" "${rep}"
    echo "[worker${worker_id}] done C=${c} rep=${rep}"
  }

  # Optional deterministic first task (to match expected initial wave).
  if [[ -n "${first_task_c}" && -n "${first_task_rep}" ]]; then
    run_one "${first_task_c}" "${first_task_rep}"
  fi

  # Pull remaining tasks from FIFO (shared queue).
  while true; do
    line="$(
      (
        flock -x 200
        if [[ ! -s "${tasks_file}" ]]; then
          exit 1
        fi
        IFS= read -r first_line < "${tasks_file}"
        tail -n +2 "${tasks_file}" > "${tasks_file}.tmp" && mv "${tasks_file}.tmp" "${tasks_file}"
        printf '%s\n' "${first_line}"
      ) 200>"${lock_file}"
    )" || break

    c=""
    rep=""
    read -r c rep _rest <<<"${line}"
    [[ -z "${c}" || -z "${rep}" ]] && continue
    run_one "${c}" "${rep}"
  done
}

TASKS_FILE="${SHARED_OUT_DIR}/task_queue.txt"
LOCK_FILE="${SHARED_OUT_DIR}/task_queue.lock"
rm -f "${TASKS_FILE}" "${LOCK_FILE}" "${TASKS_FILE}.tmp"

# Skip settings already appended to the report (in case we restart the sweep).
declare -A DONE=()
STATE_JSON="${REPORT_MD}.state.json"
if [[ -f "${STATE_JSON}" ]]; then
  DONE_PAIRS_TXT="${SHARED_OUT_DIR}/done_pairs.txt"
  STATE_JSON="${STATE_JSON}" python - <<'PY' > "${DONE_PAIRS_TXT}"
import json, os, re
from pathlib import Path

p = Path(os.environ["STATE_JSON"])
obj = json.loads(p.read_text(encoding="utf-8"))
for out_dir in obj.get("processed_out_dirs", []):
    m = re.search(r"_c(?P<c>\d+)_rep(?P<rep>\d+)_", str(out_dir))
    if not m:
        continue
    print(m.group("c"), m.group("rep"))
PY
  while read -r c rep; do
    [[ -z "${c}" || -z "${rep}" ]] && continue
    DONE["${c}:${rep}"]=1
  done < "${DONE_PAIRS_TXT}"
fi

# Start workers; enforce the first wave to be C=24/32/40 rep=1 on GPU0/1/2.
FIRST_C0=""; FIRST_R0=""
FIRST_C1=""; FIRST_R1=""
FIRST_C2=""; FIRST_R2=""
if [[ "${FORCE_FIRST_WAVE}" == "1" ]]; then
  FIRST_C0=24; FIRST_R0=1
  FIRST_C1=32; FIRST_R1=1
  FIRST_C2=40; FIRST_R2=1
fi
if [[ -n "${FIRST_C0}" && -n "${FIRST_R0}" ]] && [[ -n "${DONE["${FIRST_C0}:${FIRST_R0}"]+x}" ]]; then FIRST_C0=""; FIRST_R0=""; fi
if [[ -n "${FIRST_C1}" && -n "${FIRST_R1}" ]] && [[ -n "${DONE["${FIRST_C1}:${FIRST_R1}"]+x}" ]]; then FIRST_C1=""; FIRST_R1=""; fi
if [[ -n "${FIRST_C2}" && -n "${FIRST_R2}" ]] && [[ -n "${DONE["${FIRST_C2}:${FIRST_R2}"]+x}" ]]; then FIRST_C2=""; FIRST_R2=""; fi

run_worker 0 0 "$((ORCH_PORT_BASE + 0))" "$((TR_ROUTER_PORT_BASE + 0))" "$((TR_BACKEND_PORT_BASE + 0))" "${TASKS_FILE}" "${LOCK_FILE}" "${FIRST_C0}" "${FIRST_R0}" &
P0=$!
run_worker 1 1 "$((ORCH_PORT_BASE + 1))" "$((TR_ROUTER_PORT_BASE + 1))" "$((TR_BACKEND_PORT_BASE + 1))" "${TASKS_FILE}" "${LOCK_FILE}" "${FIRST_C1}" "${FIRST_R1}" &
P1=$!
run_worker 2 2 "$((ORCH_PORT_BASE + 2))" "$((TR_ROUTER_PORT_BASE + 2))" "$((TR_BACKEND_PORT_BASE + 2))" "${TASKS_FILE}" "${LOCK_FILE}" "${FIRST_C2}" "${FIRST_R2}" &
P2=$!

if [[ -n "${CUSTOM_TASKS_FILE}" ]]; then
  echo "[queue] using CUSTOM_TASKS_FILE=${CUSTOM_TASKS_FILE}"
  while read -r c rep _rest; do
    [[ -z "${c}" || -z "${rep}" ]] && continue
    # Skip tasks already completed (present in report state).
    if [[ -n "${DONE["${c}:${rep}"]+x}" ]]; then
      continue
    fi
    # Skip tasks already assigned as the deterministic first wave (if enabled).
    if [[ -n "${FIRST_C0}" && "${c}" == "${FIRST_C0}" && "${rep}" == "${FIRST_R0}" ]]; then continue; fi
    if [[ -n "${FIRST_C1}" && "${c}" == "${FIRST_C1}" && "${rep}" == "${FIRST_R1}" ]]; then continue; fi
    if [[ -n "${FIRST_C2}" && "${c}" == "${FIRST_C2}" && "${rep}" == "${FIRST_R2}" ]]; then continue; fi
    echo "${c} ${rep}" >> "${TASKS_FILE}"
  done < "${CUSTOM_TASKS_FILE}"
else
  # Feed remaining tasks: prioritize rep=1 coverage, then rep=2, then rep=3.
  for rep in "${REPS[@]}"; do
    for c in "${C_LIST[@]}"; do
      # Skip tasks already assigned as the deterministic first wave.
      if [[ "${FORCE_FIRST_WAVE}" == "1" && "${rep}" == "1" && ( "${c}" == "24" || "${c}" == "32" || "${c}" == "40" ) ]]; then
        continue
      fi
      # Skip tasks already completed (present in report state).
      if [[ -n "${DONE["${c}:${rep}"]+x}" ]]; then
        continue
      fi
      echo "${c} ${rep}" >> "${TASKS_FILE}"
    done
  done
fi

wait "${P0}" "${P1}" "${P2}"
echo "[done] all workers finished. outputs_root=${REPO_DIR}/outputs"
