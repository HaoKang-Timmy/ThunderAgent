#!/usr/bin/env bash
# Run ONE HLE serving setting on a 2×5090 box (10–130min active window).
#
# Topology:
# - GPU0: Orchestrator-8B vLLM (baseline / continuum / TR-new backend)
# - GPU1: retrieval_hle.py (FAISS on GPU, micro-batching)
# - Experts: H100 cluster reached via SSH tunnel to localhost ports on 5090
#
# Window definition:
#   t0 = first orchestrator response recorded in orchestrator_usage.jsonl
#   window = [t0 + 10min, t0 + 130min]  (600s .. 7800s)
#
# Usage (on 5090):
#   export CKPT_DIR=/path/to/Nemotron-Orchestrator-8B
#   export INDEX_DIR=/path/to/index_dir_with_eval.index_and_eval.jsonl
#   bash scripts/5090/hle_serving/run_hle_serving_one_w130.sh <scheduler> <C> [rep]
#
# scheduler:
#   - baseline     (standard vLLM, no router)
#   - continuum    (vllm-continuum, no router; RUN_OUTPUT_DIR per run)
#   - trnew        (TR-new router -> 1x backend)
#
# Requirements (5090):
# - conda envs:
#   - baseline/trnew: vllm1
#   - continuum: vllm-continuum
#   - retrieval: retriever (override RETRIEVER_CONDA_ENV if needed)
# - SSH tunnels already established so these are reachable on 5090:
#   - http://127.0.0.1:1840-1843/health (search)
#   - http://127.0.0.1:1810-1811/health (reasoner)
#   - http://127.0.0.1:1820-1821/health (answer)

set -euo pipefail

SCHEDULER="${1:?scheduler required (baseline|continuum|trnew)}"
CONCURRENCY="${2:?concurrency required}"
REP_IDX="${3:-1}"

WINDOW_START_SEC="${WINDOW_START_SEC:-600}"
WINDOW_END_SEC="${WINDOW_END_SEC:-7800}"
EVAL_TIMEOUT_MIN="${EVAL_TIMEOUT_MIN:-145}"
SAMPLE_INTERVAL_SEC="${SAMPLE_INTERVAL_SEC:-2}"

ORCH_GPU="${ORCH_GPU:-0}"
RET_GPU="${RET_GPU:-1}"

ORCH_PORT="${ORCH_PORT:-1900}"
TR_ROUTER_PORT="${TR_ROUTER_PORT:-8000}"
TR_BACKEND_PORT="${TR_BACKEND_PORT:-8100}"
RETRIEVAL_PORT="${RETRIEVAL_PORT:-1401}"

VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.95}"

RETRIEVER_CONDA_ENV="${RETRIEVER_CONDA_ENV:-retriever-clean}"
BASELINE_CONDA_ENV="${BASELINE_CONDA_ENV:-vllm1}"
TR_CONDA_ENV="${TR_CONDA_ENV:-vllm1}"
CONTINUUM_CONDA_ENV="${CONTINUUM_CONDA_ENV:-vllm-continuum}"

CKPT_DIR="${CKPT_DIR:?ERROR: CKPT_DIR not set}"
INDEX_DIR="${INDEX_DIR:?ERROR: INDEX_DIR not set (needed by retrieval_hle.py)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EVAL_DIR="${REPO_DIR}/evaluation"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${REPO_DIR}/outputs/hle_serving_5090_${SCHEDULER}_c${CONCURRENCY}_rep${REP_IDX}_${TS}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

USAGE_JSONL="${OUT_DIR}/orchestrator_usage.jsonl"
METRICS_CSV="${OUT_DIR}/prefix_cache_timeseries.csv"
GPU_UTIL_CSV="${OUT_DIR}/gpu_sm_util_timeseries.csv"
MODEL_CONFIG_PATH="${OUT_DIR}/model_config_hle_serving.json"
SUMMARY_JSON="${OUT_DIR}/window_summary.json"
STEPS_JSON="${OUT_DIR}/steps_summary.json"
COMBINED_JSON="${OUT_DIR}/combined_summary.json"

RETRIEVAL_LOG="${LOG_DIR}/retrieval.log"
ORCH_LOG="${LOG_DIR}/orchestrator.log"
TR_ROUTER_LOG="${LOG_DIR}/tr_router.log"
EVAL_LOG="${LOG_DIR}/eval_driver.log"
SAMPLER_LOG="${LOG_DIR}/metrics_sampler.log"
GPU_SAMPLER_LOG="${LOG_DIR}/gpu_sm_sampler.log"

START_RETRIEVAL="${START_RETRIEVAL:-1}"
RETRIEVAL_CACHE_DIR="${RETRIEVAL_CACHE_DIR:-${OUT_DIR}/retrieval_cache}"
KILL_PORTS="${KILL_PORTS:-1}"

RET_PID=""
ORCH_PID=""
TR_PID=""

source ~/miniconda3/etc/profile.d/conda.sh

ulimit -n 1048576 2>/dev/null || ulimit -n 65535 2>/dev/null || true
echo "[ulimit] nofile soft=$(ulimit -Sn) hard=$(ulimit -Hn)"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

port_in_use() {
  local port="${1:?port}"
  lsof -tiTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
}

kill_port_listeners() {
  local port="${1:?port}"
  local pids=""
  pids="$(lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  echo "[cleanup] killing listeners on port ${port}: ${pids}"
  kill ${pids} >/dev/null 2>&1 || true
  sleep 0.2
  kill -9 ${pids} >/dev/null 2>&1 || true
}

wait_port_free() {
  local port="${1:?port}"
  local timeout_sec="${2:-20}"
  local t0
  t0="$(date +%s)"
  while port_in_use "${port}"; do
    if (( $(date +%s) - t0 > timeout_sec )); then
      return 1
    fi
    sleep 0.2
  done
  return 0
}

cleanup() {
  set +e
  echo "[cleanup] stopping background processes..."

  if [[ -n "${TR_PID}" ]] && kill -0 "${TR_PID}" >/dev/null 2>&1; then
    kill "${TR_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${ORCH_PID}" ]] && kill -0 "${ORCH_PID}" >/dev/null 2>&1; then
    kill "${ORCH_PID}" >/dev/null 2>&1 || true
  fi
  if [[ "${START_RETRIEVAL}" == "1" ]] && [[ -n "${RET_PID}" ]] && kill -0 "${RET_PID}" >/dev/null 2>&1; then
    kill "${RET_PID}" >/dev/null 2>&1 || true
  fi

  if [[ "${KILL_PORTS}" == "1" ]]; then
    kill_port_listeners "${ORCH_PORT}"
    kill_port_listeners "${TR_ROUTER_PORT}"
    kill_port_listeners "${TR_BACKEND_PORT}"
    if [[ "${START_RETRIEVAL}" == "1" ]]; then
      kill_port_listeners "${RETRIEVAL_PORT}"
    fi
  fi
}
trap cleanup EXIT

echo "=========================================="
echo "=== HLE Serving (5090) (${SCHEDULER}) ==="
echo "=========================================="
echo "C=${CONCURRENCY} rep=${REP_IDX}"
echo "window offsets (sec): ${WINDOW_START_SEC}..${WINDOW_END_SEC}"
echo "timeout: ${EVAL_TIMEOUT_MIN}m"
echo "out: ${OUT_DIR}"

if [[ "${KILL_PORTS}" == "1" ]]; then
  kill_port_listeners "${ORCH_PORT}"
  kill_port_listeners "${TR_ROUTER_PORT}"
  kill_port_listeners "${TR_BACKEND_PORT}"
  if [[ "${START_RETRIEVAL}" == "1" ]]; then
    kill_port_listeners "${RETRIEVAL_PORT}"
  fi
  wait_port_free "${ORCH_PORT}" || { echo "ERROR: port ${ORCH_PORT} still in use after cleanup" >&2; exit 1; }
  wait_port_free "${TR_ROUTER_PORT}" || { echo "ERROR: port ${TR_ROUTER_PORT} still in use after cleanup" >&2; exit 1; }
  wait_port_free "${TR_BACKEND_PORT}" || { echo "ERROR: port ${TR_BACKEND_PORT} still in use after cleanup" >&2; exit 1; }
  if [[ "${START_RETRIEVAL}" == "1" ]]; then
    wait_port_free "${RETRIEVAL_PORT}" || { echo "ERROR: port ${RETRIEVAL_PORT} still in use after cleanup" >&2; exit 1; }
  fi
fi

log "phase=preflight done"

echo "[preflight] checking expert tunnels..."
for p in 1840 1841 1842 1843 1810 1811 1820 1821; do
  if ! curl -sf --max-time 2 "http://127.0.0.1:${p}/health" >/dev/null 2>&1; then
    echo "ERROR: expert tunnel not ready on 127.0.0.1:${p} (/health failed)" >&2
    exit 1
  fi
done
echo "[preflight] expert tunnels OK."

echo "[model_config] writing ${MODEL_CONFIG_PATH}"
cat > "${MODEL_CONFIG_PATH}" <<EOF
{
  "_comment": "Auto-generated for HLE serving on 5090 (experts via localhost tunnels, retrieval local).",
  "vllm_model_config_path": "${MODEL_CONFIG_PATH}",
  "retrieval": [{"ip_addr": "127.0.0.1", "port": "${RETRIEVAL_PORT}"}],
  "${CKPT_DIR}": [{"ip_addr": "127.0.0.1", "port": "${ORCH_PORT}"}],
  "openai/gpt-oss-20b": [
    {"ip_addr": "127.0.0.1", "port": "1840"},
    {"ip_addr": "127.0.0.1", "port": "1841"},
    {"ip_addr": "127.0.0.1", "port": "1842"},
    {"ip_addr": "127.0.0.1", "port": "1843"}
  ],
  "Qwen/Qwen2.5-Coder-14B-Instruct": [
    {"ip_addr": "127.0.0.1", "port": "1810"},
    {"ip_addr": "127.0.0.1", "port": "1811"}
  ],
  "Qwen/Qwen3-32B-FP8": [
    {"ip_addr": "127.0.0.1", "port": "1820"},
    {"ip_addr": "127.0.0.1", "port": "1821"}
  ]
}
EOF

log "phase=retrieval start"
if [[ "${START_RETRIEVAL}" == "1" ]]; then
  echo "[retrieval] starting on GPU ${RET_GPU} port ${RETRIEVAL_PORT} (env=${RETRIEVER_CONDA_ENV})..."
  (conda activate "${RETRIEVER_CONDA_ENV}" >/dev/null 2>&1 || conda activate "${RETRIEVER_CONDA_ENV}") || true
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
else
  echo "[retrieval] using existing server on port ${RETRIEVAL_PORT} (START_RETRIEVAL=0)"
fi

echo "[retrieval] waiting /openapi.json..."
for _ in $(seq 1 600); do
  if [[ "${START_RETRIEVAL}" == "1" ]]; then
    if ! kill -0 "${RET_PID}" >/dev/null 2>&1; then
      echo "ERROR: retrieval exited early. See ${RETRIEVAL_LOG}" >&2
      exit 1
    fi
  fi
  if curl -sf --max-time 2 "http://127.0.0.1:${RETRIEVAL_PORT}/openapi.json" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
echo "[retrieval] ready."
log "phase=retrieval ready"

echo "[orchestrator] starting (${SCHEDULER}) on GPU ${ORCH_GPU}..."

start_baseline() {
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate "${BASELINE_CONDA_ENV}"
  export CUDA_VISIBLE_DEVICES="${ORCH_GPU}"
  vllm serve "${CKPT_DIR}" \
    --host 0.0.0.0 \
    --port "${ORCH_PORT}" \
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}" \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --enable-force-include-usage \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    > "${ORCH_LOG}" 2>&1
}

start_continuum() {
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONTINUUM_CONDA_ENV}"
  export CUDA_VISIBLE_DEVICES="${ORCH_GPU}"
  export RUN_OUTPUT_DIR="${OUT_DIR}"
  vllm serve "${CKPT_DIR}" \
    --host 0.0.0.0 \
    --port "${ORCH_PORT}" \
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}" \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --enable-force-include-usage \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --scheduling-policy continuum \
    > "${ORCH_LOG}" 2>&1
}

start_trnew() {
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate "${TR_CONDA_ENV}"
  export CUDA_VISIBLE_DEVICES="${ORCH_GPU}"
  vllm serve "${CKPT_DIR}" \
    --host 0.0.0.0 \
    --port "${TR_BACKEND_PORT}" \
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}" \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --enable-force-include-usage \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    > "${ORCH_LOG}" 2>&1
}

case "${SCHEDULER}" in
  baseline)
    start_baseline &
    ORCH_PID=$!
    ;;
  continuum)
    start_continuum &
    ORCH_PID=$!
    ;;
  trnew)
    start_trnew &
    ORCH_PID=$!
    ;;
  *)
    echo "ERROR: unknown scheduler '${SCHEDULER}'" >&2
    exit 2
    ;;
esac

echo "[orchestrator] waiting /health..."
ORCH_HEALTH_PORT="${ORCH_PORT}"
METRICS_PORT="${ORCH_PORT}"
if [[ "${SCHEDULER}" == "trnew" ]]; then
  ORCH_HEALTH_PORT="${TR_BACKEND_PORT}"
  METRICS_PORT="${TR_BACKEND_PORT}"
fi

for _ in $(seq 1 600); do
  if ! kill -0 "${ORCH_PID}" >/dev/null 2>&1; then
    echo "ERROR: orchestrator process exited early. See ${ORCH_LOG}" >&2
    exit 1
  fi
  if curl -sf --max-time 2 "http://127.0.0.1:${ORCH_HEALTH_PORT}/health" >/dev/null 2>&1 && \
     curl -sf --max-time 2 "http://127.0.0.1:${ORCH_HEALTH_PORT}/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
echo "[orchestrator] ready."
log "phase=orchestrator ready"

if [[ "${SCHEDULER}" == "trnew" ]]; then
  log "phase=tr_router start"
  echo "[trnew] starting router on port ${TR_ROUTER_PORT}..."
  export ROUTER_URL="http://127.0.0.1:${TR_ROUTER_PORT}"
  # Must run from repo root so `python -m ThunderReact` resolves the symlink.
  (cd "${REPO_DIR}/.." && \
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate "${TR_CONDA_ENV}" && \
    python -m ThunderReact \
      --host 0.0.0.0 \
      --port "${TR_ROUTER_PORT}" \
      --backends "http://127.0.0.1:${TR_BACKEND_PORT}" \
      --router tr \
      --profile \
      --profile-dir "${OUT_DIR}/tr_profiles" \
      --metrics \
      --metrics-interval 5 \
      > "${TR_ROUTER_LOG}" 2>&1) &
  TR_PID=$!
  echo "[trnew] waiting router /health..."
  for _ in $(seq 1 300); do
    if ! kill -0 "${TR_PID}" >/dev/null 2>&1; then
      echo "ERROR: TR router exited early. See ${TR_ROUTER_LOG}" >&2
      exit 1
    fi
    if curl -sf --max-time 2 "http://127.0.0.1:${TR_ROUTER_PORT}/health" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  echo "[trnew] router ready."
  log "phase=tr_router ready"

  # Update orchestrator endpoint in model_config to point to TR router.
  python - <<PY
import json
from pathlib import Path
p=Path("${MODEL_CONFIG_PATH}")
cfg=json.loads(p.read_text())
cfg["${CKPT_DIR}"]=[{"ip_addr":"127.0.0.1","port":"${TR_ROUTER_PORT}"}]
p.write_text(json.dumps(cfg,indent=2))
print("[model_config] updated orchestrator endpoint -> TR router")
PY
fi

echo "[sampler] starting /metrics sampler (interval=${SAMPLE_INTERVAL_SEC}s) from port ${METRICS_PORT}..."
bash "${REPO_DIR}/scripts/hle_preexp/kv_prefix_cache_hit_sampler.sh" \
  "http://127.0.0.1:${METRICS_PORT}/metrics" \
  "${METRICS_CSV}" \
  "${SAMPLE_INTERVAL_SEC}" \
  > "${SAMPLER_LOG}" 2>&1 &

echo "[sampler] starting GPU SM sampler (interval=${SAMPLE_INTERVAL_SEC}s) gpu_index=${ORCH_GPU}..."
(source ~/miniconda3/etc/profile.d/conda.sh && conda activate "${BASELINE_CONDA_ENV}" && \
  python "${REPO_DIR}/scripts/preexp/gpu_sm_util_sampler.py" \
    --out-csv "${GPU_UTIL_CSV}" \
    --gpu-index "${ORCH_GPU}" \
    --interval-sec "${SAMPLE_INTERVAL_SEC}" \
    > "${GPU_SAMPLER_LOG}" 2>&1) &

echo "[eval] starting eval_hle_local.py (timeout ${EVAL_TIMEOUT_MIN}m) ..."
export REPO_PATH="${REPO_DIR}"
export TOOL_ORCH_USAGE_LOG_PATH="${USAGE_JSONL}"
export HLE_LOG_LEVEL="PROFILE"
export HLE_LOG_STREAM="1"
cd "${EVAL_DIR}"
HLE_EXAMPLE_PATH="${HLE_EXAMPLE_PATH:-${EVAL_DIR}/hle.jsonl}"

set +e
timeout --signal=TERM --kill-after=30s "${EVAL_TIMEOUT_MIN}m" bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate '${BASELINE_CONDA_ENV}' && \
  exec python eval_hle_local.py \
    --model_name '${CKPT_DIR}' \
    --output_dir '${OUT_DIR}/outputs' \
    --model_config '${MODEL_CONFIG_PATH}' \
    --example_path '${HLE_EXAMPLE_PATH}' \
    --concurrency '${CONCURRENCY}' \
    --max_rounds 50 \
    --log_level PROFILE \
    --log_file '${LOG_DIR}/hle_profile.log' \
  " 2>&1 | tee -a "${EVAL_LOG}"
EVAL_RC=${PIPESTATUS[0]}
set -e
echo "[eval] exit_code=${EVAL_RC} (expected 124 if timed out)"

echo "[summarize] computing window stats (${WINDOW_START_SEC}..${WINDOW_END_SEC})..."
python "${REPO_DIR}/scripts/hle_preexp/summarize_hle_prefix_cache_window.py" \
  --metrics-csv "${METRICS_CSV}" \
  --usage-jsonl "${USAGE_JSONL}" \
  --eval-log "${EVAL_LOG}" \
  --start-offset-sec "${WINDOW_START_SEC}" \
  --end-offset-sec "${WINDOW_END_SEC}" \
  --windows "${WINDOW_START_SEC}:${WINDOW_END_SEC}" \
  --out-json "${SUMMARY_JSON}" \
  >/dev/null 2>&1 || true

python "${REPO_DIR}/scripts/preexp/summarize_steps_per_sec_window.py" \
  --usage-jsonl "${USAGE_JSONL}" \
  --eval-log "${EVAL_LOG}" \
  --start-offset-sec "${WINDOW_START_SEC}" \
  --end-offset-sec "${WINDOW_END_SEC}" \
  > "${STEPS_JSON}" 2>/dev/null || true

SUMMARY_JSON="${SUMMARY_JSON}" STEPS_JSON="${STEPS_JSON}" METRICS_CSV="${METRICS_CSV}" USAGE_JSONL="${USAGE_JSONL}" GPU_UTIL_CSV="${GPU_UTIL_CSV}" \
  python - <<'PY' > "${COMBINED_JSON}"
import json, os
from pathlib import Path

summary = {}
steps = {}
sp = Path(os.environ["SUMMARY_JSON"])
tp = Path(os.environ["STEPS_JSON"])
if sp.exists():
    try:
        summary = json.loads(sp.read_text(encoding="utf-8"))
    except Exception:
        summary = {}
if tp.exists():
    try:
        steps = json.loads(tp.read_text(encoding="utf-8"))
    except Exception:
        steps = {}

out = {
    "ok": bool(summary.get("ok")),
    "t0_first_request_unix": summary.get("t0_first_request_unix"),
    "t0_first_request_iso": summary.get("t0_first_request_iso"),
    "windows": summary.get("windows"),
    "steps": steps,
    "paths": {
        "metrics_csv": os.environ.get("METRICS_CSV"),
        "usage_jsonl": os.environ.get("USAGE_JSONL"),
        "gpu_util_csv": os.environ.get("GPU_UTIL_CSV"),
    },
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY

echo "[done] OUT_DIR=${OUT_DIR}"
