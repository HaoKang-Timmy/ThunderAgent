#!/bin/bash
set -euo pipefail

# Run baseline vLLM experiments at concurrency 32, 64, 128 sequentially.
#
# Usage:
#   export CKPT_DIR=/path/to/Nemotron-Orchestrator-8B
#   bash scripts/5090/run_baseline_matrix.sh /workspace/ToolOrchestra/outputs/<EXPERIMENT_ID>
#
# Or to just run one at a time:
#   bash scripts/5090/launch_baseline.sh 32

OUTPUT_BASE="${1:?Usage: $0 /workspace/ToolOrchestra/outputs/<EXPERIMENT_ID>}"
TOOL_ORCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DOMAINS="${DOMAINS:-retail telecom airline}"
LOG_LEVEL="${LOG_LEVEL:-PROFILE}"

IDLE_TIMEOUT_S="${IDLE_TIMEOUT_S:-21600}"   # 6h
FORCE_RERUN="${FORCE_RERUN:-0}"

busy_snapshot() {
  echo "[runner] busy snapshot (ports/procs):"
  ss -lntp 2>/dev/null | egrep ":(1900)\\b" || true
  ps -ef | egrep "vllm serve|evaluation/tau2-bench/run_oss\\.py|collect_kv_cache_timeseries\\.sh" | grep -v egrep || true
}

wait_for_idle() {
  echo "[runner] waiting for idle (no vllm/run_oss processes)..."
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    if ps -ef | egrep -q "vllm serve|evaluation/tau2-bench/run_oss\\.py|collect_kv_cache_timeseries\\.sh" || \
       ss -lntp 2>/dev/null | egrep -q ":(1900)\\b"; then
      local now
      now="$(date +%s)"
      if (( now - start_ts > IDLE_TIMEOUT_S )); then
        echo "[runner] ERROR: timed out waiting for idle after ${IDLE_TIMEOUT_S}s"
        busy_snapshot
        return 1
      fi
      if (( (now - start_ts) % 60 == 0 )); then
        busy_snapshot
      fi
      sleep 10
    else
      break
    fi
  done
  echo "[runner] idle."
}

metrics_error_count() {
  local metrics_path="$1"
  python3 - <<'PY' "$metrics_path" 2>/dev/null || true
import json,sys
p=sys.argv[1]
try:
  with open(p) as f: d=json.load(f)
  print(d.get("error", ""))
except Exception:
  pass
PY
}

is_completed_bl() {
  local dest="$1"
  if [[ -f "${dest}/outputs/metrics_global.json" ]]; then
    local err
    err="$(metrics_error_count "${dest}/outputs/metrics_global.json")"
    [[ "${err}" == "0" ]]
    return
  fi
  return 1
}

move_latest() {
  local pattern="$1"
  local dest="$2"

  local latest
  latest="$(ls -1dt "${TOOL_ORCH_DIR}/outputs/${pattern}" 2>/dev/null | head -1 || true)"
  if [[ -z "${latest}" ]]; then
    echo "[runner] WARN: no output dir matched pattern ${pattern}"
    return 0
  fi

  mkdir -p "$(dirname "${dest}")"
  local final_dest="${dest}"
  if [[ -e "${final_dest}" ]]; then
    final_dest="${dest}_$(date +%Y%m%d_%H%M%S)"
  fi

  echo "[runner] moving ${latest} -> ${final_dest}"
  mv "${latest}" "${final_dest}"

  if [[ -f "${final_dest}/outputs/metrics_global.json" ]]; then
    echo "[runner] metrics: $(cat "${final_dest}/outputs/metrics_global.json" | tr -d '\n')"
  else
    echo "[runner] WARN: metrics_global.json not found under ${final_dest}/outputs/"
  fi
}

maybe_archive_existing() {
  local dest="$1"
  if [[ -e "${dest}" ]]; then
    local archived="${dest}_old_$(date +%Y%m%d_%H%M%S)"
    echo "[runner] archiving existing ${dest} -> ${archived}"
    mv "${dest}" "${archived}"
  fi
}

run_bl() {
  local conc="$1"
  echo "[runner] Baseline C=${conc}"
  local dest="${OUTPUT_BASE}/baseline_c${conc}"
  if [[ "${FORCE_RERUN}" != "1" ]] && is_completed_bl "${dest}"; then
    echo "[runner] skip (already completed): ${dest}"
    return 0
  fi
  if [[ -e "${dest}" ]]; then
    maybe_archive_existing "${dest}"
  fi
  /tmp/pre_experiment_check.sh
  export DOMAINS LOG_LEVEL
  wait_for_idle
  (
    set +u
    source /workspace/ToolOrchestra/setup_envs.sh 2>/dev/null || true
    set -u
    cd "${TOOL_ORCH_DIR}"
    bash scripts/5090/launch_baseline.sh "${conc}"
  )
  wait_for_idle
  move_latest "baseline_c${conc}_*" "${dest}"
}

echo "[runner] OUTPUT_BASE=${OUTPUT_BASE}"
mkdir -p "${OUTPUT_BASE}"

wait_for_idle

run_bl 32
wait_for_idle
run_bl 64
wait_for_idle
run_bl 128
wait_for_idle

echo "[runner] DONE - All baseline experiments completed"

