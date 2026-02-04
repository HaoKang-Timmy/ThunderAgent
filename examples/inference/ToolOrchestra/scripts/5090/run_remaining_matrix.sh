#!/bin/bash
set -euo pipefail

# Run the remaining experiment matrix sequentially on the 5090 box.
# Intended usage: after TR C=32 has been started (or completed), run the rest:
#   bash scripts/5090/run_remaining_matrix.sh /path/to/OUTPUT_BASE
#
# Notes:
# - This script waits for GPU/ports to be idle between runs.
# - Each run is moved into OUTPUT_BASE/{thunderreact,continuum}_c{...}/

OUTPUT_BASE="${1:?Usage: $0 /workspace/ToolOrchestra/outputs/<EXPERIMENT_ID>}"
TOOL_ORCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DOMAINS="${DOMAINS:-retail telecom airline}"
LOG_LEVEL="${LOG_LEVEL:-PROFILE}"

IDLE_TIMEOUT_S="${IDLE_TIMEOUT_S:-21600}"   # 6h
FORCE_RERUN="${FORCE_RERUN:-0}"            # 1 to rerun even if already completed

busy_snapshot() {
  echo "[runner] busy snapshot (ports/procs):"
  ss -lntp 2>/dev/null | egrep ":(8000|8100|1900)\\b" || true
  ps -ef | egrep "vllm serve .*--port (8100|1900)|python multinode_router\\.py|evaluation/tau2-bench/run_oss\\.py|collect_kv_cache_timeseries\\.sh" | grep -v egrep || true
}

wait_for_idle() {
  echo "[runner] waiting for idle (no vllm/router/run_oss processes)..."
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    if ps -ef | egrep -q "vllm serve .*--port (8100|1900)|python multinode_router\\.py|evaluation/tau2-bench/run_oss\\.py|collect_kv_cache_timeseries\\.sh" || \
       ss -lntp 2>/dev/null | egrep -q ":(8000|8100|1900)\\b"; then
      local now
      now="$(date +%s)"
      if (( now - start_ts > IDLE_TIMEOUT_S )); then
        echo "[runner] ERROR: timed out waiting for idle after ${IDLE_TIMEOUT_S}s"
        busy_snapshot
        return 1
      fi
      # Print a snapshot occasionally so it's obvious what's blocking.
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
  # Print ".error" from metrics_global.json; empty if unreadable.
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

is_completed_tr() {
  local dest="$1"
  if [[ -f "${dest}/outputs/metrics_global.json" ]]; then
    local err
    err="$(metrics_error_count "${dest}/outputs/metrics_global.json")"
    [[ "${err}" == "0" ]]
    return
  fi
  return 1
}

is_completed_ct() {
  local dest="$1"
  if [[ -f "${dest}/outputs/metrics_global.json" && -f "${dest}/scheduler_timestamps" ]]; then
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

  # IMPORTANT: do NOT pre-create dest directory, otherwise `mv src dest` will
  # move src INTO dest (nested), instead of renaming.
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

run_tr() {
  local conc="$1"
  echo "[runner] ThunderReact C=${conc}"
  local dest="${OUTPUT_BASE}/thunderreact_c${conc}"
  if [[ "${FORCE_RERUN}" != "1" ]] && is_completed_tr "${dest}"; then
    echo "[runner] skip (already completed): ${dest}"
    return 0
  fi
  if [[ -e "${dest}" ]]; then
    maybe_archive_existing "${dest}"
  fi
  /tmp/pre_experiment_check.sh
  export DOMAINS LOG_LEVEL
  wait_for_idle
  (cd "${TOOL_ORCH_DIR}" && bash scripts/5090/launch_thunderreact.sh "${conc}")
  wait_for_idle
  move_latest "thunderreact_c${conc}_*" "${dest}"
}

run_ct() {
  local conc="$1"
  echo "[runner] Continuum C=${conc}"
  local dest="${OUTPUT_BASE}/continuum_c${conc}"
  if [[ "${FORCE_RERUN}" != "1" ]] && is_completed_ct "${dest}"; then
    echo "[runner] skip (already completed): ${dest}"
    return 0
  fi
  if [[ -e "${dest}" ]]; then
    maybe_archive_existing "${dest}"
  fi
  /tmp/pre_experiment_check.sh
  export DOMAINS LOG_LEVEL
  wait_for_idle
  (cd "${TOOL_ORCH_DIR}" && bash scripts/5090/launch_continuum.sh "${conc}")
  wait_for_idle
  move_latest "continuum_c${conc}_*" "${dest}"
}

echo "[runner] OUTPUT_BASE=${OUTPUT_BASE}"
mkdir -p "${OUTPUT_BASE}"

# If a run is currently active (e.g., TR32 already started), wait for it to finish first.
wait_for_idle

# If TR32 has been run outside the runner, move it into OUTPUT_BASE for consistency.
if [[ "${FORCE_RERUN}" != "1" ]] && [[ -e "${OUTPUT_BASE}/thunderreact_c32" ]]; then
  echo "[runner] thunderreact_c32 already present under OUTPUT_BASE; not moving"
else
  if ls -1dt "${TOOL_ORCH_DIR}/outputs/thunderreact_c32_"* >/dev/null 2>&1; then
    if [[ -e "${OUTPUT_BASE}/thunderreact_c32" ]]; then
      maybe_archive_existing "${OUTPUT_BASE}/thunderreact_c32"
    fi
    move_latest "thunderreact_c32_*" "${OUTPUT_BASE}/thunderreact_c32"
  fi
fi

# Remaining runs (TR64, TR128, CT32, CT64, CT128)
run_tr 64
wait_for_idle
run_tr 128
wait_for_idle
run_ct 32
wait_for_idle
run_ct 64
wait_for_idle
run_ct 128
wait_for_idle

echo "[runner] DONE"


