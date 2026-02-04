#!/bin/bash
set -euo pipefail

# Pre-flight checklist for 5090 experiments.
#
# Checks:
# - SSH tunnels for 8 expert ports
# - Expert /health endpoints reachable via localhost forwards
# - GPU memory status (warn if busy)

echo "=== Pre-Experiment Checklist ==="

echo ""
echo "[1] SSH tunnel ports:"
need_ports=(1910 1911 1912 1913 1904 1905 1920 1921)
missing=0
for p in "${need_ports[@]}"; do
  if ss -lntp 2>/dev/null | grep -qE "127\\.0\\.0\\.1:${p}\\b"; then
    echo "  - ${p}: OK"
  else
    echo "  - ${p}: MISSING"
    missing=$((missing + 1))
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  echo "ERROR: missing ${missing}/${#need_ports[@]} forwarded ports"
  exit 1
fi

echo ""
echo "[2] Expert /health endpoints:"
for p in 1910 1904 1920; do
  if curl -sf --max-time 5 "http://127.0.0.1:${p}/health" >/dev/null 2>&1; then
    echo "  - ${p}/health: OK"
  else
    echo "ERROR: ${p}/health not reachable"
    exit 1
  fi
done

echo ""
echo "[3] GPU status:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv || true
  procs="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l || true)"
  if [[ "${procs}" -gt 0 ]]; then
    echo "WARN: GPU has ${procs} compute process(es) running:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
  fi
else
  echo "WARN: nvidia-smi not found; skipping GPU checks"
fi

echo ""
echo "=== Checks passed ==="


