#!/bin/bash
set -euo pipefail

# Print a ready-to-run SSH -L tunnel command for the 5090 box based on the latest
# rl_* HLE expert SLURM logs on this repo checkout (H100 head node).
#
# It reads:
#   slurm/hle_expert_logs/search/rl_hle_expert_search_<jobid>/node_ip.txt
#   slurm/hle_expert_logs/reasoner_answer/rl_hle_expert_reasoner_answer_<jobid>/node_ip.txt
#
# Usage (on H100 head node):
#   bash scripts/hle_serving/print_5090_tunnel_cmd.sh [HEAD_SSH_HOST]
#
# Then copy the printed ssh command to 5090 and run it under tmux.

HEAD_SSH_HOST="${1:-H100-Together}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OSS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${OSS_DIR}/.." && pwd)"

pick_latest_node_ip() {
  local glob="$1"
  local d
  d="$(ls -1dt ${glob} 2>/dev/null | head -n 1 || true)"
  if [[ -z "${d}" ]]; then
    echo ""
    return 0
  fi
  cat "${d}/node_ip.txt" 2>/dev/null | head -n 1 || true
}

SEARCH_IP="$(pick_latest_node_ip "${REPO_ROOT}/slurm/hle_expert_logs/search/rl_hle_expert_search_*/")"
RA_IP="$(pick_latest_node_ip "${REPO_ROOT}/slurm/hle_expert_logs/reasoner_answer/rl_hle_expert_reasoner_answer_*/")"

if [[ -z "${SEARCH_IP}" || -z "${RA_IP}" ]]; then
  echo "ERROR: could not find node_ip.txt for rl_* expert jobs." >&2
  echo "Expected directories under:" >&2
  echo "  ${REPO_ROOT}/slurm/hle_expert_logs/search/rl_hle_expert_search_<jobid>/" >&2
  echo "  ${REPO_ROOT}/slurm/hle_expert_logs/reasoner_answer/rl_hle_expert_reasoner_answer_<jobid>/" >&2
  exit 1
fi

cat <<EOF
# Run on 5090 (under tmux). Assumes you can SSH to the H100 head node as: ${HEAD_SSH_HOST}
ssh -N -o ExitOnForwardFailure=yes \\
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \\
  -L 1840:${SEARCH_IP}:1840 -L 1841:${SEARCH_IP}:1841 -L 1842:${SEARCH_IP}:1842 -L 1843:${SEARCH_IP}:1843 \\
  -L 1810:${RA_IP}:1810 -L 1811:${RA_IP}:1811 \\
  -L 1820:${RA_IP}:1820 -L 1821:${RA_IP}:1821 \\
  ${HEAD_SSH_HOST}

# Ready check (on 5090):
for p in 1840 1841 1842 1843 1810 1811 1820 1821; do echo -n "p=\$p "; curl -sf --max-time 2 http://127.0.0.1:\$p/health && echo OK || echo FAIL; done
EOF
