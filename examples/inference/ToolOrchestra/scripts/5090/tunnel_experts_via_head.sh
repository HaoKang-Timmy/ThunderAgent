#!/bin/bash
set -euo pipefail

# Create SSH local port-forwards on the 5090 box so tau2-bench can call
# Together H100 experts via localhost. This avoids exposing compute-node
# IPs to the public internet.
#
# NOTE:
# - The IP defaults below are just historical examples and may be stale.
# - For the TAU2Bench 5090 inference experiment, prefer generating a fresh command
#   from the H100 head node via:
#     `bash oss-ToolOrchestra/scripts/experts/print_5090_tunnel_cmd_tau2_5090.sh H100-Together`
#   then copy/paste to 5090 under tmux.
#
# You can override via env vars:
#   HEAD_SSH_HOST=H100-Together
#   EXPERT_1_IP=...
#   EXPERT_2_IP=...
#   EXPERT_3_IP=...
#
# Usage:
#   bash scripts/5090/tunnel_experts_via_head.sh
#
# Notes:
# - This command is expected to stay running; run it under tmux.
# - If you prefer background mode, set DETACH=1 (uses ssh -fN).

HEAD_SSH_HOST="${HEAD_SSH_HOST:-H100-Together}"
EXPERT_1_IP="${EXPERT_1_IP:-172.27.27.153}"
EXPERT_2_IP="${EXPERT_2_IP:-172.27.25.244}"
EXPERT_3_IP="${EXPERT_3_IP:-172.27.19.213}"
DETACH="${DETACH:-0}"

common_opts=(
  -o ExitOnForwardFailure=yes
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=3
)

forward_opts=(
  -L "1910:${EXPERT_1_IP}:1910" -L "1911:${EXPERT_1_IP}:1911" -L "1912:${EXPERT_1_IP}:1912" -L "1913:${EXPERT_1_IP}:1913"
  -L "1904:${EXPERT_2_IP}:1904" -L "1905:${EXPERT_2_IP}:1905" -L "1906:${EXPERT_2_IP}:1906"
  -L "1920:${EXPERT_3_IP}:1920"
)

echo "[tunnel] Head SSH host: ${HEAD_SSH_HOST}"
echo "[tunnel] expert-1: ${EXPERT_1_IP} ports 1910-1913 -> localhost:1910-1913"
echo "[tunnel] expert-2: ${EXPERT_2_IP} ports 1904-1906 -> localhost:1904-1906"
echo "[tunnel] expert-3: ${EXPERT_3_IP} port 1920 -> localhost:1920"

if [[ "${DETACH}" == "1" ]]; then
  echo "[tunnel] Starting in background (ssh -fN)."
  ssh -fN "${common_opts[@]}" "${forward_opts[@]}" "${HEAD_SSH_HOST}"
  echo "[tunnel] Started. Verify with: ss -lntp | egrep ':(1910|1911|1912|1913|1904|1905|1906|1920) '"
else
  echo "[tunnel] Starting in foreground (ssh -N). Run this under tmux; Ctrl-C to stop."
  exec ssh -N "${common_opts[@]}" "${forward_opts[@]}" "${HEAD_SSH_HOST}"
fi
