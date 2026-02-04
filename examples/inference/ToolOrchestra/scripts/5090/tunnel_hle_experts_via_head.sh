#!/bin/bash
set -euo pipefail

# Create SSH local port-forwards on the 5090 box so HLE eval can call
# H100 expert vLLM servers via localhost. This avoids exposing compute-node
# IPs to the public internet.
#
# Ports (localhost on 5090):
# - Search (openai/gpt-oss-20b): 1840-1843  (4 instances, TP=2)
# - Reasoner (Qwen/Qwen2.5-Coder-14B-Instruct): 1810-1811 (2 instances, TP=2)
# - Answer (Qwen/Qwen3-32B-FP8): 1820-1821 (2 instances, TP=2)
#
# You must provide:
#   HEAD_SSH_HOST (default: H100-Together)
#   SEARCH_NODE_IP
#   RA_NODE_IP
#
# Usage:
#   export SEARCH_NODE_IP=172.27.x.y
#   export RA_NODE_IP=172.27.a.b
#   bash scripts/5090/tunnel_hle_experts_via_head.sh
#
# Notes:
# - This command is expected to stay running; run it under tmux.
# - If you prefer background mode, set DETACH=1 (uses ssh -fN).

HEAD_SSH_HOST="${HEAD_SSH_HOST:-H100-Together}"
SEARCH_NODE_IP="${SEARCH_NODE_IP:?SEARCH_NODE_IP required}"
RA_NODE_IP="${RA_NODE_IP:?RA_NODE_IP required}"
DETACH="${DETACH:-0}"

common_opts=(
  -o ExitOnForwardFailure=yes
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=3
)

forward_opts=(
  -L "1840:${SEARCH_NODE_IP}:1840" -L "1841:${SEARCH_NODE_IP}:1841" -L "1842:${SEARCH_NODE_IP}:1842" -L "1843:${SEARCH_NODE_IP}:1843"
  -L "1810:${RA_NODE_IP}:1810" -L "1811:${RA_NODE_IP}:1811"
  -L "1820:${RA_NODE_IP}:1820" -L "1821:${RA_NODE_IP}:1821"
)

echo "[tunnel_hle] Head SSH host: ${HEAD_SSH_HOST}"
echo "[tunnel_hle] search node: ${SEARCH_NODE_IP} ports 1840-1843 -> localhost:1840-1843"
echo "[tunnel_hle] reasoner+answer node: ${RA_NODE_IP} ports 1810-1811,1820-1821 -> localhost"

if [[ "${DETACH}" == "1" ]]; then
  echo "[tunnel_hle] Starting in background (ssh -fN)."
  ssh -fN "${common_opts[@]}" "${forward_opts[@]}" "${HEAD_SSH_HOST}"
  echo "[tunnel_hle] Started. Verify with: ss -lntp | egrep ':(1840|1841|1842|1843|1810|1811|1820|1821) '"
else
  echo "[tunnel_hle] Starting in foreground (ssh -N). Run this under tmux; Ctrl-C to stop."
  exec ssh -N "${common_opts[@]}" "${forward_opts[@]}" "${HEAD_SSH_HOST}"
fi

