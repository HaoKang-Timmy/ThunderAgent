#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen2.5-32B.sh"

THUNDERAGENT_HOST=${THUNDERAGENT_HOST:-"127.0.0.1"}
THUNDERAGENT_PORT=${THUNDERAGENT_PORT:-"8300"}
THUNDERAGENT_LOG=${THUNDERAGENT_LOG:-"/root/slime/thunderagent_qwen2.5_32B.out"}

# Align with Qwen2.5-32B-Instruct HF config (rms_norm_eps=1e-6).
MODEL_OVERRIDE_ARGS=(
   --norm-epsilon 1e-6
)
ROLLOUT_DUMP_DIR=${ROLLOUT_DUMP_DIR:-"/root/slime/rollout_traj_qwen2.5_32B"}
TAU_TRAJ_JSON_DIR=${TAU_TRAJ_JSON_DIR:-"${ROLLOUT_DUMP_DIR}/samples_json"}
TAU_STEP_TIMING_PATH=${TAU_STEP_TIMING_PATH:-"${TAU_TRAJ_JSON_DIR}/step_timing.jsonl"}
ENABLE_PT_DEBUG_DUMP=${ENABLE_PT_DEBUG_DUMP:-0}

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen2.5-32B-Instruct/
   --ref-load /root/Qwen2.5-32B-Instruct_torch_dist/
   --load /root/Qwen2.5-32B-Instruct_slime/
   --save /root/Qwen2.5-32B-Instruct_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /root/tau-bench/retail_train_tasks.jsonl
   --input-key index
   --rollout-shuffle
   --num-rollout 2
   --rollout-batch-size 32
   --n-samples-per-prompt 24
   --rollout-max-response-len 2048
   --rollout-temperature 1
   --global-batch-size 768
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --balance-data
)

ENABLE_EVAL=${ENABLE_EVAL:-0}
EVAL_ARGS=()
if [ "${ENABLE_EVAL}" = "1" ]; then
   EVAL_ARGS=(
      --eval-interval 5
      --eval-prompt-data retail-dev /root/tau-bench/retail_dev_tasks.jsonl
      --n-samples-per-eval-prompt 1
      --eval-max-response-len 1024
      --eval-top-k 1
   )
fi

PERF_ARGS=(
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-tau-bench
   # --wandb-group qwen2.5-32B
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-enable-metrics
   --sglang-mem-fraction-static 0.85
   # If gemini API reports concurrency limit error, set this parameter to reduce the concurrency
   # --sglang-server-concurrency 32
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_tau.generate
)

DEBUG_ARGS=()
if [ "${ENABLE_PT_DEBUG_DUMP}" = "1" ]; then
   DEBUG_ARGS=(
      --save-debug-rollout-data "${ROLLOUT_DUMP_DIR}/{rollout_id}.pt"
   )
fi

ROUTER_ARGS=(
   --use-slime-router
   --sglang-router-ip "${THUNDERAGENT_HOST}"
   --sglang-router-port "${THUNDERAGENT_PORT}"
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# If you want more or less GPUs, change this parameter
NUM_GPUS=${NUM_GPUS:-8}
mkdir -p "${ROLLOUT_DUMP_DIR}"
mkdir -p "${TAU_TRAJ_JSON_DIR}"

pkill -f "python3 -m ThunderAgent" || true
sleep 1
PYTHONPATH="/root:${PYTHONPATH}" \
python3 -m ThunderAgent \
   --host 0.0.0.0 \
   --port "${THUNDERAGENT_PORT}" \
   --backends "" \
   --backend-type sglang \
   --router tr \
   --use-acting-token-decay \
   --log-level info \
   --profile \
   --profile-dir . \
   --metrics \
   --metrics-interval 2 \
   --scheduler-interval 2 \
   --enable-slime-adapter \
   > "${THUNDERAGENT_LOG}" 2>&1 &
THUNDERAGENT_PID=$!
sleep 2
if ! kill -0 "${THUNDERAGENT_PID}" 2>/dev/null; then
   echo "ThunderAgent failed to start, check ${THUNDERAGENT_LOG}" >&2
   exit 1
fi
trap 'kill "${THUNDERAGENT_PID}" 2>/dev/null || true' EXIT

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir /root/shared/ray_temp

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"TAU_TRAJ_JSON_DIR\": \"${TAU_TRAJ_JSON_DIR}\",
    \"TAU_STEP_TIMING_PATH\": \"${TAU_STEP_TIMING_PATH}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --rollout-num-gpus ${NUM_GPUS} \
   --colocate \
   ${ROUTER_ARGS[@]} \
   ${MODEL_ARGS[@]} \
   ${MODEL_OVERRIDE_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${DEBUG_ARGS[@]}
