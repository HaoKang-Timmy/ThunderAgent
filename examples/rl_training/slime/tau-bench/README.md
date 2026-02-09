# slime + tau-bench + ThunderAgent (RL Training Reproduction)

## Overview
This folder provides a reproducible pipeline for running slime RL training on tau-bench with ThunderAgent as the SGLang router replacement.

## Prerequisites
- NVIDIA GPU environment.
- Docker with `--gpus all`.
- Hugging Face access for model download.
- A LiteLLM-compatible API key for tau-bench user simulation (for example Gemini).

## Manual End-to-End Steps
### 1) Start container
```bash
docker pull slimerl/slime:latest
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it slimerl/slime:latest /bin/bash
```

### 2) Clone and install slime + tau-bench
```bash
cd /root/
git clone https://github.com/THUDM/slime.git
cd slime
pip install -e . --no-deps

cd /root/
git clone https://github.com/JD-ETH/tau-bench.git
cd tau-bench
pip install -e . --no-deps
```

### 3) Replace slime tau-bench example with ThunderAgent version
```bash
rsync -a --delete \
  /root/ThunderAgent/examples/rl_training/slime/tau-bench/ \
  /root/slime/examples/tau-bench/
```

### 4) Generate mock data
```bash
cd /root/slime/examples/tau-bench
python tau1_mock.py --local_dir /root/tau-bench/
```

### 5) Download and convert model checkpoint (Qwen2.5-32B)
```bash
huggingface-cli download Qwen/Qwen2.5-32B-Instruct --local-dir /root/Qwen2.5-32B-Instruct

cd /root/slime
source scripts/models/qwen2.5-32B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /root/Qwen2.5-32B-Instruct \
  --save /root/Qwen2.5-32B-Instruct_torch_dist
```

If you need a different model, replace model name and corresponding `scripts/models/*.sh` in the same pattern.

### 6) Configure LiteLLM API in tau-bench generator
Edit:

```bash
/root/slime/examples/tau-bench/generate_with_tau.py
```

Set your user-simulator model/provider and API key (for example `GEMINI_API_KEY`).

### 7) Start training
```bash
cd /root/slime
bash examples/tau-bench/run_qwen2.5_32B_thunderagent.sh
```

## What we changed in `slime/examples/tau-bench`
1. Per-sample `program_id` is explicitly generated and propagated.
   - Each `(prompt, sample)` is treated as one independent program.
   - The `program_id` is attached to `/generate` payloads for router-side program tracking.

2. Explicit program release is added at sample end.
   - On completed/truncated/aborted sample end, this version calls:
     - `POST /programs/release`
   - This avoids stale program state accumulation in ThunderAgent.

## Repository Layout
```text
tau-bench/
|-- README.md
|-- run_qwen2.5_32B_thunderagent.sh
|-- generate_with_tau.py
`-- tau1_mock.py
```
