# mini-swe-agent + ThunderAgent Use Case

## Prerequisites
- Python 3.10 -- 3.13
- [uv](https://docs.astral.sh/uv/) package manager
- GPU(s) with appropriate CUDA drivers

## 1. Intro
Run the official [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) on SWE-Bench while routing model traffic through ThunderAgent to manage vLLM capacity and per-task program tracking.

## 2. Software Setup (uv)
```bash
# Create and activate env
uv venv --python 3.12
source .venv/bin/activate

# Install vLLM (GPU build), mini-swe-agent in editable mode, and datasets
uv pip install vllm --torch-backend=auto
uv pip install -e examples/inference/mini-swe-agent
uv pip install datasets
```

### How to run the experiment(One node example)
1) Start vLLM to serve the model:
```bash
vllm serve <MODEL_NAME> --tensor-parallel-size <NUM_GPUS> --enable-auto-tool-choice --tool-call-parser <TOOL_PARSER> --port <VLLM_PORT>
```
2) Start ThunderAgent (pointing at your vLLM backend):
```bash
python -m ThunderAgent --backends http://localhost:<VLLM_PORT> --port <TA_PORT>
```
3) Run SWE-Bench via mini-swe-agent through ThunderAgent:
```bash
mini-extra swebench \
  --subset lite \
  --split test \
  --workers 128 \
  --output ./swebench_output 
```

## 3. What we changed (to reuse in your own setup)
- **Per-task program_id injection:** Each SWE-Bench instance gets a unique `program_id` (in `swebench.py`), automatically sent via `model_kwargs.extra_body` so ThunderAgent can track capacity per task.
- **Program release hook:** After each instance finishes, `swebench.py` calls `/programs/release` on the same `api_base` to free ThunderAgent resources.
-  **Tool resource release:**


