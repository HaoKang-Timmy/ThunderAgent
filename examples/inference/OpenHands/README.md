# OpenHands (code) + ThunderAgent Use Case

## Overview
This folder contains reproducible end-to-end guides for running SWE-bench evaluations with **ThunderAgent + vLLM** across  **OpenHands (code)**.

![ThunderAgent overview](../docs/thunder/figures/thunder.jpg)

## Prerequisites
- Python 3.10 -- 3.13
- [uv](https://docs.astral.sh/uv/) package manager
- Docker daemon (SWE-bench instances run in Docker)
- GPU(s) with appropriate CUDA drivers

## Reproduction
These scripts reproduce end-to-end runs with **vLLM + ThunderAgent + OpenHands (code)**.

Test hardware: 8x H100.

Environment setup: [`setup.sh`](scripts/setup/setup.sh).

- [`reproduce_glm4.6`](scripts/reproduce/reproduce_glm4.6): reproduce with GLM-4.6-FP8.
- [`reproduce_qwen3_235B`](scripts/reproduce/reproduce_qwen3_235B): reproduce with Qwen3-235B-A22B-Instruct-2507.

Throughput measurement: we count the total number of served LLM calls ("steps") within a stable serving window (e.g., from 10 minutes after startup to 1 hour 10 minutes after startup), then divide by the window duration to get throughput (steps/min).

Expected result:
![throughput_compare](../docs/openhands/figures/throughput_compare.png)

## Setup
```bash
# Create and activate env
uv venv --python 3.12
source .venv/bin/activate

# Install vLLM (GPU build) and OpenHands (code) in editable mode
uv pip install vllm --torch-backend=auto
uv pip install -e examples/inference/OpenHands
```

## How to run the experiment yourself

### One node example
1) Start vLLM to serve the model:
```bash
vllm serve <MODEL_NAME> --tensor-parallel-size <NUM_GPUS> --enable-auto-tool-choice --tool-call-parser <TOOL_PARSER> --port <VLLM_PORT>
```

2) Start ThunderAgent (pointing at your vLLM backend):
```bash
python -m ThunderAgent --backends http://localhost:<VLLM_PORT> --port <TA_PORT>
```

3) Configure `examples/inference/OpenHands/config.toml`.
   - `--llm-config vllm_local` selects the `[llm.vllm_local]` section in this file.
   - Set `base_url` to `http://localhost:<TA_PORT>/v1` so OpenHands sends OpenAI-compatible requests to ThunderAgent (instead of directly to vLLM). Keep the `/v1` suffix.
   - Set `model` to the same model you serve in vLLM (e.g., an absolute local path like `/data/models/...` or a served model name).
   - For local ThunderAgent, `custom_llm_provider = "openai"` and `api_key = "dummy"` are OK.

4) Run SWE-Bench via OpenHands through ThunderAgent:
```bash
cd examples/inference
python -m evaluation.benchmarks.swe_bench.run_infer \
  --config-file OpenHands/config.toml \
  --llm-config vllm_local \
  --agent-cls CodeActAgent \
  --dataset princeton-nlp/SWE-bench_Lite \
  --split test \
  --max-iterations 50 \
  --eval-num-workers 128 \
  --eval-output-dir test
```

### Multi nodes example
1) Start vLLM to serve the model:
```bash
vllm serve <MODEL_NAME> --tensor-parallel-size <NUM_GPUS> --enable-auto-tool-choice --tool-call-parser <TOOL_PARSER> --host 0.0.0.0 --port <VLLM_PORT>
```

```bash
vllm serve <MODEL_NAME> --tensor-parallel-size <NUM_GPUS> --enable-auto-tool-choice --tool-call-parser <TOOL_PARSER> --host 0.0.0.0 --port <VLLM_PORT>
```

2) Start ThunderAgent (pointing at your vLLM backend):
```bash
python -m ThunderAgent --backends http://<VLLM_HOST1>:<VLLM_PORT>,http://<VLLM_HOST2>:<VLLM_PORT> --port <TA_PORT>
```

3) Configure `examples/inference/OpenHands/config.toml`.
   - Same as the single-node setup: point `base_url` to `http://<TA_HOST>:<TA_PORT>/v1` (where `<TA_HOST>` is reachable from your runner node), set `model` accordingly, and keep `custom_llm_provider = "openai"`.

4) Run SWE-Bench via OpenHands through ThunderAgent:
```bash
cd examples/inference
python -m evaluation.benchmarks.swe_bench.run_infer \
  --config-file OpenHands/config.toml \
  --llm-config vllm_local \
  --agent-cls CodeActAgent \
  --dataset princeton-nlp/SWE-bench_Lite \
  --split test \
  --max-iterations 50 \
  --eval-num-workers 128 \
  --eval-output-dir test
```

## What we changed in OpenHands(to reuse in your own agent workflow)
- **Program ID injection**  
  Location: [`run_infer.py`](evaluation/benchmarks/swe_bench/run_infer.py#L612) (`process_instance()`), plus [`llm.py`](openhands/llm/llm.py#L330).  

  How it works: OpenHands routes model calls through `openhands/llm/llm.py`. By exporting `OPENHANDS_PROGRAM_ID` per SWE-bench instance in `run_infer.py`, we inject `extra_body.program_id` into every request payload. ThunderAgent uses this field to separate requests into per-program routing state.  

  ```python
  # openhands/llm/llm.py (inject into each request)
  thunderagent_program_id = os.environ.get("OPENHANDS_PROGRAM_ID")
  if thunderagent_program_id:
      extra_body = kwargs.get("extra_body")
      extra_body = extra_body.copy() if isinstance(extra_body, dict) else {}
      extra_body["program_id"] = thunderagent_program_id
      kwargs["extra_body"] = extra_body

  resp = self._completion_unwrapped(*args, **kwargs)
  ```

  ```python
  # run_infer.py: create per-instance program_id and export it
  digest = hashlib.sha1(f"{instance_id}:{os.getpid()}".encode("utf-8")).hexdigest()
  program_id = f"swe-{digest[:16]}"
  os.environ["OPENHANDS_PROGRAM_ID"] = program_id
  ```

  **vLLM** vs **ThunderAgent**: the only request difference we rely on is adding `extra_body.program_id=<program_id>` on every call.

- **Program release hook**  
  Location: [`run_infer.py`](evaluation/benchmarks/swe_bench/run_infer.py#L753) (`finally:`).  

  How it works: After the instance finishes (success or failure), send `POST /programs/release` to ThunderAgent with the same `program_id` to free router-side bookkeeping (tokens / pause-resume state).  

  Implementation snippet:
  ```python
  # base_url is the same one used for LLM calls (usually ends with /v1)
  # POST {base_url_without_/v1}/programs/release with {"program_id": program_id}
  _release_thunderagent_program(metadata.llm_config.base_url, program_id)
  ```

## Repository Layout

This tree summarizes the key paths used by this guide:
- `OpenHands/` contains the OpenHands (code) checkout used in this guide: `config.toml` is the ThunderAgent+vLLM LLM config, `scripts/` has runnable helpers (`setup/` and `reproduce/`), and `evaluation/benchmarks/swe_bench/` is the SWE-bench harness entry.


```text
OpenHands/
|-- config.toml
|-- scripts/
|   |-- setup/
|   |   `-- setup.sh
|   `-- reproduce/
|       |-- reproduce_glm4.6
|       `-- reproduce_qwen3_235B
|-- evaluation/
|   `-- benchmarks/
|       `-- swe_bench/
`-- openhands/
    `-- llm/
        `-- llm.py
```
