# HLE Local Evaluation + Profiling (Refactor Notes)

This document summarizes the **workspace changes** made after the last git commit in `ToolOrchestra` (`7eae53c`) to enable **local** HLE evaluation (2x RTX 5090) while keeping the original HLE logic aligned (especially retrieval + Tavily fallback).

> Note: `cursor_hle_evaluation_script_resource.md` (in `/workspace`) is an external “work log / design transcript” and is **not tracked** by this repo. This README summarizes the practical outcomes of that work log.

---

## What Changed vs `7eae53c`

### Modified (tracked) files

- `.gitignore`
  - Ignores `evaluation/cache/*` and `*.arrow` runtime artifacts.
  - Ignores `setup_envs.sh` to prevent accidental commits of secrets.

- `LLM_CALL.py`
  - Added support for **Together-hosted OSS models** via Together’s OpenAI-compatible API (`base_url=https://api.together.xyz/v1`).
  - Adds a **legacy/back-compat** path for `model="endpoint-..."` (Together dedicated endpoint IDs).
  - Also accepts `model_type='together'` (same behavior as `model_type='nv/dev'`).
  - Keeps existing routing: local vLLM for `model_type='vllm'`, and **Nebius** routing for `Qwen3-32B` when `NEBIUS_API_KEY` is set.

- `evaluation/retrieval_hle.py`
  - Added `/health` endpoint (for readiness checks).
  - Made startup logs more explicit (index/corpus/model loading).
  - Hardened HF download behavior: if `HF_HUB_ENABLE_HF_TRANSFER=1` but `hf_transfer` is missing, auto-disables fast-download to avoid startup crash.
  - Validates `INDEX_DIR` is set (must contain `eval.index` + `eval.jsonl`).
  - Added a global lock around the GPU-backed retrieval critical section to avoid FAISS GPU stack allocator assertion failures under high concurrency.
  - Retrieval behavior is unchanged: if fewer than **3** retrieved docs meet `(score > 0.1 && len(content) > 100)`, it falls back to **Tavily** web search.

- `evaluation/retriever_env_setup.md`
  - Expanded retriever env setup instructions and added troubleshooting for `numpy/scipy/faiss` ABI conflicts.
  - Added a note about the `hf_transfer` fast-download crash and two ways to fix it.

- `evaluation/tau2-bench/RUN_LOCAL_GUIDE.md`
  - Minor local-running documentation adjustments (non-functional).

### Added (tracked) local runner + profiling utilities

- `evaluation/run_hle_local.py`
  - Local HLE runner that:
    - starts retrieval server (GPU1, `retriever` conda env)
    - starts Orchestrator vLLM server (GPU0, `vllm1` conda env)
    - writes `evaluation/model_configs/hle_local.json`
    - runs `evaluation/eval_hle_local.py`
  - Does **not** use `conda run`; it uses: `source conda.sh && conda activate <env> && exec ...`
  - Defaults to **reusing already-running servers** if ports respond:
    - retrieval: `http://127.0.0.1:<retrieval_port>/openapi.json`
    - orchestrator: `http://127.0.0.1:<orchestrator_port>/health`
  - Prints a red overall ETA in the runner by watching `[HLE_TASK_COMPLETE]` markers.

- `evaluation/eval_hle_local.py`
  - Local version of HLE eval script (same tool/retrieval structure as `eval_hle.py`), with backend routing:
    - Orchestrator: local vLLM (started by `run_hle_local.py`)
    - `Qwen/Qwen3-32B`: Nebius API when `NEBIUS_API_KEY` is set (implemented in `LLM_CALL.py`)
    - Other open-source “expert” models: Together (serverless turbo or dedicated endpoints)
  - Adds structured profiling logs + configurable concurrency.

- `evaluation/hle_logging.py`
  - tau2-bench style structured logging with custom levels `PROFILE` and `USER_JUDGE`.

- `evaluation/analyze_hle_timing.py`
  - Post-run analysis: parses `hle.log` and writes stats + **10-bin linear and log-scale** histograms.

- `evaluation/run_hle_profile.sh`
  - Launches the full eval in the background and runs `analyze_hle_timing.py` after completion.

- `evaluation/check_hle_duplicate_ids.py`
  - Utility to detect duplicate `id` values in an HLE jsonl.

- `evaluation/hle_20.jsonl`
  - 20-task subset for smoke testing.

- `evaluation/model_configs/hle_local.json`
  - Example local model config pointing retrieval -> `127.0.0.1:1401` and orchestrator -> `127.0.0.1:1406`.

- `test_together_endpoint_math.py`
  - One-off sanity test for Together dedicated endpoint **Name** strings.

- `START_LOCAL.md`
  - Quick local setup notes (no secrets; export keys in your shell).

- `setup_envs.sh` (repo root, untracked)
  - Convenience env var setup (CKPT_DIR/REPO_PATH/HF_HOME/TAVILY_KEY/TOGETHER_API_KEY/NEBIUS_API_KEY, etc.).
  - Kept **untracked** (contains secrets). Create your own locally.

### Runtime artifacts (untracked)

These directories/files are generated while running locally:

- `evaluation/logs/hle_local/` (stdout/err logs for retrieval + vLLM)
- `evaluation/model_configs/` (generated JSON model configs)
- `evaluation/cache/` and `evaluation/cache/hle/` (HF/datasets + retrieval caches)

---

## Quick Start (Local)

### 0) Prepare environment variables

From `/workspace/ToolOrchestra`:

```bash
source setup_envs.sh
```

Important env vars:

- `CKPT_DIR`: local Orchestrator checkpoint path
- `INDEX_DIR`: directory containing `eval.index` and `eval.jsonl`
- `TAVILY_KEY`: required if you want retrieval fallback-to-web to work
- `TOGETHER_API_KEY`: required for Together serverless / dedicated endpoints
- `NEBIUS_API_KEY`: required to route `Qwen/Qwen3-32B` to Nebius

### 1) Retrieval smoke test (GPU1)

```bash
python evaluation/run_hle_local.py \
  --smoke-retrieval \
  --index-dir /workspace/dataset/multi-train/index
```

Notes:
- This starts `retrieval_hle.py` and runs one `/retrieve` request, then exits.
- If a retrieval server is already running on the same port, it will be reused unless you pass `--no-reuse-running`.

### 2) Full local run (retrieval + orchestrator)

```bash
python evaluation/run_hle_local.py \
  --agent-model "$CKPT_DIR" \
  --index-dir /workspace/dataset/multi-train/index
```

---

## Useful Flags (`evaluation/run_hle_local.py`)

- `--conda-sh <path>`: path to `conda.sh` if auto-detect fails.
- `--no-reuse-running`: always start fresh servers (do not reuse ports).
- `--skip-server-start`: assume servers already running; only run eval.
- `--skip-retrieval-start` / `--skip-orchestrator-start`: start only one side.
- `--skip-eval`: start servers and block (Ctrl-C to stop).
- Ports / GPUs:
  - `--retrieval-gpu` (default `1`), `--retrieval-port` (default `1401`)
  - `--orchestrator-gpu` (default `0`), `--orchestrator-port` (default `1406`)

---

## Logs & Debugging

- Retrieval logs:
  - `evaluation/logs/hle_local/retrieval_hle.out`
  - `evaluation/logs/hle_local/retrieval_hle.err`
- Orchestrator vLLM logs:
  - `evaluation/logs/hle_local/orchestrator_port_*.out`
  - `evaluation/logs/hle_local/orchestrator_port_*.err`

Tail logs:

```bash
tail -f evaluation/logs/hle_local/retrieval_hle.out
tail -f evaluation/logs/hle_local/orchestrator_port_1406_*.out
```

---

## Common Issues

### 1) GPU memory “not released” after smoke test

Root cause was typically an orphaned child process (wrapper shells), leaving Python/vLLM alive.

Mitigations implemented:
- `run_hle_local.py` starts subprocesses in a new session and kills the **process group** on cleanup.
- Default `--reuse-running` avoids accidentally starting duplicates on the same ports.

### 2) `hf_transfer` crash (HF_HUB_ENABLE_HF_TRANSFER=1)

Symptom:
- `ValueError: Fast download using 'hf_transfer' is enabled ... but 'hf_transfer' package is not available`

Mitigation:
- `evaluation/retrieval_hle.py` and `evaluation/eval_hle_local.py` now auto-disable `HF_HUB_ENABLE_HF_TRANSFER` if `hf_transfer` is missing.

Alternative fix:

```bash
conda activate vllm1
pip install hf_transfer
```

### 3) Together dedicated endpoint 404 (`model_not_available`)

Symptom:
- `Unable to access model ... model_not_available`

Cause:
- The dedicated endpoint is not started / not accessible to the API key, **or** you are calling it with the wrong `model` string.

Fix:
- Ensure you pass the endpoint **Name** (e.g. `HK123/...`) as `model` (see `/workspace/HLE-expert-lm.md`), start the dedicated endpoint in Together, or switch to a serverless model.

### 4) FAISS GPU crash under high concurrency (`StackDeviceMemory.cpp:144`)

Symptom:
- `Faiss assertion 'p + size == head_' failed ... StackDeviceMemory.cpp:144`

Cause:
- Multiple CPU threads concurrently calling a shared FAISS GPU index + a shared torch embedding model (not thread-safe).

Mitigation implemented (simplest / safest):
- `evaluation/retrieval_hle.py` guards the critical section with a global lock.

Alternative (better throughput) option:
- Implement **micro-batching** in the retrieval server: queue per-request queries, then run one `encode(batch)` + one `faiss.search(batch)`, then split results back to callers.

---

## Notes on Data

`evaluation/hle.jsonl` contains benchmark data; treat it as evaluation-only. In many checkouts this file is a Git-LFS pointer; pass the real dataset path via `--example-path` when running locally.

