# FRAMES OSS Full Evaluation - Experiment Log

**Date**: December 31, 2025
**Experiment**: FRAMES Benchmark Full Run with OSS Expert Models
**Tasks**: 824 multi-hop reasoning questions (Wikipedia corpus)

---

## 1. Overview

This experiment evaluates the Nemotron-Orchestrator-8B on the FRAMES benchmark using fully open-source expert models, with an optimized GPU-accelerated wiki retrieval service.

### Key Results

| Metric | Value |
|--------|-------|
| Total Tasks | 824 |
| Accuracy | TBD (check outputs) |
| Search E2E Latency (P50) | 3.79s |
| Search E2E Latency (P80) | 26.69s |
| Answer E2E Latency (P50) | 4.73s |
| <5s Search Calls | 56.1% |

---

## 2. Model Configuration

### Orchestrator
- **Model**: Nemotron-Orchestrator-8B
- **Deployment**: vLLM on research-secure-05 (4 GPUs, DP=4)
- **Ports**: 1800-1803

### Expert Models (OSS)
| Tool | Model | Node | Ports |
|------|-------|------|-------|
| Search (Query LLM) | openai/gpt-oss-20b | 172.27.31.10 | 1840-1843 |
| Answer | Qwen/Qwen3-32B-FP8 | research-secure-12 | 1820-1827 |
| Enhance Reasoning | Qwen/Qwen2.5-Coder-14B-Instruct | research-secure-11 | 1810-1817 |

### Wiki Retrieval Service
- **Embedding Model**: Qwen3-Embedding-8B
- **FAISS Index**: 97GB, GPU-sharded across 3 GPUs
- **Corpus**: 6.3M Wikipedia documents (17GB), preloaded into RAM
- **Node**: research-secure-19:8766

---

## 3. Reproduction Commands

### Step 1: Deploy vLLM Expert Servers

```bash
# Deploy on allocated SLURM nodes (see oss_scripts/ for details)
# Node 1: Orchestrator (research-secure-05)
sbatch oss_scripts/launch_orch.sh

# Node 2: Expert models (research-secure-11, 12)
sbatch oss_scripts/launch_experts_node2.sh

# Node 3: gpt-oss-20b (172.27.31.10)
# Already deployed via together.ai or custom vLLM
```

### Step 2: Deploy Optimized Wiki Retrieval

```bash
# Launch fast wiki retrieval with GPU FAISS + corpus preloading
sbatch evaluation/oss_scripts/slurm_launch_wiki_retrieval_fast.sh

# Verify deployment
curl -X POST http://research-secure-19:8766/retrieve \
  -H "Content-Type: application/json" \
  -d '{"queries": ["test query"], "topk": 5}'
```

### Step 3: Run FRAMES Evaluation

```bash
# Set up environment
source ~/haokang/ToolOrchestra/setup_envs.sh
conda activate vllm1
cd ~/haokang/ToolOrchestra/evaluation

# Run full evaluation (824 tasks, concurrency=32)
python -u eval_frames_oss.py \
  --model_name /home/junxiong/haokang/ckpt/nvidia/Nemotron-Orchestrator-8B/ \
  --model_config model_config_oss_new_wiki_fast19.json \
  --output_dir outputs_oss_frames_full_new \
  --concurrency 32 \
  2>&1 | tee ~/logs/oss_eval/frames_full_fast_run.log
```

### Step 4: Analyze Results

```bash
# Generate latency statistics
python analyze_tool_latency.py outputs_oss_frames_full_new

# Generate latency distribution plots
python logs_oss/frames_full_new_latency_plots/replot_frames_e2e_comprehensive.py
```

---

## 4. Key Files

| File | Description |
|------|-------------|
| `eval_frames_oss.py` | Main evaluation script |
| `model_config_oss_new_wiki_fast19.json` | Model endpoint configuration |
| `retrieval_wiki.py` | Wiki retrieval server with GPU FAISS + preloading |
| `analyze_tool_latency.py` | Latency analysis script |
| `oss_scripts/slurm_launch_wiki_retrieval_fast.sh` | SLURM deployment for fast retrieval |
| `stress_test_wiki_retrieval.py` | Retrieval stress testing utility |

---

## 5. Outputs

### Per-Task JSON Output
Each completed task produces a JSON file in `outputs_oss_frames_full_new/`:
```json
{
  "id": "wiki____0",
  "all_tool_calls": [...],
  "all_tool_responses": {
    "0": [{"tool": "search", "_query_llm_ms": 1234, "_retrieval_ms": 120, ...}],
    "1": [{"tool": "answer", "_expert_ms": 3200, "_judge_ms": 800, ...}]
  },
  "correct": true,
  "status": "done"
}
```

### Latency Profiling Fields
- `search`: `_query_llm_ms`, `_retrieval_ms`, `_retrieval_retries`
- `answer`: `_expert_ms`, `_judge_ms`
- `enhance_reasoning`: `_llm_ms`, `_exec_ms`

### Generated Plots
- `logs_oss/frames_full_new_latency_plots/frames_full_new_e2e_search.png`
- `logs_oss/frames_full_new_latency_plots/frames_full_new_e2e_answer.png`
- `logs_oss/frames_full_new_latency_plots/frames_full_new_e2e_enhance_reasoning.png`
- `logs_oss/frames_full_new_latency_plots/frames_full_new_e2e_all_tools.png`

---

## 6. Wiki Retrieval Optimization Details

### Problem
Original retrieval latency: **~280s per request** (HuggingFace datasets disk I/O)

### Solution (see WORKLOG_20251231.md)
1. **Corpus Preloading** (`WIKI_PRELOAD_CORPUS=1`): Load 6.3M docs into RAM at startup
2. **GPU FAISS Sharding** (`WIKI_FAISS_GPU=1`): 97GB index sharded across 3 GPUs
3. **Server-side Batching** (`WIKI_REQUEST_BATCHING=1`): Batch concurrent requests

### Result
- Single request: **P50=0.12s, P80=0.12s**
- 32 concurrent: **P50=0.30s, P80=0.33s**
- Throughput: **~100 requests/second**

---

## 7. Known Issues & Notes

1. **Search long-tail latency**: P90=42s is dominated by gpt-oss-20b query generation, not retrieval
2. **LLM-as-judge**: Uses proprietary gpt-5 API (requires CLIENT_ID/CLIENT_SECRET)
3. **Memory requirements**: Wiki retrieval needs ~100GB RAM for corpus + 150GB GPU memory for index

---

## 8. Environment Variables Required

```bash
export CLIENT_ID='...'        # For gpt-5 LLM-as-judge
export CLIENT_SECRET='...'    # For gpt-5 LLM-as-judge
export OPENAI_API_KEY='...'   # Alternative to CLIENT_ID/SECRET
export INDEX_DIR='/data/haokang/dataset/multi-train/index/'
```

Or source the setup script:
```bash
source ~/haokang/ToolOrchestra/setup_envs.sh
```
