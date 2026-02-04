# 2025-12-28 本地评测相关增量变更说明（vs `origin/main`）

> **对比基线**：`origin/main` = `d5cf5b5`  
> **本说明覆盖的 3 个 commit**：`4c75418`、`4c3b777`、`f455a8c`  
> **更早的背景/整体架构**：如果需要，可参考  
> - `evaluation/HLE_progress_20251227_072936.md`（更大范围、包含更早 staged 变更的设计/问题记录）  
> - `evaluation/HLE_README.md`（本地 HLE eval+profiling 的整体说明/quickstart）  

---

## 1) TL;DR（这 3 个 commit 带来的核心能力）

- **HLE retrieval server 的稳定性 + 吞吐量**：
  - 早期“全局锁”修复基础上，升级为 **服务端 micro-batching**：把大量并发 `/retrieve` 请求合并成一次 `encode(batch)+faiss.search(batch)`，然后按请求拆分返回（减少 FAISS GPU 非线程安全问题，同时显著减少 search 调用次数）。
  - **Tavily fallback 更安全**：当没有配置 key 时不会尝试调用 Tavily（避免无 key 时的 crash）。
  - **k/batch 策略可配置**：`--faiss_k` / `--max_batch_size` / `--max_wait_ms` 或对应环境变量。

- **HLE 评测可恢复（resume）**：
  - `eval_hle_local.py` **遇到单个 task 异常不会中止整跑**（`return_exceptions=True`）。
  - 同一个 `--output-dir` 下已经写过 `{id}.json` 的任务会 **自动 skip** 并仍然打印 `[HLE_TASK_COMPLETE] status=skipped`，便于 runner/monitor 继续推进 ETA。

- **更完整的 profiling 分析链路**：
  - `analyze_hle_timing.py` 新增 `--all-runs`：如果同一个 `hle.log` 里 append 了多次 run，默认只分析最后一次；也可以分析全量 run。
  - 新增两份“今天写的分析脚本”：
    - `evaluation/analyze_tool_model_usage.py`：统计 HLE tools（`enhance_reasoning/search/answer`）分别用了哪些模型（GPT-5/GPT-5-mini vs OSS 模型）。
    - `evaluation/analyze_specific_tool_duration.py`：针对某个 tool+model 组合，做 duration 分布/分位数统计并生成直方图。

- **tau2-bench 本地跑更稳**：
  - `evaluation/tau2-bench/run_full_c48_profile.sh` 现在会：
    - 启动前主动清理遗留进程（vLLM / tau2 / retrieval）
    - **拉起 retrieval 服务**（GPU1、端口 1401、带 health check）
    - 把 `model_config_local.json` 固定写到当前 run 的 `LOG_DIR`，便于归档与复现
  - `evaluation/tau2-bench/run_local.py` 增强：
    - vLLM server 的 stdout/err 落到 `--log-dir`
    - `TAU2_LOG_FILE` 按 domain 写入 `tau2_{domain}.log`（append，便于中断恢复）
    - 非阻塞读子进程输出 + 心跳 + 总体 ETA（红字 `[OVERALL_ETA]`）

---

## 2) Commit-by-commit 变更清单

### 2.1 `4c75418` — Tavily fallback 更安全（retrieval 端）

**改动文件**：`evaluation/retrieval_hle.py`

- **变更点**：只有在 `config.tavily_key` 存在时才进行 Tavily fallback（否则直接返回当前已有检索结果）
- **目的**：避免未设置 Tavily key 时，retrieval 在“doc 不足”分支触发 Tavily 调用导致异常/浪费重试。

---

### 2.2 `4c3b777` — retrieval server micro-batching + 降开销

**改动文件**：`evaluation/retrieval_hle.py`

#### A. 服务端 micro-batching（核心）

- 新增 `BatchQueue` + 后台 worker：
  - 请求到 `/retrieve` 后不再直接在 FastAPI worker thread 里跑 GPU 检索
  - 而是把请求入队，后台 worker 按策略组合成 batch：
    - **满 batch 立即 flush**（`max_batch_size`）
    - **否则等待最长 `max_wait_ms`**（把同时间窗内的并发请求聚合）
  - batch 侧只做一次：
    - `encoder.encode(queries)`（GPU）
    - `faiss.index.search(batch_emb, k)`（GPU）
  - 然后逐请求做 `eid` 过滤、格式化结果并回填 Future（返回给对应 HTTP 请求）。

#### B. 性能/稳定性细节

- 移除每次 encode/search 后的 `torch.cuda.empty_cache()`（这个在高并发下会非常伤吞吐）
- 关闭 tqdm 输出（server path 下 `disable=True`，避免刷屏与额外开销）
- `example_ids[eid]` 从 list 转成 set，加速 membership check（过滤时 `doc_id in allowed_ids`）
- 新增 `raw_batch_search()`：提供“只做 GPU encode+search，不做 load_docs/filter”的批量接口，供 micro-batching worker 使用

#### C. 当时的默认策略（commit 当时）

- micro-batching 参数在当时是“写死的”：`max_batch_size=256`、`max_wait_ms=5ms`、`k=100`（打印日志里明确显示 `k=100`）
- **注意**：后续在 `f455a8c` 里把这些变成可配置参数（见下节）。

---

### 2.3 `f455a8c` — helper scripts + 可恢复 eval + 更细粒度统计

这个 commit 覆盖 9 个文件，是“今天新增脚本 + 把日志打通 + runner 更可用”的集合。

#### A. 新增：HLE 工具调用模型分布统计脚本

**新增文件**：`evaluation/analyze_tool_model_usage.py`

- 解析 `hle.log` 里结构化的 `type=tool_call tool=... model=...`
- 输出每个 tool 的：
  - 总调用次数
  - GPT-5 / GPT-5-mini 占比（粗略以 `model` 字符串包含 `gpt-5` 判断）
  - 其余模型（Together/Nebius/OSS）占比
  - 按 model 的详细 breakdown

#### B. 新增：指定 tool+model 的耗时分布脚本

**新增文件**：`evaluation/analyze_specific_tool_duration.py`

- 从 `hle.log` 提取某个 `tool=<tool_name> model=<model_name>` 的 `duration_ms`
- 打印：n/min/max/mean/std + 一组分位数（p1/p10/.../p99）
- 输出直方图到：`/tmp/enhance_reasoning_qwen_coder_duration_hist.png`
- **默认分析目标是写死的**：  
  - `tool_name = "enhance_reasoning"`  
  - `model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"`  
  - 如需换别的组合，直接改脚本里这两行即可（快速一次性分析用）。

#### C. `analyze_hle_timing.py`：支持同一 log 文件多次 run 的分析

**修改文件**：`evaluation/analyze_hle_timing.py`

- 新增 `--all-runs`
  - **默认行为**：如果 log 内有 `type=run_start` marker，只分析最后一次 run（适配“append 到同一 hle.log”的情况）
  - `--all-runs`：分析整个文件内的所有 run 段

#### D. `eval_hle_local.py`：更稳的并发执行 + 更可用的结构化字段

**修改文件**：`evaluation/eval_hle_local.py`

关键点（都直接影响“可恢复”和“可分析”）：

- **任务级可恢复（resume）**
  - 对每个 example：如果输出文件已存在（`<output-dir>/<id>.json`），直接 skip，并打印：
    - `[HLE_TASK_COMPLETE] id=... status=skipped`
- **单 task 异常不会把全 run 打崩**
  - `asyncio.run(run_all(..., return_exceptions=True))`
- **结构化 tool_call 更细**
  - tool_call 现在记录 `tool=<tool_name> model=<model> duration_ms=<...>`
  - `answer` tool：区分
    - `duration_ms`（expert 部分）
    - `duration_total_ms`（包含其他开销）
    - `judge_ms`（如果有 LLM-as-judge）
  - `search` tool：额外记录
    - `search_backend=local_only|tavily_fallback`
    - `search_local_hits` / `search_tavily_hits`
    - 判定逻辑基于 retrieval server 返回的 hit dict：`source=="tavily"` 或 `score<0` 视为 Tavily hit
- **run_start marker**
  - 运行开始时写入 `type=run_start`，给 `analyze_hle_timing.py` 切分 run 用

#### E. `retrieval_hle.py`：micro-batching 参数化 + 增加 source 字段

**修改文件**：`evaluation/retrieval_hle.py`

- **对 micro-batching 参数做成 CLI + env 可配置**：
  - `--faiss_k`（默认：`$RETRIEVAL_FAISS_K`，未设置则 1000）
  - `--max_batch_size`（默认：`$RETRIEVAL_MAX_BATCH_SIZE`，未设置则 256）
  - `--max_wait_ms`（默认：`$RETRIEVAL_MAX_WAIT_MS`，未设置则 5）
- **返回结果新增 `source` 字段**（便于 eval 侧识别 backend）
  - 本地检索 hit：`{"document": ..., "score": <pos>, "source": "local"}`
  - Tavily fallback hit：`{"document": {"content": ...}, "score": <neg>, "source": "tavily"}`

> 备注：如果你想复现 `4c3b777` 当时“k=100”的更快行为，请显式传：`--faiss_k 100` 或设置 `RETRIEVAL_FAISS_K=100`。

#### F. `retriever_env_setup.md`：补充环境信息 + 安装 hf_transfer

**修改文件**：`evaluation/retriever_env_setup.md`

- 记录了验证通过的 container image
- 依赖安装里新增 `hf_transfer`（避免 HF_HUB_ENABLE_HF_TRANSFER 路径缺包导致崩）

#### G. `run_hle_profile.sh`：一键后台跑 + 自动清理端口 + 进度监控

**修改文件**：`evaluation/run_hle_profile.sh`

- 自动 `source setup_envs.sh`（保证 CKPT_DIR/TAVILY_KEY 等环境变量）
- 启动前 kill 端口监听（默认 1406 orchestrator / 1401 retrieval）
- 用 `--no-reuse-running` 强制每次都 fresh start（避免复用到脏状态）
- `monitor.out` 每 60s 写一行 done/total/pid
- 结束后自动运行 `analyze_hle_timing.py`

#### H. `tau2-bench/run_full_c48_profile.sh`：新增 retrieval 启动 & 强清理 & 归档 model_config

**修改文件**：`evaluation/tau2-bench/run_full_c48_profile.sh`

- 用 `kill_by_pattern` 在开跑前清掉：`run_local.py` / `tau2.cli` / `vllm serve` / `retrieval_hle.py`
- 启动 retrieval（GPU1、1401）并 health check
- 把 `--model-config-path` 指向当前 `LOG_DIR/model_config_local.json`
- run 完后主动停掉 retrieval（并在 EXIT/INT/TERM trap 里做 best-effort cleanup）

#### I. `tau2-bench/run_local.py`：日志、ETA、子进程读写改进

**修改文件**：`evaluation/tau2-bench/run_local.py`

- vLLM server 的 stdout/err 写到 `--log-dir`（`VLLMServer.start(log_dir=...)`）
- `tau2.cli` 通过 `sys.executable -m tau2.cli` 启动，确保用当前 env 的 python
- `TAU2_LOG_FILE` 自动设置为 `log_dir/tau2_{domain}.log`（append，结构化日志可解析）
- `eval_{domain}.log` 也会 append（stdout/stderr 统一加前缀写进去，方便复盘）
- 非阻塞 selector 读取 stdout/stderr，避免 child “安静时 readline 卡住”
- 心跳：30s 无输出会打印 “still running”
- 通过 `[TAU2_TASK_COMPLETE]` marker 维护总体 done，并打印红字 `[OVERALL_ETA]`

---

## 3) 如何使用（可直接复制的命令）

下面所有命令默认在 `/workspace/ToolOrchestra` 下执行。

### 3.1 HLE：一键后台跑 + 自动分析（推荐）

```bash
cd /workspace/ToolOrchestra
source setup_envs.sh   # 你本地的（含 CKPT_DIR / INDEX_DIR / OPENAI_API_KEY / TAVILY_KEY 等）

# 可选：覆盖默认并发/轮数/输入数据
export CONCURRENCY=512
export MAX_ROUNDS=50
export EXAMPLE_PATH=/workspace/ToolOrchestra/evaluation/hle.jsonl
export RUN_NAME=hle_profile_$(date +%Y%m%d_%H%M%S)

bash evaluation/run_hle_profile.sh

# 运行中进度（每分钟一条）
tail -f evaluation/logs/${RUN_NAME}/monitor.out
```

产物（都在 `evaluation/logs/${RUN_NAME}` / `evaluation/outputs/${RUN_NAME}`）：
- `driver.out`：runner 总输出
- `eval_hle_local.log`：eval 过程 + `[HLE_TASK_COMPLETE]`
- `hle.log`：结构化 PROFILE/USER_JUDGE（用于离线分析）
- `analysis.out` + 一组 `*_linear.png`/`*_log.png` + `*_stats.json`

---

### 3.2 HLE：单独跑离线分析（hist/stats）

```bash
cd /workspace/ToolOrchestra

# 默认：如果 hle.log 里 append 了多次 run，只分析最后一次
python evaluation/analyze_hle_timing.py \
  --log-dir evaluation/logs/<RUN_NAME> \
  --log-glob "hle.log" \
  --out-prefix <RUN_NAME> \
  --bins 10

# 如果要把同一个 hle.log 里的所有 run 段都统计进去：
python evaluation/analyze_hle_timing.py \
  --log-dir evaluation/logs/<RUN_NAME> \
  --log-glob "hle.log" \
  --out-prefix <RUN_NAME> \
  --bins 10 \
  --all-runs
```

---

### 3.3 HLE：统计每个 tool 用了哪些模型（今天新增脚本）

```bash
cd /workspace/ToolOrchestra
python evaluation/analyze_tool_model_usage.py evaluation/logs/<RUN_NAME>/hle.log

# 也可以聚合多个 log 文件
python evaluation/analyze_tool_model_usage.py \
  evaluation/logs/<RUN1>/hle.log \
  evaluation/logs/<RUN2>/hle.log
```

---

### 3.4 HLE：分析某个 tool+model 的 duration 分布（今天新增脚本）

```bash
cd /workspace/ToolOrchestra
python evaluation/analyze_specific_tool_duration.py evaluation/logs/<RUN_NAME>/hle.log

# 输出图默认写到：
# /tmp/enhance_reasoning_qwen_coder_duration_hist.png
```

如需换分析目标，编辑 `evaluation/analyze_specific_tool_duration.py` 里：
- `tool_name = ...`
- `model_name = ...`

---

### 3.5 仅启动 retrieval server（用于 smoke / 单独压测）

```bash
cd /workspace/ToolOrchestra
source /root/miniconda3/etc/profile.d/conda.sh
conda activate retriever

export INDEX_DIR=/workspace/dataset/multi-train/index
export HF_HOME=/workspace/cache/huggingface

# 推荐：复现 “更快的候选 k=100”
python evaluation/retrieval_hle.py \
  --port 1401 \
  --new_cache_dir evaluation/cache/hle \
  --example_id_file evaluation/examples.json \
  --tavily_key "${TAVILY_KEY:-}" \
  --faiss_k 100 \
  --max_batch_size 256 \
  --max_wait_ms 5

# 健康检查
curl -sSf http://127.0.0.1:1401/health
```

---

### 3.6 tau2-bench：一键 full c48 PROFILE（今天增强了 runner）

```bash
cd /workspace/ToolOrchestra
source setup_envs.sh

cd evaluation/tau2-bench
bash run_full_c48_profile.sh

# 查看最近一次 run 的 log dir
cat /workspace/ToolOrchestra/evaluation/tau2-bench/logs/last_full_c48_profile_dir.txt
```

这个脚本现在会：
- 自动清理旧进程（vLLM/tau2/retrieval）
- 拉起 retrieval（GPU1/1401）
- 运行 `run_local.py`（PROFILE）并输出到 timestamped 目录
- 自动运行 `analyze_timing_from_tau2_log.py` 生成统计/图

---

## 4) 关机前 checklist（建议）

- 确认你需要的日志/输出已保存（尤其是 `evaluation/logs/...`、`evaluation/outputs/...` 和 `evaluation/tau2-bench/logs/...`）。
- 如果你有把日志移动到 `/workspace/exp-log/...` 的习惯，建议把本次 HLE/tau2 的 run 目录也一起挪走备份。


