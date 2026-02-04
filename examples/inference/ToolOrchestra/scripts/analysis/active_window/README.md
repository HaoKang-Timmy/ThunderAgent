# Active-window throughput analysis (KV cache > threshold)

This folder contains the **saved analysis scripts** used in the 2026-01-04 RunPod 5090 experiments comparing **ThunderReact** vs **Continuum** on tau2-bench (global scheduling, domains retail/telecom/airline).

## Inputs (per experiment run directory)

Each run directory must contain:

- `kv_cache_timeseries.csv`
  - Collected by `scripts/rollout/collect_kv_cache_timeseries.sh`
  - Contains `vllm:kv_cache_usage_perc` samples (fraction in [0,1])
- `driver.log`
  - Produced by `evaluation/tau2-bench/run_oss.py` (`--log-level PROFILE` recommended)

Continuum runs additionally contain:

- `scheduler_timestamps` (written by vLLM Continuum scheduler when vLLM exits gracefully)

## Metric definitions (what these scripts compute)

- **KV cache usage**: read from `kv_cache_timeseries.csv` rows where `metric` contains `kv_cache_usage_perc`.
  - Values are **fractions**; scripts convert to **percent** for reporting.
- **Tasks done**: counted via `driver.log` lines matching `FINISHED SIMULATION:` (one per completed tau2 task).
  - We use this rather than “tasks started”.
- **Steps done**: counted via `driver.log` PROFILE lines containing `type=step_complete`.

## Active window definition

Default “active window” used for throughput tables:

> **window_start** = first timestamp where KV cache usage **> threshold**  
> **window_end** = last timestamp where KV cache usage **> threshold**

This is intentionally robust to “tail domination” when a run gets stuck and KV drops near 0 for a long time.

Threshold default: **70%**.

## Scripts

- `compare_tr_ct_active_window.py`
  - Produces two markdown tables:
    - tasks/min (tasks done per minute)
    - steps/s (steps done per second)
- `window_outside_breakdown.py`
  - For each run: counts tasks/steps **before**, **inside**, and **after** the active window.
- `compare_bottleneck_active_window.py`
  - Active-window bottleneck summary:
    - step rates by role edge (e.g., `agent->env/s`)
    - `llm_call` long-tail counts and top slow calls
    - `user_sim` / `expert_call` latency distributions
- `tr128_stuck_report.py`
  - Deep dive specifically for the TR128 stuck incident:
    - How many tasks didn’t finish
    - “last good task” time
    - KV cache around that time
    - KV cache min/max distribution during the stuck period
- `check_kv_above_threshold_after.py`
  - Given a cutoff timestamp, confirms whether KV ever exceeded a threshold after that point.

## Example usage

```bash
# Compare 6 runs (TR32/64/128 and CT32/64/128)
python3 scripts/analysis/active_window/compare_tr_ct_active_window.py \
  --threshold 70 \
  --tr32 /path/to/OUTPUT_BASE/thunderreact_c32 \
  --tr64 /path/to/OUTPUT_BASE/thunderreact_c64 \
  --tr128 /path/to/OUTPUT_BASE/thunderreact_c128 \
  --ct32 /path/to/outputs/continuum_c32_YYYYMMDD_HHMMSS \
  --ct64 /path/to/outputs/continuum_c64_YYYYMMDD_HHMMSS \
  --ct128 /path/to/outputs/continuum_c128_YYYYMMDD_HHMMSS
```

