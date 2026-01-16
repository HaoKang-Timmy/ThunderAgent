# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ThunderReact is a FastAPI-based request router for vLLM that implements capacity-based scheduling for multi-turn agentic LLM workflows. It sits between clients and one or more vLLM backends, tracking program (task) state across turns and managing KV cache capacity.

## Running the Router

```bash
# Basic usage with single backend
python -m ThunderReact --backends http://localhost:8000

# Full options
python -m ThunderReact \
  --host 0.0.0.0 \
  --port 8300 \
  --backends "http://localhost:8000,http://localhost:8001" \
  --router tr \          # "tr" (capacity scheduling) or "default" (pure proxy)
  --profile \            # Enable timing profiling
  --profile-dir /tmp/thunderreact_profiles \
  --metrics \            # Enable vLLM metrics monitoring
  --metrics-interval 5.0
```

## Dependencies

- fastapi, uvicorn
- httpx (async HTTP client)
- No formal requirements.txt - install manually: `pip install fastapi uvicorn httpx`

## Architecture

```
ThunderReact/
├── app.py              # FastAPI app, routes: /v1/chat/completions, /programs, /health, /metrics
├── config.py           # Global Config singleton (set via CLI args)
├── backend/
│   ├── state.py        # BackendState: KV cache capacity tracking, pause/resume cooldowns
│   └── vllm_metrics.py # VLLMMetrics, VLLMCacheConfig: Prometheus parsing
├── program/
│   └── state.py        # ProgramState, ProgramStatus: per-task lifecycle
├── profile/
│   └── state.py        # ProfileState, StepMetrics: timing per request
└── scheduler/
    └── router.py       # MultiBackendRouter: core scheduling logic
```

### Program Lifecycle (ProgramStatus)

```
PAUSED → REASONING → ACTING → REASONING → ... → STOPPED
         (GPU)      (tool)    (GPU)
```

- **REASONING**: On GPU, running inference
- **ACTING**: Off GPU, executing tool call (tracked via `acting_since` timestamp)
- **PAUSED**: Waiting for capacity (in `paused_pool`, can migrate backends on resume)
- **STOPPED**: Released/completed (removed from `programs` dict)

Key flags in `ProgramState`:
- `tool_finished`: Set True when client sends next request (indicates tool execution complete)
- `marked_for_pause`: REASONING programs marked to pause on next ACTING transition
- `acting_since`: Unix timestamp for time-decay weighting

### Capacity Scheduling (--router tr)

When capacity is exceeded:
1. Pause ACTING programs (smallest first) - immediate
2. Mark REASONING programs for pause (deferred until they become ACTING)
3. Resume paused programs when capacity frees (smallest first, prioritize `step_count > 1`)

Key constants in `backend/state.py`:
- `DECODE_BUFFER = 256` - tokens reserved per active program for decode
- `PAUSE_COOLDOWN = 5.0` - seconds between pause operations
- `RESUME_COOLDOWN = 5.0` - seconds between resume operations
- `TOOL_COEFFICIENT = 1.0` - weight for ACTING program tokens in capacity calculation

**Capacity formula:** `required = active_tokens + active_count * DECODE_BUFFER - max(0, count-1) * shared_tokens <= total_capacity`

**Time-decay weighting for ACTING programs:** `weight = 2^(-t)` where t is seconds since `acting_since`. This reduces ACTING programs' effective token contribution over time, allowing earlier resume of paused programs.

### Request Flow

1. Client sends `/v1/chat/completions` with `program_id` in payload or `extra_body`
2. Router assigns program to least-loaded backend (sticky routing)
3. Capacity check - may pause and wait (max 30 min timeout, then force resume)
4. Forward to vLLM, extract usage from response
5. Transition to ACTING, update token counts, run capacity enforcement
6. Client calls `/programs/release` when task completes (triggers background resume attempt)

### Resume Monitor

A background task (`_resume_monitor_loop`) runs every 5s attempting to resume paused programs across all backends. This complements the on-demand resume triggered by `/programs/release`.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Proxy to vLLM (extracts `program_id` from payload) |
| `/programs` | GET | List all programs with status and profile data |
| `/programs/release` | POST | Release a program `{"program_id": "..."}` |
| `/health` | GET | Router stats: mode, backends, program counts |
| `/metrics` | GET | Per-backend vLLM metrics aggregation |
| `/profiles` | GET | Timing metrics for all programs |
| `/profiles/{program_id}` | GET | Timing metrics for specific program |
| `/v1/models` | GET | Proxy to first backend |

## Key Implementation Details

- **Shared tokens**: System prompt tokens estimated from first request (`len(text) // 5`), stored in `BackendState.shared_token` (class variable, shared across all backends)
- **Global paused pool**: Lock-protected `paused_pool` dict enables backend migration on resume
- **Sticky routing**: Programs stay on assigned backend unless paused/resumed to different one
- **Fairness**: New programs (`step_count == 1`) must queue if paused programs are waiting
- **Profile output**: CSV written to `--profile-dir/step_profiles.csv`
- **Metrics history**: Last 12 samples kept per backend (`METRICS_HISTORY_SIZE`)
- **Non-blocking scheduling**: `scheduling_in_progress` flag prevents concurrent capacity enforcement on same backend
