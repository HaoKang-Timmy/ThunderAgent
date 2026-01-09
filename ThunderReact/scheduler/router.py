"""Multi-backend router - core scheduling logic."""
import asyncio
import hashlib
import json
import math
import logging
import time
from typing import Any, Dict, Optional, List, Callable, Awaitable

import httpx
from fastapi.responses import Response, StreamingResponse
from prometheus_client.parser import text_string_to_metric_families

from ..config import (
    THRASHING_DISTANCE,
    CONTROL_TIME_WINDOW_S,
    CONTROL_STEP,
    PAUSE_TRIGGER_USAGE,
    RESUME_TRIGGER_USAGE,
    TRANSFER_IMBALANCE_TRIGGER,
    KV_CACHE_TOKEN_BUDGET,
)
from ..backend import BackendState
from ..program import ProgramState

logger = logging.getLogger(__name__)


class MultiBackendRouter:
    """Multi-backend load balancing router."""

    def __init__(self, backend_urls: List[str]) -> None:
        self.backends: Dict[str, BackendState] = {
            url: BackendState(url=url) for url in backend_urls
        }
        self.program_affinity: Dict[str, str] = {}
        self._transfer_imbalance_start_time: float = 0.0

        self.client = httpx.AsyncClient(
            timeout=900.0,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        )
        self.metrics_client = httpx.AsyncClient(timeout=5.0)
        self.monitor_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the router monitoring task."""
        self.monitor_task = asyncio.create_task(self.monitor_loop())
        logger.info(f"Started router with {len(self.backends)} backends: {list(self.backends.keys())}")

    async def stop(self):
        """Stop the router."""
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        await self.client.aclose()
        await self.metrics_client.aclose()

    # -------------------------------------------------------------------------
    # Monitoring
    # -------------------------------------------------------------------------

    @staticmethod
    def parse_metric_value(line: str) -> Optional[float]:
        """Parse the value from a Prometheus metric line."""
        parts = line.split()
        if len(parts) < 2:
            return None
        try:
            value = float(parts[1])
        except ValueError:
            return None
        if not math.isfinite(value):
            return None
        return value

    def extract_metrics_fallback(self, text: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Fallback method for parsing metrics."""
        usage = None
        running = None
        waiting = None
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            if line.startswith("vllm:kv_cache_usage_perc"):
                if usage is None:
                    usage = self.parse_metric_value(line)
            elif line.startswith("vllm:num_requests_running"):
                if running is None:
                    running = self.parse_metric_value(line)
            elif line.startswith("vllm:num_requests_waiting"):
                if waiting is None:
                    waiting = self.parse_metric_value(line)
            if usage is not None and running is not None and waiting is not None:
                break
        return usage, running, waiting

    async def monitor_loop(self):
        """Background monitoring loop."""
        while True:
            tasks = [
                self.fetch_backend_usage(backend)
                for backend in self.backends.values()
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

            now = time.time()
            await self.schedule_global_transfers(now)

            for backend in self.backends.values():
                async with backend.lock:
                    self.update_backend_state(backend)

            await asyncio.sleep(0.5)

    async def fetch_backend_usage(self, backend: BackendState):
        """Fetch resource usage from a backend."""
        try:
            resp = await self.metrics_client.get(backend.metrics_url, timeout=3.0)
            resp.raise_for_status()

            running_req = None
            waiting_req = None
            usage_val = None

            try:
                families = list(text_string_to_metric_families(resp.text))
            except Exception as exc:
                logger.warning(
                    f"Failed to parse metrics from {backend.url}, falling back to line parsing: {exc!r}"
                )
                usage_val, running_req, waiting_req = self.extract_metrics_fallback(resp.text)
            else:
                for family in families:
                    if family.name == "vllm:kv_cache_usage_perc":
                        values = [
                            float(sample.value)
                            for sample in family.samples
                            if sample.name == family.name and math.isfinite(float(sample.value))
                        ]
                        if values:
                            usage_val = max(values)
                    elif family.name == "vllm:num_requests_running":
                        running_req = sum(
                            float(sample.value)
                            for sample in family.samples
                            if sample.name == family.name and math.isfinite(float(sample.value))
                        )
                    elif family.name == "vllm:num_requests_waiting":
                        waiting_req = sum(
                            float(sample.value)
                            for sample in family.samples
                            if sample.name == family.name and math.isfinite(float(sample.value))
                        )

            if usage_val is not None:
                backend.usage = usage_val / 100.0 if usage_val > 1.0 else usage_val
            if running_req is not None:
                backend.running_requests = running_req
            if waiting_req is not None:
                backend.waiting_requests = waiting_req
            backend.healthy = True
        except Exception as exc:
            logger.warning(f"Failed to fetch metrics from {backend.url}: {exc}")
            backend.healthy = False

    # -------------------------------------------------------------------------
    # Pause/Resume/Transfer Logic
    # -------------------------------------------------------------------------

    def update_backend_state(self, backend: BackendState):
        """Update backend state and perform pause/resume operations."""
        now = time.time()
        usage = backend.usage

        thrashing_margin = max(0.0, THRASHING_DISTANCE)
        low_watermark = max(0.0, PAUSE_TRIGGER_USAGE - thrashing_margin)

        if usage >= PAUSE_TRIGGER_USAGE:
            if backend.overload_start_time == 0.0:
                backend.overload_start_time = now
        else:
            backend.overload_start_time = 0.0

        overload_sustained = (
            backend.overload_start_time > 0.0
            and (now - backend.overload_start_time) >= CONTROL_TIME_WINDOW_S
        )
        if overload_sustained:
            tokens_to_pause = (
                max(0.0, usage - low_watermark)
                * KV_CACHE_TOKEN_BUDGET
                * CONTROL_STEP
            )
            self.pause_programs_on_backend(backend, tokens_to_pause)
            return

        if usage < RESUME_TRIGGER_USAGE:
            if backend.underload_start_time == 0.0:
                backend.underload_start_time = now
        else:
            backend.underload_start_time = 0.0

        underload_sustained = (
            backend.underload_start_time > 0.0
            and (now - backend.underload_start_time) >= CONTROL_TIME_WINDOW_S
        )
        if underload_sustained:
            high_watermark = min(1.0, RESUME_TRIGGER_USAGE + thrashing_margin)
            tokens_to_resume = (
                max(0.0, high_watermark - usage)
                * KV_CACHE_TOKEN_BUDGET
                * CONTROL_STEP
            )
            self.resume_programs_on_backend(backend, tokens_to_resume)

    def pause_programs_on_backend(
        self,
        backend: BackendState,
        tokens_to_pause: float,
    ):
        """Pause low-priority programs on this backend."""
        active = [
            (pid, state)
            for pid, state in backend.programs.items()
            if not state.paused
            and state.transfer_target is None
        ]
        if not active:
            return

        if tokens_to_pause <= 0:
            return
        active.sort(key=lambda item: item[1].est_tokens)

        paused_tokens = 0
        newly_paused = []

        for pid, state in active:
            if paused_tokens >= tokens_to_pause:
                break

            state.paused = True
            state.resume_event.clear()

            paused_tokens += state.est_tokens
            newly_paused.append(pid)

        if newly_paused:
            logger.info(
                f"[{backend.url}] Paused {len(newly_paused)} programs "
                f"(usage={backend.usage:.2%}, tokens={paused_tokens}): {newly_paused}"
            )

    def resume_programs_on_backend(self, backend: BackendState, tokens_to_resume: float):
        """Resume paused programs on this backend."""
        if tokens_to_resume <= 0:
            return

        paused = [
            (pid, state)
            for pid, state in backend.programs.items()
            if state.paused and state.transfer_target is None
        ]
        if not paused:
            return

        paused.sort(key=lambda item: -item[1].est_tokens)

        resumed_tokens = 0
        resumed_pids = []

        for pid, state in paused:
            if resumed_tokens >= tokens_to_resume:
                break
            state.paused = False
            state.resume_event.set()
            resumed_tokens += state.est_tokens
            resumed_pids.append(pid)

        if resumed_pids:
            logger.info(
                f"[{backend.url}] Resumed {len(resumed_pids)} programs "
                f"(usage={backend.usage:.2%}, budget={tokens_to_resume:.0f}): {resumed_pids}"
            )

    async def schedule_global_transfers(self, now: float) -> None:
        """Schedule cross-backend program transfers."""
        healthy = [b for b in self.backends.values() if b.healthy]
        if len(healthy) < 2:
            self._transfer_imbalance_start_time = 0.0
            return

        source = max(healthy, key=lambda b: b.usage)
        target = min(healthy, key=lambda b: b.usage)
        imbalance = source.usage - target.usage
        if imbalance < TRANSFER_IMBALANCE_TRIGGER:
            self._transfer_imbalance_start_time = 0.0
            return

        if self._transfer_imbalance_start_time == 0.0:
            self._transfer_imbalance_start_time = now
            return
        if (now - self._transfer_imbalance_start_time) < CONTROL_TIME_WINDOW_S:
            return

        transfer_budget = imbalance * KV_CACHE_TOKEN_BUDGET * CONTROL_STEP
        target_free_tokens = max(0.0, 1.0 - target.usage) * KV_CACHE_TOKEN_BUDGET
        transfer_budget = min(transfer_budget, target_free_tokens)
        if transfer_budget <= 0:
            return

        transfer_pids: list[str] = []
        transfer_tokens = 0.0
        async with source.lock:
            active = [
                (pid, state)
                for pid, state in source.programs.items()
                if state.transfer_target is None
            ]
            if not active:
                return

            active.sort(key=lambda item: item[1].est_tokens)
            for pid, state in active:
                if transfer_tokens >= transfer_budget:
                    break
                state.transfer_target = target.url
                if state.paused and state.waiting_on_resume:
                    state.paused = False
                    state.resume_event.set()
                transfer_tokens += state.est_tokens
                transfer_pids.append(pid)

        if transfer_pids:
            self._transfer_imbalance_start_time = 0.0
            logger.info(
                f"[transfer] Scheduled {len(transfer_pids)} programs "
                f"from {source.url} (usage={source.usage:.2%}) "
                f"to {target.url} (usage={target.usage:.2%}) "
                f"(imbalance={imbalance:.2%}, budget={transfer_budget:.0f}, tokens={transfer_tokens:.0f}): "
                f"{transfer_pids}"
            )

    async def apply_pending_transfer(
        self,
        program_id: str,
        backend: BackendState,
    ) -> BackendState:
        """Apply pending program transfer."""
        async with backend.lock:
            state = backend.programs.get(program_id)
            if state is None or state.transfer_target is None:
                return backend
            target_url = state.transfer_target

        target_backend = self.backends.get(target_url)
        if target_backend is None or not target_backend.healthy:
            async with backend.lock:
                state = backend.programs.get(program_id)
                if state and state.transfer_target == target_url:
                    state.transfer_target = None
            return backend

        if target_backend.url == backend.url:
            async with backend.lock:
                state = backend.programs.get(program_id)
                if state:
                    state.transfer_target = None
            return backend

        first, second = sorted([backend, target_backend], key=lambda b: b.url)
        async with first.lock:
            async with second.lock:
                state = backend.programs.get(program_id)
                if (
                    state is None
                    or state.transfer_target != target_url
                    or state.inflight
                ):
                    return backend
                del backend.programs[program_id]
                target_backend.programs[program_id] = state
                self.program_affinity[program_id] = target_url
                state.transfer_target = None
                state.paused = False
                state.resume_event.set()
                return target_backend

    # -------------------------------------------------------------------------
    # Backend Selection (Sticky Routing + Load Balancing)
    # -------------------------------------------------------------------------

    def get_backend_for_program(self, program_id: str) -> BackendState:
        """Get backend for a program using sticky routing or least-loaded assignment."""
        # Check existing affinity
        if program_id in self.program_affinity:
            backend_url = self.program_affinity[program_id]
            backend = self.backends.get(backend_url)
            if backend and backend.healthy:
                return backend
            # Backend is unhealthy, need to reassign
            logger.warning(
                f"Backend {backend_url} unhealthy for {program_id}, reassigning"
            )
            del self.program_affinity[program_id]

        # New program: assign to least loaded backend
        if program_id == "default":
            logger.warning("Missing job_id in extra_body; routing will not be balanced across backends.")
        backend = self.pick_least_loaded_backend(program_id)
        self.program_affinity[program_id] = backend.url
        logger.debug(f"Assigned {program_id} to {backend.url}")
        return backend

    def pick_least_loaded_backend(self, program_id: str) -> BackendState:
        """Select a healthy backend, prefer those with no paused programs."""
        healthy = [b for b in self.backends.values() if b.healthy]
        if not healthy:
            logger.warning("No healthy backends, using first available")
            return list(self.backends.values())[0]

        candidates = [b for b in healthy if b.paused_count == 0] or healthy

        def score(b: BackendState) -> float:
            return b.running_requests + b.waiting_requests * 4.0

        scored = [(b, score(b)) for b in candidates]
        min_score = min(val for _, val in scored)
        best = [b for b, val in scored if val == min_score]
        if len(best) == 1:
            return best[0]
        digest = hashlib.sha256(program_id.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:8], "big") % len(best)
        return best[idx]

    # -------------------------------------------------------------------------
    # Request Proxying
    # -------------------------------------------------------------------------

    @staticmethod
    def extract_total_tokens(payload: Any) -> Optional[int]:
        """Extract total_tokens from the response."""
        if not isinstance(payload, dict):
            return None
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        if "total_tokens" in usage:
            val = usage.get("total_tokens")
            if isinstance(val, (int, float)) and math.isfinite(val):
                return int(val)
        return None

    @staticmethod
    def filtered_headers(headers: httpx.Headers) -> Dict[str, str]:
        """Filter out hop-by-hop headers."""
        hop_by_hop = {"content-length", "transfer-encoding", "connection"}
        return {k: v for k, v in headers.items() if k.lower() not in hop_by_hop}

    async def proxy_request(
        self,
        backend: BackendState,
        payload: Dict[str, Any],
        *,
        on_total_tokens: Callable[[int], Awaitable[None]] | None = None,
    ) -> Response:
        """Proxy request to the backend."""
        url = backend.completions_url

        if payload.get("stream"):
            stream_options = payload.get("stream_options")
            if stream_options is None:
                payload["stream_options"] = {"include_usage": True}
            elif isinstance(stream_options, dict):
                stream_options.setdefault("include_usage", True)

            resp_cm = self.client.stream("POST", url, json=payload)
            resp = await resp_cm.__aenter__()
            headers = self.filtered_headers(resp.headers)
            status = resp.status_code
            media_type = resp.headers.get("content-type")

            async def iterator():
                buffer = b""
                total_tokens: Optional[int] = None
                try:
                    async for chunk in resp.aiter_raw():
                        buffer += chunk
                        while b"\n\n" in buffer:
                            event, buffer = buffer.split(b"\n\n", 1)
                            for line in event.split(b"\n"):
                                if not line.startswith(b"data:"):
                                    continue
                                data = line[5:].strip()
                                if not data or data == b"[DONE]":
                                    continue
                                if total_tokens is not None:
                                    continue
                                try:
                                    payload_obj = json.loads(data)
                                except Exception:
                                    continue
                                extracted = self.extract_total_tokens(payload_obj)
                                if extracted is not None:
                                    total_tokens = extracted
                        yield chunk
                finally:
                    await resp_cm.__aexit__(None, None, None)
                    if total_tokens is not None and on_total_tokens is not None:
                        await on_total_tokens(total_tokens)

            return StreamingResponse(
                iterator(),
                status_code=status,
                headers=headers,
                media_type=media_type,
            )

        resp = await self.client.post(url, json=payload)
        total_tokens: Optional[int] = None
        try:
            payload_obj = resp.json()
        except Exception:
            payload_obj = None
        extracted = self.extract_total_tokens(payload_obj)
        if extracted is not None:
            total_tokens = extracted
        if total_tokens is not None and on_total_tokens is not None:
            await on_total_tokens(total_tokens)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=self.filtered_headers(resp.headers),
            media_type=resp.headers.get("content-type"),
        )
