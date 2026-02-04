"""Router with program state tracking - supports multiple backends."""
import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Awaitable, Tuple

import httpx
from fastapi.responses import Response

from ..backend import BackendState
from ..program import Program, ProgramStatus, ProgramState
from ..profile.state import ProfileState
from ..config import get_config
from .vllm_request_processor import (
    forward_streaming_request,
    forward_non_streaming_request,
    forward_get_request,
)

logger = logging.getLogger(__name__)


@dataclass
class PausedInfo:
    """Metadata for a paused program stored in the global paused pool."""
    program_id: str
    total_tokens: int
    paused_at: float
    origin_backend: Optional[str]  # None for new programs that haven't been assigned yet
    step_count: int
    paused_from_status: Optional[str] = None  # Status before pause: "reasoning", "acting", or None for new programs


class MultiBackendRouter:
    """Router with program state tracking, supports multiple backends."""

    def __init__(
        self, 
        backend_urls: str | List[str], 
        *, 
        profile_enabled: bool = False,
        scheduling_enabled: bool = True,
    ) -> None:
        # Support single URL string or list of URLs
        if isinstance(backend_urls, str):
            backend_urls = [url.strip() for url in backend_urls.split(",") if url.strip()]
        
        # All backends
        self.backends: Dict[str, BackendState] = {
            url: BackendState(url=url) for url in backend_urls
        }
        
        # All programs (single source of truth)
        # Key: program_id, Value: Program (which includes backend_url)
        self.programs: Dict[str, Program] = {}

        # Global paused pool shared across backends (program_id -> PausedInfo).
        self.global_waiting_queue: Dict[str, PausedInfo] = {}

        # Lock for atomic claim (select + pop) from global_waiting_queue.
        self.pause_resume_lock = asyncio.Lock()
        
        # Profile configuration
        self.profile_enabled = profile_enabled
        
        # Scheduling mode: True = "tr" (capacity scheduling), False = "default" (pure proxy)
        self.scheduling_enabled = scheduling_enabled

        self.client = httpx.AsyncClient(
            timeout=900.0,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        )
        
        # Scheduler task for periodic capacity check
        #TODO interval could be defined by others
        self._scheduler_task: Optional[asyncio.Task] = None
        self._scheduler_stop = False
        self._scheduler_interval = 5.0  # seconds

    async def start(self):
        """Start the router."""
        logger.info(f"Started router with {len(self.backends)} backend(s): {list(self.backends.keys())}")
        
        # Always fetch cache config (needed for active_program_tokens_ratio)
        for backend in self.backends.values():
            await backend.fetch_cache_config()
        
        # Start metrics monitoring on each backend if enabled
        config = get_config()
        if config.metrics_enabled:
            for backend in self.backends.values():
                await backend.start_monitoring(config.metrics_interval)
        
        # Start the periodic scheduler if scheduling is enabled
        if self.scheduling_enabled:
            self._scheduler_stop = False
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            logger.info(f"Started scheduler loop (interval={self._scheduler_interval}s)")

    async def stop(self):
        """Stop the router."""
        # Stop the scheduler
        if self._scheduler_task:
            self._scheduler_stop = True
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None
            logger.info("Stopped scheduler loop")
        
        # Stop metrics monitoring on each backend
        for backend in self.backends.values():
            await backend.stop_monitoring()
        
        await self.client.aclose()
        logger.info("Router stopped")

    # -------------------------------------------------------------------------
    # Backend Selection
    # -------------------------------------------------------------------------

    def get_backend(self, url: str) -> Optional[BackendState]:
        """Get a backend by URL."""
        return self.backends.get(url)

    def get_default_backend(self) -> BackendState:
        """Get the first backend (for simple single-backend usage)."""
        return next(iter(self.backends.values()))

    def select_backend_for_new_program_default(self) -> BackendState:
        """Select the least loaded backend for a new program."""
        # Count programs per backend
        #### TODO change backend assign logistics
        backend_load: Dict[str, int] = {url: 0 for url in self.backends}
        for state in self.programs.values():
            if state.backend_url in backend_load:
                backend_load[state.backend_url] += 1
        
        # Find the backend with least programs (only consider healthy ones)
        min_load = float('inf')
        best_backend = None
        for url, load in backend_load.items():
            backend = self.backends[url]
            if backend.healthy and load < min_load:
                min_load = load
                best_backend = backend
        
        # Fallback to first backend if all unhealthy
        return best_backend or self.get_default_backend()

    # -------------------------------------------------------------------------
    # Program State Management
    # -------------------------------------------------------------------------

    @staticmethod
    def _estimate_system_prompt_tokens(payload: Dict[str, Any]) -> int:
        """Estimate system prompt tokens from the first request payload."""
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return 0
        
        parts: List[str] = []
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "system":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    text = item.get("text") or item.get("input_text")
                    if isinstance(text, str):
                        parts.append(text)
        
        if not parts:
            return 0
        
        text = "\n".join(parts)
        return max(0, len(text) // 5)

    def get_or_create_program(self, program_id: str, payload: Dict[str, Any] = None) -> Program:
        """Get existing program or create new one.
        
        Only creates the program and estimates token count.
        Backend assignment is deferred to update_program_before_request.
        
        Args:
            program_id: Unique identifier for the program
            payload: Request payload, used to estimate token count for new programs
        """
        if program_id not in self.programs:
            profile = ProfileState(program_id=program_id) if self.profile_enabled else None
            state = Program(
                backend_url=None,
                status=ProgramStatus.REASONING,  # Request arrived, program is reasoning
                state=ProgramState.ACTIVE,
                profile=profile,
            )
            
            # Estimate token count from payload
            if payload:
                state.context_len = len(json.dumps(payload, ensure_ascii=False))
                state.total_tokens = state.context_len // 5
            
            self.programs[program_id] = state
            logger.debug(f"Created program {program_id} (estimated tokens={state.total_tokens})")
        
        return self.programs[program_id]
    
    def _select_backend_for_new_program(self, estimated_tokens: int = 0) -> Optional[str]:
        """Select a backend for a new program (TR mode only).
        
        This function is only called when scheduling_enabled=True.
        
        Args:
            estimated_tokens: Estimated token count for the new program
        
        Returns:
            backend_url if can assign directly (queue empty + has capacity)
            None if should wait in queue
        """
        from ..backend.state import BUFFER_PER_PROGRAM
        
        # Only assign if queue is empty and backend has capacity
        if len(self.global_waiting_queue) > 0:
            return None  # Queue not empty, must wait for fairness
        
        # Find backend with least active tokens that has capacity for this program
        best_backend = None
        min_tokens = float('inf')
        required_capacity = estimated_tokens + BUFFER_PER_PROGRAM
        
        for backend in self.backends.values():
            if not backend.healthy:
                continue
            if backend.remaining_capacity() < required_capacity:
                continue  # Not enough capacity for this program
            if backend.active_program_tokens < min_tokens:
                min_tokens = backend.active_program_tokens
                best_backend = backend
        
        return best_backend.url if best_backend else None

    def get_backend_for_program(self, program_id: str) -> BackendState:
        """Get the backend assigned to a program."""
        state = self.programs.get(program_id)
        if state and state.backend_url in self.backends:
            return self.backends[state.backend_url]
        return self.get_default_backend()

    async def update_program_before_request(self, program_id: str, state: Program, payload: Dict[str, Any]) -> bool:
        """Update program state before sending request to vLLM.
        
        If scheduling_enabled=False (default mode): pure proxy, no capacity checks.
        If scheduling_enabled=True (tr mode): wait for scheduler to resume if PAUSED.
        
        Returns: True if can proceed
        """
        state.step_count += 1
        is_new_program = state.step_count == 1
        
        # Update context_len (for non-new programs, context grows with each request)
        if not is_new_program:
            state.context_len = len(json.dumps(payload, ensure_ascii=False))
        # Note: For new programs, total_tokens was already estimated in get_or_create_program
        
        # ---------------------------------------------------------------------
        # Default mode: pure proxy, no scheduling
        # ---------------------------------------------------------------------
        if not self.scheduling_enabled:
            backend = self.backends.get(state.backend_url)
            if not backend:
                # Assign to least loaded backend
                backend = self.select_backend_for_new_program_default()
                state.backend_url = backend.url
            
            if state.status == ProgramStatus.ACTING:
                backend.shift_tokens_to_reasoning(state.total_tokens)
            if is_new_program:
                backend.add_total_program(state.total_tokens)
                backend.add_active_program(state.total_tokens, is_acting=False)
            state.status = ProgramStatus.REASONING
            return True
        
        # ---------------------------------------------------------------------
        # TR mode: scheduler-based capacity management
        # ---------------------------------------------------------------------
        # Step 1: Handle PAUSED programs (wait for resume)
        # Note: waiting_event is created in _pause_program and cleared in _resume_program.
        # If waiting_event is not None, the program is still PAUSED and we must wait.
        if state.waiting_event is not None:
            await self._wait_for_resume(program_id, state)
            # After _wait_for_resume returns, _resume_program has been called which:
            # - Added tokens to backend (as ACTING or REASONING based on status at pause time)
            # - Set state.state = ACTIVE
            # - Cleared waiting_event = None
            # Fall through to Step 3 to handle token shift if needed
        
        # Step 2: Handle new programs (assign backend or queue)
        if is_new_program and state.backend_url is None:
            backend_url = self._select_backend_for_new_program(state.total_tokens)
            if backend_url:
                # Direct assignment: add tokens immediately
                state.backend_url = backend_url
                backend = self.backends[backend_url]
                backend.add_total_program(state.total_tokens)
                backend.add_active_program(state.total_tokens, is_acting=False)
                logger.debug(f"Assigned new program {program_id} to {backend_url}")
                state.status = ProgramStatus.REASONING
                return True
            else:
                # Queue and wait
                state.waiting_event = asyncio.Event()
                state.state = ProgramState.PAUSED
                self._add_to_global_waiting_queue_sync(program_id, state, backend=None)
                logger.debug(f"Queued new program {program_id} (tokens={state.total_tokens})")
                await self._wait_for_resume(program_id, state)
                # After resume: tokens already added by _resume_program
                state.status = ProgramStatus.REASONING
                return True
        
        # Step 3: Normal case - existing ACTIVE program with backend
        backend = self.backends.get(state.backend_url)
        if not backend:
            logger.error(f"Program {program_id} has no valid backend")
            return False
        
        # Shift from ACTING to REASONING
        if state.status == ProgramStatus.ACTING:
            backend.shift_tokens_to_reasoning(state.total_tokens)
        
        state.status = ProgramStatus.REASONING
        return True

    def update_program_after_request(self, program_id: str, state: Program, total_tokens: int) -> None:
        """Update program state after receiving response from vLLM.
        
        Transitions to ACTING (off GPU, executing tool).
        Updates token counts. If marked for pause, pause immediately.
        """
        # Transition to ACTING
        state.status = ProgramStatus.ACTING
        
        backend = self.backends.get(state.backend_url)
        if not backend:
            state.total_tokens = total_tokens
            return
        
        # Update tokens
        old_tokens = state.total_tokens
        state.total_tokens = total_tokens
        backend.shift_tokens_to_acting(old_tokens, total_tokens)
        
        # If marked for pause, pause now (while in ACTING state)
        if state.marked_for_pause:
            self._clear_mark_and_pause(program_id, state)

    async def release_program(self, program_id: str) -> bool:
        """Stop a program and release its resources.
        
        Removes tokens from tracking. Resume of waiting programs is handled by the scheduler.
        """
        if program_id not in self.programs:
            return False
        
        state = self.programs[program_id]
        backend = self.backends.get(state.backend_url) if state.backend_url else None
        
        # Clean up based on current lifecycle state
        if state.state == ProgramState.PAUSED:
            # Remove from waiting queue
            await self._remove_from_global_waiting_queue(program_id)
            if state.waiting_event:
                state.waiting_event.set()  # Unblock any waiting coroutine
        elif backend and state.state == ProgramState.ACTIVE:
            backend.remove_active_program(
                state.total_tokens,
                is_acting=(state.status == ProgramStatus.ACTING),
            )
            backend.remove_total_program(state.total_tokens)
        
        # Clear mark if was marked
        if backend and state.marked_for_pause:
            backend.future_paused_tokens -= state.total_tokens
            if backend.future_paused_tokens < 0:
                backend.future_paused_tokens = 0
            state.marked_for_pause = False
        
        state.state = ProgramState.TERMINATED
        
        # Remove from programs dict
        del self.programs[program_id]
        
        logger.info(f"Released and removed program: {program_id}")
        return True

    def get_programs_on_backend(self, backend_url: str) -> Dict[str, Program]:
        """Get all programs assigned to a specific backend."""
        return {
            pid: state for pid, state in self.programs.items()
            if state.backend_url == backend_url
        }

    # -------------------------------------------------------------------------
    # Global Paused Pool
    # -------------------------------------------------------------------------

    def _add_to_global_waiting_queue_sync(
        self, program_id: str, state: Program, backend: Optional[BackendState] = None,
        paused_from_status: Optional[str] = None
    ) -> None:
        """Add a program to the global waiting queue (synchronous version)."""
        paused_info = PausedInfo(
            program_id=program_id,
            total_tokens=state.total_tokens,
            paused_at=time.time(),
            origin_backend=backend.url if backend else None,
            step_count=state.step_count,
            paused_from_status=paused_from_status,
        )
        self.global_waiting_queue[program_id] = paused_info

    async def _add_to_global_waiting_queue(
        self, program_id: str, state: Program, backend: Optional[BackendState] = None,
        paused_from_status: Optional[str] = None
    ) -> None:
        """Add or update a program in the global paused pool (async version with lock)."""
        paused_info = PausedInfo(
            program_id=program_id,
            total_tokens=state.total_tokens,
            paused_at=time.time(),
            origin_backend=backend.url if backend else None,
            step_count=state.step_count,
            paused_from_status=paused_from_status,
        )
        async with self.pause_resume_lock:
            self.global_waiting_queue[program_id] = paused_info

    async def _remove_from_global_waiting_queue(self, program_id: str) -> Optional[PausedInfo]:
        """Remove a program from the global paused pool."""
        async with self.pause_resume_lock:
            return self.global_waiting_queue.pop(program_id, None)

    def _get_paused_programs_sorted(
        self, ascending: bool = True
    ) -> List[Tuple[str, Program, PausedInfo]]:
        """Get paused programs from the global pool, sorted by total_tokens.

        Callers that require consistency should hold pause_resume_lock.
        """
        programs: List[Tuple[str, Program, PausedInfo]] = []
        for pid, info in self.global_waiting_queue.items():
            state = self.programs.get(pid)
            if not state:
                continue
            programs.append((pid, state, info))
        return sorted(programs, key=lambda x: x[2].total_tokens, reverse=not ascending)

    def get_paused_counts_by_backend(self) -> Dict[str, int]:
        """Count paused programs per backend using paused pool metadata."""
        counts = {url: 0 for url in self.backends}
        for info in self.global_waiting_queue.values():
            if info.origin_backend in counts:
                counts[info.origin_backend] += 1
        return counts

    # -------------------------------------------------------------------------
    # Capacity-based Scheduling (pause/resume)
    # -------------------------------------------------------------------------

    def _get_acting_programs_sorted(self, backend_url: str, ascending: bool = True) -> List[Tuple[str, Program]]:
        """Get ACTING programs on a backend, sorted by total_tokens.
        
        Args:
            ascending: If True, smallest first. If False, largest first.
        """
        programs = [
            (pid, state) for pid, state in self.programs.items()
            if state.backend_url == backend_url and state.status == ProgramStatus.ACTING
        ]
        return sorted(programs, key=lambda x: x[1].total_tokens, reverse=not ascending)

    def _get_reasoning_programs_sorted(self, backend_url: str, ascending: bool = True) -> List[Tuple[str, Program]]:
        """Get REASONING programs on a backend, sorted by total_tokens.
        
        Args:
            ascending: If True, smallest first. If False, largest first.
        """
        programs = [
            (pid, state) for pid, state in self.programs.items()
            if state.backend_url == backend_url 
            and state.status == ProgramStatus.REASONING
            and not state.marked_for_pause  # Exclude already marked
        ]
        return sorted(programs, key=lambda x: x[1].total_tokens, reverse=not ascending)

    def _pause_program(self, program_id: str, state: Program) -> None:
        """Pause a program: remove from active and total, add to global paused pool.
        
        Only call this for ACTING programs. REASONING programs should be marked instead.
        """
        backend = self.backends.get(state.backend_url)
        if not backend:
            return
        
        # Record the status before pause (for resume)
        paused_from_status = state.status.value if state.status else None
        
        # Remove from active counts
        backend.remove_active_program(
            state.total_tokens,
            is_acting=(state.status == ProgramStatus.ACTING),
        )
        
        # Remove from total counts (will be re-added on resume to target backend)
        backend.remove_total_program(state.total_tokens)
        
        # Add to global paused pool with original status recorded
        self._add_to_global_waiting_queue_sync(program_id, state, backend, paused_from_status)
        state.state = ProgramState.PAUSED
        
        # Create waiting event if needed
        if state.waiting_event is None:
            state.waiting_event = asyncio.Event()
        else:
            state.waiting_event.clear()
        
        logger.info(f"Paused program {program_id} (tokens={state.total_tokens}, active={backend.active_program_tokens})")

    def _mark_program_for_pause(self, program_id: str, state: Program) -> None:
        """Mark a REASONING program for pause. It will be paused on next request.
        
        Adds the program's tokens to future_paused_tokens for capacity calculation.
        """
        backend = self.backends.get(state.backend_url)
        if not backend:
            return
        
        state.marked_for_pause = True
        backend.future_paused_tokens += state.total_tokens
        
        logger.info(f"Marked program {program_id} for pause (tokens={state.total_tokens}, future_paused={backend.future_paused_tokens})")

    def _clear_mark_and_pause(self, program_id: str, state: Program) -> None:
        """Clear the mark from a program and pause it.
        
        Called when a marked program's next request arrives.
        """
        backend = self.backends.get(state.backend_url)
        if not backend:
            return
        
        # Clear the mark and subtract from future_paused_tokens
        state.marked_for_pause = False
        backend.future_paused_tokens -= state.total_tokens
        if backend.future_paused_tokens < 0:
            backend.future_paused_tokens = 0
        
        # Now actually pause the program
        self._pause_program(program_id, state)

    def _resume_program(
        self,
        program_id: str,
        state: Program,
        target_backend: Optional[BackendState] = None,
        paused_info: Optional[PausedInfo] = None,
    ) -> None:
        """Resume a paused program after it has been claimed from the pool.
        
        Args:
            program_id: The program ID
            state: The program state
            target_backend: Backend to resume to (may differ from origin for migration)
            paused_info: Pause metadata containing paused_from_status
        """
        origin_backend = self.backends.get(state.backend_url) if state.backend_url else None
        backend = target_backend or origin_backend
        if not backend:
            return

        if state.state != ProgramState.PAUSED:
            return

        # Add to target backend's total counts (was removed during pause)
        backend.add_total_program(state.total_tokens)
        state.backend_url = backend.url

        # Resume to ACTIVE lifecycle with ACTING status:
        # - status stays as it was before pause (ACTING or REASONING for new programs)
        # - If a request is waiting (via _wait_for_resume), before_request will shift to REASONING
        # - If no request is waiting (periodic scheduler), program stays ACTING until next request
        backend.add_active_program(state.total_tokens, is_acting=(state.status == ProgramStatus.ACTING))
        state.state = ProgramState.ACTIVE
        
        # Signal waiting event and clear it (resume completes the pause-resume cycle)
        if state.waiting_event:
            state.waiting_event.set()
            state.waiting_event = None
        
        logger.info(f"Resumed program {program_id} (status={state.status.value}, tokens={state.total_tokens}, active={backend.active_program_tokens})")

    async def _enforce_capacity(self, backend: BackendState) -> Tuple[int, int, int]:
        """Enforce capacity constraint by pausing ACTING and marking REASONING programs.
        
        Overflow check: active_tokens - future_paused_tokens + buffer > total_tokens
        
        Algorithm:
        1. Pause ACTING programs (smallest first) until no overflow
        2. If still overflow (no more ACTING), mark REASONING programs (smallest first)
        3. After actual pauses, try to backfill with small paused programs to reduce slack.
        
        Note: Respects pause cooldown to prevent rapid oscillation.
              Marking REASONING programs is not subject to cooldown (deferred pause).
        
        Returns: (paused_count, marked_count, resumed_count)
        """
        paused = 0
        marked = 0
        resumed = 0
        
        # Step 1: Pause ACTING programs (smallest first)
        # Respect cooldown: only pause if enough time has passed
        while backend.capacity_overflow(include_future_release=True) > 0:
            # Check cooldown before pausing
            if paused > 0 and not backend.can_pause():
                logger.debug("Pause cooldown active, deferring further pauses")
                break
            
            acting_programs = self._get_acting_programs_sorted(backend.url, ascending=True)
            if not acting_programs:
                break  # No more ACTING programs, move to Step 2
            
            program_id, state = acting_programs[0]
            self._pause_program(program_id, state)
            backend.record_pause()
            paused += 1
        
        # Step 2: Mark REASONING programs if still over capacity
        # Marking is not subject to cooldown (actual pause will happen later)
        while backend.capacity_overflow(include_future_release=True) > 0:
            reasoning_programs = self._get_reasoning_programs_sorted(backend.url, ascending=True)
            if not reasoning_programs:
                break  # No more REASONING programs to mark
            
            program_id, state = reasoning_programs[0]
            self._mark_program_for_pause(program_id, state)
            marked += 1

        # Step 3：After actual pauses, try to backfill with small paused programs to reduce slack.
        """if paused > 0:
            resumed = await self._try_resume_paused(backend)
            if resumed > 0:
                logger.info(f"Resumed {resumed} paused programs after capacity enforcement")
        
        if paused > 0 or marked > 0:
            logger.info(f"Capacity enforcement: paused={paused}, marked={marked}, "
                       f"active={backend.active_program_tokens}, future_paused={backend.future_paused_tokens}")"""
        
        return paused, marked, resumed

    async def _claim_paused_for_backend(
        self, backend: BackendState
    ) -> Optional[Tuple[str, Program, PausedInfo]]:
        """Claim a paused program for a backend in a lock-protected way."""
        async with self.pause_resume_lock:
            paused_programs = self._get_paused_programs_sorted(ascending=True)
            paused_programs = (
                [p for p in paused_programs if p[1].step_count > 1]
                + [p for p in paused_programs if p[1].step_count <= 1]
            )

            for program_id, state, info in paused_programs:
                if program_id not in self.programs or state.state != ProgramState.PAUSED:
                    self.global_waiting_queue.pop(program_id, None)
                    continue
                if backend.has_capacity(extra_tokens=state.total_tokens, extra_count=1):
                    # Pop is the claim: only one backend can resume this program.
                    self.global_waiting_queue.pop(program_id, None)
                    return program_id, state, info

        return None

    async def _claim_specific_paused(
        self, program_id: str
    ) -> Optional[Tuple[str, Program, PausedInfo]]:
        """Claim a specific paused program in a lock-protected way."""
        async with self.pause_resume_lock:
            info = self.global_waiting_queue.get(program_id)
            if info is None:
                return None
            state = self.programs.get(program_id)
            if not state or state.state != ProgramState.PAUSED:
                self.global_waiting_queue.pop(program_id, None)
                return None
            self.global_waiting_queue.pop(program_id, None)
            return program_id, state, info

    async def _try_resume_paused(self, backend: BackendState) -> int:
        """Resume paused programs that fit within capacity.
        
        Resumes smallest programs first from the global pool, prioritizing
        non-first-step programs.
        Respects resume cooldown to prevent rapid oscillation.
        
        Returns: number of programs resumed
        """
        resumed = 0
        
        # Check cooldown before any resume
        if not backend.can_resume():
            logger.debug("Resume cooldown active, deferring resume")
            return 0
        
        while True:
            claim = await self._claim_paused_for_backend(backend)
            if claim is None:
                break
            program_id, state, info = claim
            self._resume_program(program_id, state, backend, paused_info=info)
            backend.record_resume()
            resumed += 1
            # Continue trying to backfill until no more candidates fit.
        
        return resumed

    # -------------------------------------------------------------------------
    # Periodic Scheduler (runs every 5s)
    # -------------------------------------------------------------------------

    async def _scheduler_loop(self):
        """Periodic scheduler loop: update shared_tokens, check thrashing, resume."""
        while not self._scheduler_stop:
            try:
                await asyncio.sleep(self._scheduler_interval)
                await self._scheduled_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}", exc_info=True)

    async def _scheduled_check(self):
        """Periodic check: update shared_tokens, enforce capacity, resume waiting programs."""
        
        # Step 1 & 2: For each backend, update shared_tokens and check thrashing
        for url, backend in self.backends.items():
            # Step 1: Update shared_tokens from vLLM metrics
            if backend.latest_metrics and backend.cache_config:
                vllm_actual_used = int(
                    backend.latest_metrics.kv_cache_usage_perc 
                    * backend.cache_config.total_tokens_capacity
                )
                # shared_tokens = what we track - what vLLM actually uses
                backend.shared_tokens = max(0, backend.reasoning_program_tokens - vllm_actual_used)
            
            # Step 2: Check thrashing and pause if needed
            if backend.cache_config and backend.remaining_capacity() < 0:
                await self._pause_until_safe(backend)
        
        # Step 3: Resume waiting programs using greedy algorithm
        await self._greedy_resume()

    async def _pause_until_safe(self, backend: BackendState):
        """Pause programs until backend is within capacity.
        
        Priority: ACTING first (smallest tokens), then REASONING (smallest tokens).
        """
        paused_count = 0
        
        while backend.remaining_capacity() < 0:
            # Priority 1: Pause ACTING programs (smallest first)
            acting_programs = self._get_acting_programs_sorted(backend.url, ascending=True)
            if acting_programs:
                program_id, state = acting_programs[0]
                self._pause_program(program_id, state)
                paused_count += 1
                logger.info(f"Scheduler paused ACTING program {program_id} (tokens={state.total_tokens})")
                continue
            
            # Priority 2: Pause REASONING programs (smallest first)
            # Note: REASONING programs are on GPU, we just mark them for pause
            reasoning_programs = self._get_reasoning_programs_sorted(backend.url, ascending=True)
            if reasoning_programs:
                program_id, state = reasoning_programs[0]
                self._mark_program_for_pause(program_id, state)
                paused_count += 1
                logger.info(f"Scheduler marked REASONING program {program_id} for pause (tokens={state.total_tokens})")
                # After marking, we've accounted for future_paused_tokens, continue checking
                continue
            
            # No more programs to pause
            break
        
        if paused_count > 0:
            logger.info(f"Scheduler paused/marked {paused_count} programs on {backend.url}")

    async def _greedy_resume(self):
        """Resume waiting programs using greedy algorithm.
        
        - Backends sorted by remaining capacity (largest first)
        - Programs sorted by total_tokens (smallest first)
        """
        from ..backend.state import BUFFER_PER_PROGRAM
        
        # Get backends with remaining capacity, sorted by capacity (largest first)
        backends_with_capacity = []
        for url, backend in self.backends.items():
            if not backend.cache_config or not backend.healthy:
                continue
            remaining = backend.remaining_capacity()
            if remaining > BUFFER_PER_PROGRAM:  # Need room for at least buffer
                backends_with_capacity.append((backend, remaining))
        
        if not backends_with_capacity:
            return
        
        backends_with_capacity.sort(key=lambda x: -x[1])  # Largest capacity first
        
        # Get waiting programs sorted by tokens (smallest first)
        async with self.pause_resume_lock:
            waiting = self._get_paused_programs_sorted(ascending=True)
            
            if not waiting:
                return
            
            resumed_count = 0
            for program_id, state, info in waiting:
                # Find a backend that can fit this program
                for i, (backend, remaining) in enumerate(backends_with_capacity):
                    required = state.total_tokens + BUFFER_PER_PROGRAM
                    if remaining >= required:
                        # Resume to this backend
                        self.global_waiting_queue.pop(program_id, None)
                        self._resume_program(program_id, state, target_backend=backend, paused_info=info)
                        resumed_count += 1
                        
                        # Update remaining capacity for next iteration
                        backends_with_capacity[i] = (backend, remaining - required)
                        break
            
            if resumed_count > 0:
                logger.info(f"Scheduler resumed {resumed_count} programs from waiting queue")

    async def _wait_for_resume(self, program_id: str, state: Program, timeout: float = 1800.0) -> None:
        """Wait for a paused program to be resumed.
        
        If timeout (30 min), force resume the program regardless of capacity.
        """
        if state.waiting_event is None:
            return
        
        try:
            await asyncio.wait_for(state.waiting_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Program {program_id} wait timeout after {timeout}s, forcing resume")
            claim = await self._claim_specific_paused(program_id)
            if claim is None:
                return
            program_id, state, info = claim
            
            # For new programs without backend, force assign to least loaded backend
            target_backend = None
            if state.backend_url is None:
                target_backend = self.select_backend_for_new_program_default()
                logger.info(f"Force assigning new program {program_id} to {target_backend.url}")
            
            self._resume_program(program_id, state, target_backend=target_backend, paused_info=info)

    def get_program_stats(self) -> Dict[str, Any]:
        """Get statistics about all programs."""
        reasoning = sum(1 for p in self.programs.values() if p.status == ProgramStatus.REASONING)
        acting = sum(1 for p in self.programs.values() if p.status == ProgramStatus.ACTING)
        paused = len(self.global_waiting_queue)
        marked = sum(1 for p in self.programs.values() if p.marked_for_pause)
        
        # Per-backend stats
        paused_counts = self.get_paused_counts_by_backend()
        per_backend = {}
        for url, backend in self.backends.items():
            progs = self.get_programs_on_backend(url)
            per_backend[url] = {
                "total": len(progs),
                "reasoning": sum(1 for p in progs.values() if p.status == ProgramStatus.REASONING),
                "acting": sum(1 for p in progs.values() if p.status == ProgramStatus.ACTING),
                "paused": paused_counts.get(url, 0),
                "marked_for_pause": sum(1 for p in progs.values() if p.marked_for_pause),
                "future_paused_tokens": backend.future_paused_tokens,
            }
        
        return {
            "total": len(self.programs),
            "reasoning": reasoning,
            "acting": acting,
            "paused": paused,
            "marked_for_pause": marked,
            "per_backend": per_backend,
        }

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

    async def proxy_request(
        self,
        backend: BackendState,
        payload: Dict[str, Any],
        *,
        on_usage: Callable[[int, int, int], Awaitable[None]] | None = None,
        on_first_token: Callable[[], None] | None = None,
        on_token: Callable[[], None] | None = None,
    ) -> Response:
        """Proxy request to a specific backend.
        
        Args:
            backend: Target backend
            payload: Request payload
            on_usage: Callback with (total_tokens, prompt_tokens, cached_tokens)
            on_first_token: Callback when first token is received (streaming only)
            on_token: Callback for each token (streaming only)
        """
        url = backend.completions_url
        
        if payload.get("stream"):
            return await forward_streaming_request(
                self.client,
                url,
                payload,
                on_usage=on_usage,
                on_first_token=on_first_token,
                on_token=on_token,
            )
        else:
            return await forward_non_streaming_request(
                self.client,
                url,
                payload,
                on_usage=on_usage,
            )

    async def proxy_get(self, backend_url: str, path: str) -> Response:
        """Proxy a GET request to a backend."""
        url = f"{backend_url.rstrip('/')}{path}"
        return await forward_get_request(self.client, url)

