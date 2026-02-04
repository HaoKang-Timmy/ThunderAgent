"""Backend state management."""
import asyncio
import logging
import time
from typing import List, Optional, Set, TYPE_CHECKING

import httpx

from .vllm_metrics import VLLMMetrics, VLLMCacheConfig

if TYPE_CHECKING:
    from ..program.state import Program, ProgramState

from ..program.state import ProgramStatus

logger = logging.getLogger(__name__)

# Keep only the most recent N metrics samples
METRICS_HISTORY_SIZE = 12

# Buffer tokens reserved per program for decode phase
DECODE_BUFFER = 512

# Default coefficient for acting tokens (can be overridden per backend)
DEFAULT_TOOL_COEFFICIENT = 1.0

# Buffer tokens reserved per active program (for decode headroom)
BUFFER_PER_PROGRAM = 100

class BackendState:
    """State of a single VLLM backend with self-managed metrics monitoring."""
    
    def __init__(self, url: str, tool_coefficient: float = DEFAULT_TOOL_COEFFICIENT):
        self.url = url
        self.tool_coefficient = tool_coefficient
        self.healthy = True
        self.metrics_history: List[VLLMMetrics] = []
        
        # Static cache config (fetched once at startup)
        self.cache_config: Optional[VLLMCacheConfig] = None
        
        # Program tracking - all token stats are computed from this set
        self._programs: Set["Program"] = set()
        
        # Shared tokens (prefix cache savings), computed from vLLM metrics
        # = reasoning_program_tokens - vllm_actual_used_tokens
        self.shared_tokens: int = 0
        
        # Future paused tokens: sum of tokens from REASONING programs marked for pause
        # These will be released when they transition to ACTING
        self.future_paused_tokens: int = 0
        
        # Flag to skip concurrent scheduling (non-blocking, allows temporary overflow)
        self.scheduling_in_progress: bool = False
        
        # Metrics monitoring (self-managed)
        self._monitor_task: Optional[asyncio.Task] = None
        self._monitor_stop = False
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def completions_url(self) -> str:
        """Chat completions API endpoint."""
        return f"{self.url}/v1/chat/completions"
    
    @property
    def metrics_url(self) -> str:
        """Prometheus metrics endpoint."""
        return f"{self.url}/metrics"
    
    @property
    def latest_metrics(self) -> Optional[VLLMMetrics]:
        """Get the most recent metrics sample."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    @property
    def reasoning_program_tokens(self) -> int:
        """Sum of tokens from all REASONING programs."""
        return sum(p.total_tokens for p in self._programs 
                   if p.status == ProgramStatus.REASONING)
    
    @property
    def acting_program_tokens(self) -> int:
        """Sum of tokens from all ACTING programs."""
        return sum(p.total_tokens for p in self._programs 
                   if p.status == ProgramStatus.ACTING)
    
    @property
    def active_program_tokens(self) -> int:
        """Computed: reasoning_tokens + tool_coefficient * acting_tokens."""
        return int(self.reasoning_program_tokens + self.tool_coefficient * self.acting_program_tokens)
    
    @property
    def active_program_count(self) -> int:
        """Number of active (REASONING + ACTING) programs."""
        return len(self._programs)
    
    @property
    def total_program_tokens(self) -> int:
        """Sum of tokens from all programs on this backend."""
        return sum(p.total_tokens for p in self._programs)
    
    @property
    def active_program_tokens_ratio(self) -> float:
        """Ratio of active program tokens to total capacity."""
        if not self.cache_config or self.cache_config.total_tokens_capacity == 0:
            return 0.0
        return self.active_program_tokens / self.cache_config.total_tokens_capacity
    
    # -------------------------------------------------------------------------
    # Capacity Check
    # -------------------------------------------------------------------------
    
    def has_capacity(self, extra_tokens: int = 0, extra_count: int = 0) -> bool:
        """Check if adding extra tokens/programs would exceed capacity.
        
        Constraint: active_tokens - shared_tokens + buffer <= total_capacity
        where buffer = (active_count + extra_count) * BUFFER_PER_PROGRAM
        """
        if not self.cache_config:
            return True  # No config, assume ok
        
        tokens = self.active_program_tokens + extra_tokens
        count = self.active_program_count + extra_count
        buffer = count * BUFFER_PER_PROGRAM
        required = tokens - self.shared_tokens + buffer
        return required <= self.cache_config.total_tokens_capacity
    
    def capacity_overflow(self, include_future_release: bool = False) -> int:
        """Return how many tokens we're over capacity (0 if within capacity).
        
        Formula: active_tokens - shared_tokens + buffer - capacity
        where buffer = active_count * BUFFER_PER_PROGRAM
        
        Args:
            include_future_release: If True, subtract future_paused_tokens from the calculation.
                                   Use this when checking if more programs need to be paused.
        """
        if not self.cache_config:
            return 0
        buffer = self.active_program_count * BUFFER_PER_PROGRAM
        required = self.active_program_tokens - self.shared_tokens + buffer
        if include_future_release:
            required -= self.future_paused_tokens
        overflow = required - self.cache_config.total_tokens_capacity
        return max(0, overflow)
    
    def remaining_capacity(self) -> int:
        """Return remaining capacity for new programs (can be negative if over capacity)."""
        if not self.cache_config:
            return float('inf')
        buffer = self.active_program_count * BUFFER_PER_PROGRAM
        used = self.active_program_tokens - self.shared_tokens + buffer
        return self.cache_config.total_tokens_capacity - used
    
    # -------------------------------------------------------------------------
    # Program Registration
    # -------------------------------------------------------------------------
    
    def register_program(self, program: "Program") -> None:
        """Register a program with this backend.
        
        All token stats (reasoning_program_tokens, acting_program_tokens, etc.)
        are computed from the registered programs.
        """
        self._programs.add(program)
    
    def unregister_program(self, program: "Program") -> None:
        """Unregister a program from this backend."""
        self._programs.discard(program)
    
    # -------------------------------------------------------------------------
    # Metrics Monitoring (self-managed)
    # -------------------------------------------------------------------------
    
    async def start_monitoring(self, interval: float = 5.0):
        """Start background metrics monitoring for this backend."""
        if self._monitor_task is not None:
            return  # Already running
        
        self._monitor_stop = False
        self._client = httpx.AsyncClient(timeout=10.0)
        
        # Fetch cache config once at startup
        await self.fetch_cache_config()
        
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info(f"Started metrics monitoring for {self.url} (interval: {interval}s, kv_capacity: {self.cache_config.total_tokens_capacity if self.cache_config else 'unknown'} tokens)")
    
    async def stop_monitoring(self):
        """Stop background metrics monitoring."""
        if self._monitor_task is None:
            return
        
        self._monitor_stop = True
        self._monitor_task.cancel()
        try:
            await self._monitor_task
        except asyncio.CancelledError:
            pass
        self._monitor_task = None
        
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info(f"Stopped metrics monitoring for {self.url}")
    
    async def _monitor_loop(self, interval: float):
        """Background loop to periodically fetch metrics."""
        while not self._monitor_stop:
            try:
                await self._fetch_metrics()
            except Exception as e:
                logger.debug(f"Error fetching metrics from {self.url}: {e}")
            await asyncio.sleep(interval)
    
    async def fetch_cache_config(self) -> bool:
        """Fetch static cache config from vLLM.
        
        Can be called independently of start_monitoring().
        Uses existing client if monitoring, otherwise creates a temporary one.
        """
        client = self._client
        close_client = False
        
        if client is None:
            client = httpx.AsyncClient(timeout=10.0)
            close_client = True
        
        try:
            resp = await client.get(self.metrics_url)
            if resp.status_code == 200:
                self.cache_config = VLLMCacheConfig.from_prometheus_text(resp.text)
                logger.info(f"Fetched cache config for {self.url}: block_size={self.cache_config.block_size}, num_gpu_blocks={self.cache_config.num_gpu_blocks}, total_capacity={self.cache_config.total_tokens_capacity}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to fetch cache config from {self.url}: {e}")
            return False
        finally:
            if close_client:
                await client.aclose()
    
    async def _fetch_metrics(self) -> bool:
        """Fetch and update metrics from vLLM /metrics endpoint."""
        if not self._client:
            return False
        try:
            resp = await self._client.get(self.metrics_url)
            if resp.status_code == 200:
                metrics = VLLMMetrics.from_prometheus_text(resp.text)
                self.metrics_history.append(metrics)
                # Keep only the most recent samples
                if len(self.metrics_history) > METRICS_HISTORY_SIZE:
                    self.metrics_history = self.metrics_history[-METRICS_HISTORY_SIZE:]
                self.healthy = True
                return True
            else:
                self.healthy = False
                return False
        except Exception as e:
            logger.debug(f"Failed to fetch metrics from {self.url}: {e}")
            self.healthy = False
            return False
    
    def to_dict(self, *, paused_program_count: Optional[int] = None) -> dict:
        """Convert to dict for API response."""
        if paused_program_count is None:
            paused_program_count = 0
        result = {
            "url": self.url,
            "healthy": self.healthy,
            "monitoring": self._monitor_task is not None,
            "active_program_tokens": self.active_program_tokens,
            "reasoning_program_tokens": self.reasoning_program_tokens,
            "acting_program_tokens": self.acting_program_tokens,
            "active_program_count": self.active_program_count,
            "active_program_tokens_ratio": round(self.active_program_tokens_ratio, 4),
            "total_program_tokens": self.total_program_tokens,
            "paused_program_count": paused_program_count,
            "future_paused_tokens": self.future_paused_tokens,
            "shared_tokens": self.shared_tokens,
            "buffer_per_program": BUFFER_PER_PROGRAM,
            "capacity_overflow": self.capacity_overflow(),
        }
        # Include cache config (static)
        if self.cache_config:
            result["cache_config"] = {
                "block_size": self.cache_config.block_size,
                "num_gpu_blocks": self.cache_config.num_gpu_blocks,
                "total_tokens_capacity": self.cache_config.total_tokens_capacity,
            }
        # Include latest metrics (dynamic)
        if self.metrics_history:
            latest = self.latest_metrics
            result["metrics"] = {
                "num_requests_running": latest.num_requests_running,
                "num_requests_waiting": latest.num_requests_waiting,
                "kv_cache_usage_perc": round(latest.kv_cache_usage_perc, 4),
                "prefix_cache_hit_rate": round(latest.prefix_cache_hit_rate, 4),
                "prefix_cache_queries": latest.prefix_cache_queries,
                "prefix_cache_hits": latest.prefix_cache_hits,
                "prompt_tokens_total": latest.prompt_tokens_total,
                "generation_tokens_total": latest.generation_tokens_total,
                "num_preemptions": latest.num_preemptions,
                "requests_completed": latest.total_requests_completed,
                "last_updated": latest.timestamp,
                "history_size": len(self.metrics_history),
            }
        return result
