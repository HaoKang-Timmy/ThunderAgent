"""Backend state management."""
import asyncio
import logging
from typing import List, Optional

import httpx

from .vllm_metrics import VLLMMetrics, VLLMCacheConfig

logger = logging.getLogger(__name__)

# Keep only the most recent N metrics samples
METRICS_HISTORY_SIZE = 12


class BackendState:
    """State of a single VLLM backend with self-managed metrics monitoring."""
    
    def __init__(self, url: str):
        self.url = url
        self.healthy = True
        self.metrics_history: List[VLLMMetrics] = []
        
        # Static cache config (fetched once at startup)
        self.cache_config: Optional[VLLMCacheConfig] = None
        
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
        await self._fetch_cache_config()
        
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
    
    async def _fetch_cache_config(self) -> bool:
        """Fetch static cache config from vLLM (called once at startup)."""
        if not self._client:
            return False
        try:
            resp = await self._client.get(self.metrics_url)
            if resp.status_code == 200:
                self.cache_config = VLLMCacheConfig.from_prometheus_text(resp.text)
                logger.info(f"Fetched cache config for {self.url}: block_size={self.cache_config.block_size}, num_gpu_blocks={self.cache_config.num_gpu_blocks}, total_capacity={self.cache_config.total_tokens_capacity}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to fetch cache config from {self.url}: {e}")
            return False
    
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
    
    def to_dict(self) -> dict:
        """Convert to dict for API response."""
        result = {
            "url": self.url,
            "healthy": self.healthy,
            "monitoring": self._monitor_task is not None,
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
