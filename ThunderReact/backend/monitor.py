"""Backend metrics monitoring."""
import asyncio
import logging
from typing import Dict, Optional

import httpx

from .state import BackendState

logger = logging.getLogger(__name__)


class MetricsMonitor:
    """Background monitor for fetching vLLM metrics from all backends."""
    
    def __init__(self, backends: Dict[str, BackendState], interval: float = 5.0):
        """Initialize the metrics monitor.
        
        Args:
            backends: Dict of backend URL -> BackendState
            interval: Seconds between metrics fetches
        """
        self.backends = backends
        self.interval = interval
        self._task: Optional[asyncio.Task] = None
        self._stop = False
        self._client: Optional[httpx.AsyncClient] = None
    
    async def start(self):
        """Start the metrics monitoring background task."""
        self._stop = False
        self._client = httpx.AsyncClient(timeout=10.0)
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Started metrics monitoring (interval: {self.interval}s)")
    
    async def stop(self):
        """Stop the metrics monitoring background task."""
        self._stop = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("Stopped metrics monitoring")
    
    async def _monitor_loop(self):
        """Background loop to periodically fetch metrics."""
        while not self._stop:
            try:
                await self._fetch_all()
            except Exception as e:
                logger.warning(f"Error fetching metrics: {e}")
            await asyncio.sleep(self.interval)
    
    async def _fetch_all(self):
        """Fetch metrics from all backends in parallel."""
        if not self._client:
            return
        tasks = [backend.fetch_metrics(self._client) for backend in self.backends.values()]
        await asyncio.gather(*tasks, return_exceptions=True)

