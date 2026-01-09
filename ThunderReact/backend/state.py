"""Backend state management."""
import asyncio
from dataclasses import dataclass, field
from typing import Dict

from ..program import ProgramState


@dataclass
class BackendState:
    """State of a single VLLM backend."""
    url: str
    usage: float = 0.0
    healthy: bool = True
    running_requests: float = 0.0
    waiting_requests: float = 0.0
    programs: Dict[str, ProgramState] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    overload_start_time: float = 0.0
    underload_start_time: float = 0.0

    @property
    def metrics_url(self) -> str:
        """Prometheus metrics endpoint."""
        return f"{self.url}/metrics"

    @property
    def completions_url(self) -> str:
        """Chat completions API endpoint."""
        return f"{self.url}/v1/chat/completions"

    @property
    def total_inflight(self) -> int:
        """Number of requests currently being processed."""
        return sum(1 for p in self.programs.values() if p.inflight)

    @property
    def paused_count(self) -> int:
        """Number of paused programs."""
        return sum(1 for p in self.programs.values() if p.paused)
