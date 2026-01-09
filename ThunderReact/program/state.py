"""Program state management."""
import asyncio
from dataclasses import dataclass, field
from typing import Optional

from ..config import OUTPUT_TOKEN_ESTIMATE


@dataclass
class ProgramState:
    """Runtime state of a single program (task)."""
    context_len: int
    step_count: int = 0
    inflight: bool = False
    paused: bool = False
    transfer_target: Optional[str] = None
    resume_event: asyncio.Event = field(default_factory=asyncio.Event)
    waiting_on_resume: bool = False

    @property
    def est_tokens(self) -> int:
        """Estimate the token count occupied by this program."""
        return self.context_len + OUTPUT_TOKEN_ESTIMATE

    def __post_init__(self):
        if not self.paused:
            self.resume_event.set()
