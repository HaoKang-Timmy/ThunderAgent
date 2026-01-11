"""Program state management."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..profile.state import ProfileState


class ProgramStatus(Enum):
    """Status of a program."""
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class ProgramState:
    """State of a single program (task)."""
    backend_url: str  # Which backend this program is assigned to
    status: ProgramStatus = ProgramStatus.PAUSED
    context_len: int = 0
    total_tokens: int = 0
    step_count: int = 0
    profile: Optional["ProfileState"] = None  # Profile timing data (when profiling enabled)
