"""
ThunderReact - Multi-backend VLLM load balancing router.

Module structure:
- backend: Backend state management
- program: Program state management
- scheduler: Core scheduling and routing logic
- tools: Extension tools (reserved)
"""

from .config import (
    VLLM_BACKENDS,
    THRASHING_DISTANCE,
    CONTROL_TIME_WINDOW_S,
    CONTROL_STEP,
    PAUSE_TRIGGER_USAGE,
    RESUME_TRIGGER_USAGE,
    TRANSFER_IMBALANCE_TRIGGER,
    KV_CACHE_TOKEN_BUDGET,
    OUTPUT_TOKEN_ESTIMATE,
)
from .backend import BackendState
from .program import ProgramState
from .scheduler import MultiBackendRouter

__all__ = [
    # Config
    "VLLM_BACKENDS",
    "THRASHING_DISTANCE",
    "CONTROL_TIME_WINDOW_S",
    "CONTROL_STEP",
    "PAUSE_TRIGGER_USAGE",
    "RESUME_TRIGGER_USAGE",
    "TRANSFER_IMBALANCE_TRIGGER",
    "KV_CACHE_TOKEN_BUDGET",
    "OUTPUT_TOKEN_ESTIMATE",
    # State classes
    "BackendState",
    "ProgramState",
    # Router
    "MultiBackendRouter",
]

__version__ = "0.1.0"
