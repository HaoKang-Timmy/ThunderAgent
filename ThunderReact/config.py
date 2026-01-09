"""ThunderReact configuration constants."""
import os

# Backend service configuration
VLLM_BACKENDS = os.getenv(
    "VLLM_BACKENDS",
    "http://localhost:8100"
).split(",")

# Controller parameters
THRASHING_DISTANCE = 0.05
CONTROL_TIME_WINDOW_S = 5.0
CONTROL_STEP = 0.2

# Pause/resume trigger thresholds
PAUSE_TRIGGER_USAGE = 0.95
RESUME_TRIGGER_USAGE = 0.85
TRANSFER_IMBALANCE_TRIGGER = 0.30

# Token budget
KV_CACHE_TOKEN_BUDGET = 788976
OUTPUT_TOKEN_ESTIMATE = 500
