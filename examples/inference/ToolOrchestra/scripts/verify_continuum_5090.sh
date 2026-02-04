#!/bin/bash
set -euo pipefail

# Verify vllm-continuum environment on RunPod 5090.
#
# Intended usage:
#   bash scripts/verify_continuum_5090.sh
#
# NOTE: This script is intended to be run on the 5090 box (not SLURM head node).

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm-continuum

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
python -c "import vllm; print(f'vLLM version: {getattr(vllm, \"__version__\", \"unknown\")}')"

python - <<'PY'
try:
    from vllm.config import SchedulerPolicy  # newer vLLM
except Exception:
    try:
        from vllm.config.scheduler import SchedulerPolicy  # some forks
    except Exception as e:
        SchedulerPolicy = None
        print(f"Could not import SchedulerPolicy: {e}")

if SchedulerPolicy is not None:
    args = getattr(SchedulerPolicy, "__args__", None)
    if args is None:
        print(f"SchedulerPolicy is present but has no __args__: {SchedulerPolicy}")
    else:
        print(f"continuum supported: {'continuum' in args}")
PY


