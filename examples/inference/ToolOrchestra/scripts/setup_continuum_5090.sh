#!/bin/bash
set -euo pipefail

# Setup vllm-continuum environment on RunPod 5090.
#
# NOTE: This script is intended to be run on the 5090 box (not SLURM head node).
# For SLURM cluster long-running jobs, follow oss-ToolOrchestra/.cursor/rules/work-on-slurm-cluster-via-tmux.mdc.

source ~/miniconda3/etc/profile.d/conda.sh

# Create new conda environment
conda create -n vllm-continuum python=3.12 -y
conda activate vllm-continuum

# Install PyTorch 2.8.0 for CUDA 12.8 (required by vllm-continuum baseline)
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install vllm-continuum from source (repo sibling of oss-ToolOrchestra in compare repo)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_ORCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPARE_ROOT="$(cd "${TOOL_ORCH_DIR}/.." && pwd)"

cd "${COMPARE_ROOT}/vllm-continuum"

rm -rf build/
rm -rf .deps/
rm -rf vllm.egg-info/
pip cache purge

export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST="12.0"
time pip install -v -e .

# Install tau2-bench dependencies
pip install hf_transfer matplotlib
cd "${TOOL_ORCH_DIR}/evaluation/tau2-bench"
pip install -e .

echo "Setup complete. Test with: python -c 'import vllm; print(vllm.__version__)'"


# python -c "from vllm import LLM, SamplingParams; llm = LLM(model='facebook/opt-125m'); print(llm.generate('Hello, my name is'))"


