# install conda env vllm1

see [this markdown](evaluation/tau2-bench/RUN_LOCAL_GUIDE.md#plan-b-vllm-0101blackwell-gpu-支持)

# install conda env 2

see [this markdown](evaluation/retriever_env_setup.md#完整安装步骤)

# download ckpt

mkdir -p /workspace/ckpt/nvidia/Nemotron-Orchestrator-8B && rm -rf /workspace/ckpt/nvidia/Nemotron-Orchestrator-8B/*

export HF_TOKEN="hf_..."  # set your token (DO NOT commit secrets)

pip install -U "huggingface_hub"

pip install hf_transfer

hf download nvidia/Nemotron-Orchestrator-8B --local-dir /workspace/ckpt/nvidia/Nemotron-Orchestrator-8B

python evaluation/download_index.py

# download dataset (partial)

mkdir -p /workspace/dataset/multi-train/index

python evaluation/download_index.py