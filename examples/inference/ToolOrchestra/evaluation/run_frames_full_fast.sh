#!/bin/bash
set -e
cd /home/junxiong/haokang/ToolOrchestra/evaluation
source /home/junxiong/miniconda3/etc/profile.d/conda.sh
conda activate vllm1
echo "Starting FRAMES full eval with fast wiki retrieval..."
echo "Config: model_config_oss_new_wiki_fast19.json"
echo "Output: outputs_oss_frames_full_new"
echo "Concurrency: 32"
echo "---"
python eval_frames_oss.py \
  --model_config model_config_oss_new_wiki_fast19.json \
  --output_dir outputs_oss_frames_full_new \
  --concurrency 32
