# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import time
import requests
import subprocess, signal

SERVE_REPEAT = 1
serve_script1 = """#!/bin/bash

#SBATCH --account nvr_lpr_llm
#SBATCH --partition batch_block1,interactive
#SBATCH --time 04:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --job-name EXPERIMENT_NAME
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=slurm_out/EXPERIMENT_NAME.out
#SBATCH --error=slurm_out/EXPERIMENT_NAME.err

set -x

hostname -i
export HF_HOME=cache/huggingface
source ~/.bashrc
conda activate retriever
CUDA_VISIBLE_DEVICES=0,1 python retrieval_wiki.py --port 1401 &
conda activate vllm1
CUDA_VISIBLE_DEVICES=2,3,4,5 vllm serve Qwen/Qwen2.5-Math-72B-Instruct --port 1402 --tensor-parallel-size 4 &
CUDA_VISIBLE_DEVICES=6,7 vllm serve Qwen/Qwen3-32B --port 1403 --tensor-parallel-size 2

sleep 15000"""

serve_script2 = '''#!/bin/bash

#SBATCH --account nvr_lpr_llm
#SBATCH --partition batch_block1,interactive
#SBATCH --time 04:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --job-name EXPERIMENT_NAME
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=slurm_out/EXPERIMENT_NAME.out
#SBATCH --error=slurm_out/EXPERIMENT_NAME.err

set -x

hostname -i
export HF_HOME=cache/huggingface
source ~/.bashrc
conda activate vllm1
CUDA_VISIBLE_DEVICES=5 vllm serve Qwen/Qwen2.5-Math-7B-Instruct --port 1404 &
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve meta-llama/Llama-3.3-70B-Instruct --port 1405 --enable-auto-tool-choice --tool-call-parser llama3_json --chat-template tool_chat_template_llama3.1_json.jinja --tensor-parallel-size 4 &
CUDA_VISIBLE_DEVICES=4 vllm serve checkpoint_dir --enable-auto-tool-choice --tool-call-parser hermes --port 1406 &
CUDA_VISIBLE_DEVICES=6,7 vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --port 1407 --tensor-parallel-size 2

sleep 15000'''

def get_jobs():
    exec_result = subprocess.run(['squeue', '-u',os.environ.get('USER',None)], timeout=3600, capture_output=True, text=True)
    lines = exec_result.stdout.strip().split('\n')[1:]
    jobs = []
    for l in lines:
        components = l.split(' ')
        components = [e for e in components if e!='']
        running_time = components[5]
        total_time = 0
        time_components = running_time.split(':')
        if '-' in time_components[0]:
            total_time = 3600
        elif len(time_components)==2:
            total_time = int(time_components[0])*60+int(time_components[1])
        elif len(time_components)==3:
            total_time = int(time_components[0])*3600+int(time_components[1])*60+int(time_components[2])
        jobs.append({
            'name': components[2],
            'id': components[0],
            'status': components[4],
            'total_time': total_time,
            'reason': components[-1]
        })
    return jobs

SERVE_IPS1 = []
SERVE_IPS2 = []
run_done = True
output_dir = 'outputs/frames'
while True:
    jobs = get_jobs()
    for j in jobs:
        if j['reason'].strip().lower()=='held)':
            os.system(f"scancel {j['id']}")
            time.sleep(120)
    cur_ckpt_dir = os.getenv("CKPT_DIR")
    serve_collections = []
    for repeat in range(SERVE_REPEAT):
        exp_name1 = f"op_1{repeat}"
        serve_collections.append(exp_name1)
        cur_serve_script = serve_script1
        cur_serve_script = cur_serve_script.replace('EXPERIMENT_NAME',exp_name1)
        with open(f'{exp_name1}.sh','w') as f:
            f.write(cur_serve_script)
        exp_name2 = f"run_{repeat}"
        serve_collections.append(exp_name2)
        cur_serve_script = serve_script2.replace('checkpoint_dir',cur_ckpt_dir)
        cur_serve_script = cur_serve_script.replace('EXPERIMENT_NAME',exp_name2)
        with open(f'{exp_name2}.sh','w') as f:
            f.write(cur_serve_script)
    jobs = get_jobs()
    job_names = [j['name'] for j in jobs]
    for j in jobs:
        if j['name'] not in serve_collections and j['name'].startswith('op'):
            os.system(f"scancel {j['id']}")
    for repeat in range(SERVE_REPEAT):
        exp_name = f"op_1{repeat}"
        if not exp_name in job_names:
            if os.path.isfile(f'slurm_out/{exp_name}.out'):
                os.remove(f'slurm_out/{exp_name}.out')
            os.system('sbatch '+f' {exp_name}.sh')
        exp_name = f"run_{repeat}"
        if not exp_name in job_names:
            if os.path.isfile(f'slurm_out/{exp_name}.out'):
                os.remove(f'slurm_out/{exp_name}.out')
            os.system('sbatch '+f' {exp_name}.sh')
    job_ids = [j['id'] for j in jobs]
    already_serve = []
    for j in jobs:
        if j['name'] in serve_collections and j['status'].strip().lower()=='r':
            if not os.path.isfile(f'slurm_out/{j["name"]}.out'):
                os.system(f"scancel {j['id']}")
            else:
                if j['total_time']>=600:
                    already_serve.append({
                        'name': j['name'],
                        'total_time': j['total_time']
                    })
    if len(already_serve)!=2:
        time.sleep(30)
        continue
    all_times = [s['total_time'] for s in already_serve]
    if max(all_times)<600:
        time.sleep(600-max(all_times))
    serve_ips1 = []
    for repeat in range(SERVE_REPEAT):
        exp_name = f"op_1{repeat}"
        with open(f'slurm_out/{exp_name}.out') as f:
            lines = f.readlines()
        serve_ip = lines[0].strip()
        serve_ips1.append(serve_ip)
    serve_ips2 = []
    for repeat in range(SERVE_REPEAT):
        exp_name = f"run_{repeat}"
        with open(f'slurm_out/{exp_name}.out') as f:
            lines = f.readlines()
        serve_ip = lines[0].strip()
        serve_ips2.append(serve_ip)
    change_flag = False
    if os.path.isfile('model_configs/serve_frames.json'):
        with open('model_configs/serve_frames.json') as f:
            old_config = json.load(f)
        if not cur_ckpt_dir in old_config:
            change_flag = True
    if SERVE_IPS1!=serve_ips1 or SERVE_IPS2!=serve_ips2 or change_flag:
        SERVE_IPS1 = serve_ips1
        SERVE_IPS2 = serve_ips2
        model_config = {
            "retrieval": [],
            "Qwen/Qwen2.5-Math-72B-Instruct": [],
            "Qwen/Qwen3-32B": [],
            "Qwen/Qwen2.5-Math-7B-Instruct": [],
            "meta-llama/Llama-3.3-70B-Instruct": [],
            cur_ckpt_dir: [],
            "Qwen/Qwen2.5-Coder-32B-Instruct": [],
            "vllm_model_config_path": "model_configs/serve_frames.json"
        }
        for sip in serve_ips1:
            model_config["retrieval"].append({
                    "ip_addr": sip,
                    "port": "1401"
                })
            model_config["Qwen/Qwen2.5-Math-72B-Instruct"].append({
                    "ip_addr": sip,
                    "port": "1402"
                })
            model_config["Qwen/Qwen3-32B"].append({
                    "ip_addr": sip,
                    "port": "1403"
                })
        for sip in serve_ips2:
            model_config["Qwen/Qwen2.5-Math-7B-Instruct"].append({
                    "ip_addr": sip,
                    "port": "1404"
                })
            model_config["meta-llama/Llama-3.3-70B-Instruct"].append({
                    "ip_addr": sip,
                    "port": "1405"
                })
            model_config[cur_ckpt_dir].append({
                    "ip_addr": sip,
                    "port": "1406"
                })
            model_config["Qwen/Qwen2.5-Coder-32B-Instruct"].append({
                    "ip_addr": sip,
                    "port": "1407"
                })
        with open('model_configs/serve_frames.json','w') as f:
            json.dump(model_config,f,indent=2)

    cur_output_dir = os.path.join(output_dir,f"26")
    os.system(f"python eval_frames.py --model_name {cur_ckpt_dir} --output_dir {cur_output_dir} --model_config model_configs/serve_frames.json")

    time.sleep(30)

        

