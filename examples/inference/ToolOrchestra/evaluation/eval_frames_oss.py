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

"""
FRAMES Evaluation with OSS Expert Models

Model Mapping:
- All reasoners → Qwen/Qwen2.5-Coder-14B-Instruct (vLLM, TP=1, DP=8)
- All search → openai/gpt-oss-20b (vLLM, TP=1, DP=4)
- All answer (including math) → Qwen/Qwen3-32B-FP8 (vLLM, TP=1, DP=8, non-thinking mode)
- LLM-as-judge → gpt-5 (proprietary, unchanged)
"""

import os
import random
import time
import json
import requests
import asyncio
import subprocess
import traceback
from tqdm import tqdm
from transformers import AutoTokenizer
import sys
from openai import OpenAI

# HuggingFace Hub fast-download mode
if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") in {"1", "true", "True"}:
    try:
        import hf_transfer  # noqa: F401
    except Exception:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        print(
            "[eval_frames_oss] WARNING: HF_HUB_ENABLE_HF_TRANSFER=1 but 'hf_transfer' is not installed; "
            "disabling fast downloads.",
            flush=True,
        )

REPO_PATH = os.getenv("REPO_PATH") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
from LLM_CALL import get_llm_response
import multiprocessing as mp
import argparse
import logging
logging.disable(logging.CRITICAL)

MODEL_NAME = None
my_output_dir = None
MAX_ROUNDS = None
MODEL_TYPE = None
MODEL_MAPPING = None
TOOL_PRICING = None
vllm_model_configs = None
with open('tools.json') as f:
    raw_tools = json.load(f)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# OSS Expert Model Mapping
# All tools map to OSS models served via vLLM
MODEL_MAPPING = {
    # All reasoners → Qwen2.5-Coder-14B-Instruct (DP=8)
    "reasoner-1": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "reasoner-2": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "reasoner-3": "Qwen/Qwen2.5-Coder-14B-Instruct",

    # All search → GPT-OSS-20b (DP=4)
    "search-1": "openai/gpt-oss-20b",
    "search-2": "openai/gpt-oss-20b",
    "search-3": "openai/gpt-oss-20b",

    # All answer (including math) → Qwen3-32B-FP8 (non-thinking mode, DP=8)
    "answer-1": "Qwen/Qwen3-32B-FP8",
    "answer-2": "Qwen/Qwen3-32B-FP8",
    "answer-3": "Qwen/Qwen3-32B-FP8",
    "answer-4": "Qwen/Qwen3-32B-FP8",
    "answer-math-1": "Qwen/Qwen3-32B-FP8",
    "answer-math-2": "Qwen/Qwen3-32B-FP8",
}

TOOL_PRICING = {
    "Qwen/Qwen2.5-Coder-14B-Instruct": {
        "input_tokens_per_million": 0.4/1000000,
        "output_tokens_per_million": 0.4/1000000
    },
    "openai/gpt-oss-20b": {
        "input_tokens_per_million": 0.3/1000000,
        "output_tokens_per_million": 0.3/1000000
    },
    "Qwen/Qwen3-32B-FP8": {
        "input_tokens_per_million": 0.6/1000000,
        "output_tokens_per_million": 0.6/1000000
    },
    "gpt-5": {
        "input_tokens_per_million": 1.25/1000000,
        "output_tokens_per_million": 10/1000000
    },
    "code_interpreter_per_second": 0.0000083,
    "tavily": {
        "search": 0.01,
        "extract": 0.002
    },
}

ALL_TOOLS = {
    "enhance_reasoning": {
        'model': ["reasoner-1", "reasoner-2", "reasoner-3"]
    },
    "answer": {
        'model': ["answer-math-1", "answer-math-2", "answer-1", "answer-2", "answer-3", "answer-4"]
    },
    "search": {
        "model": ["search-1", "search-2", "search-3"]
    },
}

def cut_seq(seq,l):
    if len(seq)==0:
        return {
            'effective_length': 0,
            'string_after_cut': ''
        }
    token_ids = tokenizer(seq)['input_ids']
    rs = tokenizer.batch_decode(token_ids[-l:], skip_special_tokens=True)
    return {
        'effective_length': len(token_ids),
        'string_after_cut': ''.join(rs)
    }

def call_tool(arguments):
    tool_name = arguments.get("tool")
    t0 = time.perf_counter()

    try:
        if arguments['tool']=='enhance_reasoning':
            # All reasoners use Qwen3-Coder-30B-A3B via vLLM
            supported_models = [MODEL_MAPPING[m] for m in ALL_TOOLS['enhance_reasoning']['model']]
            assert arguments['model'] in supported_models, f"Model {arguments['model']} is not supported. Supported: {supported_models}"

            prompt = arguments['context_str'].strip()+'\n\n'
            prompt += f"Question: {arguments['problem']}\nInstead of directly answering the question, please write additional python code that will give intermidiate results after execution. Wrap the code within ```python and ```. The code should be self-contained with all the import and initialization."
            model_name = arguments['model']

            # Qwen3-Coder via vLLM
            llm_t0 = time.perf_counter()
            response = get_llm_response(
                model=model_name,
                messages=prompt,
                return_raw_response=True,
                model_type='vllm',
                max_length=8000,
                temperature=0.2,
                model_config=arguments['vllm_model_configs'][model_name],
                model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],
                model_config_idx=arguments['eid']
            )
            arguments["_llm_ms"] = (time.perf_counter() - llm_t0) * 1000.0

            if isinstance(response, str):
                arguments['generated_code'] = ''
                arguments['exec_result'] = ''
                arguments["_exec_ms"] = 0.0
                return arguments

            try:
                generated_code = response.choices[0].message.content.split('```python')[-1].split('```')[0]
            except:
                generated_code = ''

            if generated_code=='':
                arguments['generated_code'] = ''
                arguments['exec_result'] = ''
                arguments["_exec_ms"] = 0.0
                return arguments

            code_path = str(os.path.join(arguments['cur_output_dir'],f'exec_code_{arguments["id"]}.py'))
            with open(code_path,'w') as f:
                f.write(generated_code)

            exec_result = ''
            exec_t0 = time.perf_counter()
            try:
                exec_result = subprocess.run(['python', code_path], timeout=60, capture_output=True, text=True)
                exec_result = exec_result.stdout
                with open(os.path.join(arguments['cur_output_dir'],f'exec_out_{arguments["id"]}.txt'),'w') as f:
                    f.write(exec_result)
            except Exception as e:
                pass
            arguments["_exec_ms"] = (time.perf_counter() - exec_t0) * 1000.0

            arguments['generated_code'] = generated_code
            arguments['exec_result'] = exec_result
            return arguments

        elif arguments['tool']=='answer':
            # All answer models use Qwen3-32B-FP8 via vLLM in non-thinking mode
            prompt = arguments['context_str'].strip()+'\n\n'+arguments['problem']
            response_str = ''
            pred = ''

            model_name = arguments['model']
            # Use <think>/<answer> format for structured output
            prompt2 = prompt + "\nWrap the thinking process and explanation between <think> and </think> and wrap only the exact answer without any explanation within <answer> and </answer>."
            arguments['messages'] = prompt2

            # Use vLLM for Qwen3-32B-FP8 with non-thinking mode
            expert_t0 = time.perf_counter()
            response = get_llm_response(
                model=model_name,
                messages=prompt2,
                return_raw_response=True,
                model_type='vllm',
                max_length=8000,
                temperature=0.2,
                model_config=arguments['vllm_model_configs'][model_name],
                model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],
                model_config_idx=arguments['eid'],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            arguments["_expert_ms"] = (time.perf_counter() - expert_t0) * 1000.0

            if isinstance(response, str):
                arguments['response'] = ''
                arguments['pred'] = ''
                arguments['correctness'] = False
                return arguments

            response_str = response.choices[0].message.content
            if isinstance(response_str, str):
                pred = response_str.split('<answer>')[-1].split('</answer>')[0].strip()
            else:
                pred = ''

            if pred.strip()=='' or len(pred.split(' '))>500:
                correctness = False
            elif pred.strip().lower()==arguments['answer'].strip().lower():
                correctness = True
            else:
                # LLM-as-judge using gpt-5 (proprietary)
                eval_prompt = (f"Question: {arguments['problem']}\n\n"
                            f"Student answer: {pred}\n\n"
                            f"Reference answer: {arguments['answer']}\n\n"
                            "Assume that the reference answer is correct. Output <correct>True</correct> if the student answer matches the reference answer. Output <correct>False</correct> if the student answer does not match the reference answer.")
                judge_t0 = time.perf_counter()
                eval_response = get_llm_response(model='gpt-5', messages=eval_prompt, temperature=1)
                arguments["_judge_ms"] = (time.perf_counter() - judge_t0) * 1000.0
                eval_result = eval_response.split('<correct>')[-1].split('</correct>')[0]
                correctness = eval_result.lower() == 'true'

            arguments['response'] = response_str
            arguments['pred'] = pred
            arguments['correctness'] = correctness
            return arguments

        elif arguments['tool']=='search':
            # All search models use GPT-OSS-20b via vLLM
            contents = []
            prompt = arguments['context_str'].strip()+'\n\n'
            prompt += f"Question: {arguments['problem']}\nInstead of directly answering the question, please think hard and write a concise query to search Wikipedia. Wrap the query within <query> and </query>."
            cur_query_writer = arguments['model']
            query_to_call = None

            # GPT-OSS-20b via vLLM
            query_llm_t0 = time.perf_counter()
            response = get_llm_response(
                model=cur_query_writer,
                messages=[{"role": "user", "content": prompt}],
                return_raw_response=True,
                model_type='vllm',
                max_length=8000,
                temperature=0.2,
                model_config=arguments['vllm_model_configs'][cur_query_writer],
                model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],
                model_config_idx=arguments['eid']
            )
            arguments["_query_llm_ms"] = (time.perf_counter() - query_llm_t0) * 1000.0

            if isinstance(response, str) or not response.choices[0].message.content:
                query_to_call = arguments['problem']
            else:
                query_to_call = response.choices[0].message.content.split('<query>')[-1].split('</query>')[0]

            arguments["_retrieval_ms"] = 0.0
            arguments["_retrieval_retries"] = 0
            if query_to_call is None or len(query_to_call)<10:
                pass
            else:
                assert len(query_to_call)>5, f"{query_to_call}"
                payload = {
                    "queries": [query_to_call[:390]],
                    "topk": 150,
                    "return_scores": True,
                    "eid": arguments['id']
                }
                results = None
                all_vllm_model_configs = arguments['vllm_model_configs']
                max_attempts = int(os.getenv("FRAMES_RETRIEVAL_MAX_ATTEMPTS", "8"))
                connect_timeout_s = float(os.getenv("FRAMES_RETRIEVAL_CONNECT_TIMEOUT_S", "3"))
                read_timeout_s = float(os.getenv("FRAMES_RETRIEVAL_READ_TIMEOUT_S", "60"))
                base_backoff_s = float(os.getenv("FRAMES_RETRIEVAL_BACKOFF_S", "2"))

                last_err: Exception | None = None
                retrieval_t0 = time.perf_counter()
                retrieval_retries = 0

                # Prefer wiki retriever, but fall back to general retriever if wiki is down.
                endpoint_pools = []
                if all_vllm_model_configs.get('wiki_retrieval'):
                    endpoint_pools.append(('wiki_retrieval', all_vllm_model_configs['wiki_retrieval']))
                if all_vllm_model_configs.get('retrieval'):
                    endpoint_pools.append(('retrieval', all_vllm_model_configs['retrieval']))

                for pool_name, endpoints in endpoint_pools:
                    if results is not None:
                        break
                    for attempt in range(max_attempts):
                        try:
                            if not endpoints:
                                raise RuntimeError(f"no_endpoints_configured:{pool_name}")
                            cur_model_config = random.choice(endpoints)
                            url = f'http://{cur_model_config["ip_addr"]}:{cur_model_config["port"]}/retrieve'
                            resp = requests.post(url, json=payload, timeout=(connect_timeout_s, read_timeout_s))
                            resp.raise_for_status()
                            results = resp.json()
                            last_err = None
                            break
                        except Exception as e:
                            last_err = e
                            retrieval_retries += 1
                            sleep_s = min(base_backoff_s * (2 ** min(attempt, 4)), 15.0)
                            time.sleep(sleep_s)

                arguments["_retrieval_ms"] = (time.perf_counter() - retrieval_t0) * 1000.0
                arguments["_retrieval_retries"] = retrieval_retries
                if results is None:
                    print(
                        f"[search] WARNING: retrieval failed after trying {len(endpoint_pools)} pool(s); "
                        f"continuing with empty results. last_err={repr(last_err)}",
                        flush=True,
                    )
                if results:
                    for r in results[0]:
                        if 'content' in r['document']:
                            contents.append(r['document']['content'])
                        elif 'contents' in r['document']:
                            contents.append(r['document']['contents'])

            arguments['search_results_data'] = contents
            if 'tokenizer' in arguments:
                arguments.pop('tokenizer')
            return arguments

        # Unknown tool
        arguments["tool_error"] = f"unknown_tool:{tool_name}"
        arguments["tool_error_type"] = "unknown_tool"
        return arguments

    except Exception as e:
        try:
            arguments["tool_error"] = repr(e)
            arguments["tool_error_type"] = type(e).__name__
            arguments["tool_error_traceback"] = traceback.format_exc()
        except Exception:
            pass
        return arguments

import asyncio
import contextlib
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Tuple, Any, Callable

# task_list is an iterable of (func, arg) pairs
async def run_all(
    task_list: Iterable[Tuple[Callable[[Any], Any], Any]],
    concurrency: int = 2,
    progress: bool = False,
    return_exceptions: bool = False,
):
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(concurrency)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        async def run_one(idx: int, func: Callable, arg: Any):
            async with sem:
                try:
                    if asyncio.iscoroutinefunction(func):
                        res = await func(arg)
                    else:
                        res = await loop.run_in_executor(executor, func, arg)
                    return idx, res, None
                except Exception as e:
                    return idx, None, e

        task_list = list(task_list)
        tasks = [asyncio.create_task(run_one(i, f, a))
                 for i, (f, a) in enumerate(task_list)]

        results = [None] * len(tasks)

        if progress:
            from tqdm import tqdm
            pbar = tqdm(total=len(tasks))
        else:
            pbar = None

        try:
            for fut in asyncio.as_completed(tasks):
                idx, res, err = await fut
                if err is None:
                    results[idx] = res
                else:
                    if return_exceptions:
                        results[idx] = err
                    else:
                        for t in tasks:
                            t.cancel()
                        with contextlib.suppress(Exception):
                            await asyncio.gather(*tasks, return_exceptions=True)
                        raise err
                if pbar:
                    pbar.update(1)
        finally:
            if pbar:
                pbar.close()

        return results

def run_single(e):
    out_path = os.path.join(my_output_dir, f"{e['id']}.json")
    if os.path.isfile(out_path):
        print(f"[FRAMES_TASK_COMPLETE] id={e['id']} status=skipped", flush=True)
        return {"id": e["id"], "skipped": True}

    doc_list = []
    code_list = []
    attempt_list = []
    exp_start_time = time.time()
    problem = e['question']
    user_problem = problem
    answer = e['answer']
    all_tool_calls = []
    final_correct = False
    final_answer_model = None
    final_pred = ''
    all_tool_responses = {}
    used_tools = []
    task_status = "done"
    task_error_type = None

    for step in range(MAX_ROUNDS):
        cur_output_dir = os.path.join(my_output_dir, f"step_{step}")
        if not os.path.isdir(os.path.join(cur_output_dir, 'tool_return')):
            try:
                os.makedirs(os.path.join(cur_output_dir, 'tool_return'))
            except:
                pass

        tools = []
        for t in raw_tools:
            if len(doc_list)>0:
                if t['function']['name']!='search':
                    tools.append(t)
            else:
                tools.append(t)

        doc_str = ''
        for doc_idx, doc in enumerate(doc_list):
            doc_str += f"Doc {doc_idx+1}: {doc[:4000]}\n\n"
        code_str = ''
        for code_idx, code_piece in enumerate(code_list):
            code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
        attempt_str = ''
        for attempt_idx, attempt in enumerate(attempt_list):
            attempt_str += f"Attempt{attempt_idx+1} answer by {attempt['model']}: {attempt['answer']}\n"
        str_cut = cut_seq(seq=attempt_str, l=8000)
        attempt_str = str_cut['string_after_cut']
        if not attempt_str.startswith('Attempt') and len(attempt_str)>0:
            attempt_str = 'Attempt answer: '+attempt_str
        str_cut = cut_seq(seq=code_str+attempt_str, l=12000)
        code_attempt_str = str_cut['string_after_cut']
        code_attempt_str_len = str_cut['effective_length']
        if not code_attempt_str.startswith('```') and len(code_attempt_str)>0:
            code_attempt_str = '```\n'+code_attempt_str
        doc_flag = False
        if code_attempt_str_len<24000:
            context_str = cut_seq(seq=doc_str+"\npython code and execution outputs:\n"+code_attempt_str, l=24000)
            context_str = context_str['string_after_cut']
            if len(doc_str)>0:
                doc_flag = True
                context_str = 'Documents:\n'+context_str
        else:
            context_str = code_attempt_str

        removed_tool = None
        if len(used_tools)>1 and used_tools[-1]==used_tools[-2]:
            updated_tools = []
            removed_tool = used_tools[-1]
            for t in tools:
                if t['function']['name']!=used_tools[-1]:
                    updated_tools.append(t)
        else:
            updated_tools = tools
        cur_tool_set = [t['function']['name'] for t in updated_tools]
        chat = [
                    {"role": "system", "content": "You are good at using tools."},
                    {"role": "user", "content": f"Problem: {problem}\n\n{context_str}\n\nChoose an appropriate tool."}
                ]
        response = get_llm_response(
            model=MODEL_NAME,
            messages=chat,
            return_raw_response=True,
            model_type='vllm',
            model_config=vllm_model_configs[MODEL_NAME],
            temperature=1,
            max_length=12000,
            tools=tools,
            model_config_path=vllm_model_configs['vllm_model_config_path'],
            model_config_idx=e['eid']
        )

        if isinstance(response, str):
            continue
        tool_calls = response.choices[0].message.tool_calls
        if len(tool_calls)==0:
            all_tool_calls.append(f'342 invalid tool calls {tool_calls}')
            continue
        tool_call_list = []
        cur_tool_calls = []
        processed_tools = set()
        for one_tool_call in tool_calls:
            tool_name = one_tool_call.function.name
            try:
                tool_arguments = json.loads(one_tool_call.function.arguments)
            except:
                tool_arguments = None
            if not tool_name in ALL_TOOLS:
                cur_tool_calls.append(f'350 invalid tool calls {tool_calls}')
                continue
            if not isinstance(tool_arguments, dict):
                cur_tool_calls.append(f'351 invalid tool call args {tool_calls}')
                continue
            func_signature = ALL_TOOLS[tool_name]
            valid_tool_call = True
            for parameter_name, parameter_values in func_signature.items():
                if (not parameter_name in tool_arguments):
                    valid_tool_call = False
                    continue
                if (not tool_arguments[parameter_name] in parameter_values) and parameter_values!='any':
                    valid_tool_call = False
            if not valid_tool_call:
                cur_tool_calls.append(f'360 invalid tool calls {tool_calls}')
                continue

            if tool_name in processed_tools:
                continue
            processed_tools.add(tool_name)
            tool_call = {
                'name': tool_name,
                'arguments': tool_arguments
            }
            cur_tool_calls.append(tool_call)
            expert_model_to_call = MODEL_MAPPING[tool_arguments['model']]

            call_tool_argument = None
            used_tools.append(tool_name)

            # All OSS models use similar context length limits
            max_code_length = 12000
            max_context_length = 24000

            if tool_name=='enhance_reasoning':
                doc_str = ''
                for doc_idx, doc in enumerate(doc_list):
                    doc_str += f"Doc {doc_idx+1}: {doc}\n\n"
                code_str = ''
                for code_idx, code_piece in enumerate(code_list):
                    code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                str_cut = cut_seq(seq=code_str, l=max_code_length)
                code_str = str_cut['string_after_cut']
                if not code_str.startswith('```') and len(code_str)>0:
                    code_str = '```\n'+code_str
                problem_len = len(tokenizer(user_problem)['input_ids'])
                context_str = cut_seq(seq=doc_str+code_str, l=max_context_length-problem_len)
                context_str = context_str['string_after_cut']
                if len(doc_str)>0:
                    context_str = 'Documents:\n'+context_str
                call_tool_argument = {
                    'tool': tool_name,
                    'model': expert_model_to_call,
                    'context_str': context_str,
                    'vllm_model_configs': vllm_model_configs,
                    'cur_output_dir': cur_output_dir,
                    'problem': user_problem,
                    'id': e['id'],
                    'eid': e['eid']
                }
            elif tool_call['name']=='answer':
                # Qwen3-32B-FP8 has 32K context, leave room for output
                max_code_length = 10000
                max_context_length = 24000
                doc_str = ''
                for doc_idx, doc in enumerate(doc_list):
                    doc_str += f"Doc {doc_idx+1}: {doc}\n\n"
                code_str = ''
                for code_idx, code_piece in enumerate(code_list):
                    code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                str_cut = cut_seq(seq=code_str, l=max_code_length)
                code_str = str_cut['string_after_cut']
                if not code_str.startswith('```') and len(code_str)>0:
                    code_str = '```\n'+code_str
                problem_len = len(tokenizer(user_problem)['input_ids'])
                context_str = cut_seq(seq=doc_str+code_str, l=max_context_length-problem_len)
                context_str = context_str['string_after_cut']
                if len(doc_str)>0:
                    context_str = 'Documents:\n'+context_str
                call_tool_argument = {
                    'tool': tool_name,
                    'model': expert_model_to_call,
                    'context_str': context_str,
                    'vllm_model_configs': vllm_model_configs,
                    'cur_output_dir': cur_output_dir,
                    'problem': user_problem,
                    'answer': answer,
                    'id': e['id'],
                    'eid': e['eid']
                }
            elif tool_call['name'] in ['search']:
                doc_str = ''
                for doc_idx, doc in enumerate(doc_list):
                    doc_str += f"Doc {doc_idx+1}: {doc}\n\n"
                code_str = ''
                for code_idx, code_piece in enumerate(code_list):
                    code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                str_cut = cut_seq(seq=code_str, l=max_code_length)
                code_str = str_cut['string_after_cut']
                if not code_str.startswith('```') and len(code_str)>0:
                    code_str = '```\n'+code_str
                problem_len = len(tokenizer(user_problem)['input_ids'])
                context_str = cut_seq(seq=doc_str+code_str, l=max_context_length-problem_len)
                context_str = context_str['string_after_cut']
                if len(doc_str)>0:
                    context_str = 'Documents:\n'+context_str
                call_tool_argument = {
                    'tool': tool_name,
                    'model': expert_model_to_call,
                    'context_str': context_str,
                    'vllm_model_configs': vllm_model_configs,
                    'cur_output_dir': cur_output_dir,
                    'problem': user_problem,
                    'answer': answer,
                    'id': e['id'],
                    'eid': e['eid']
                }
            tool_call_list.append([call_tool, call_tool_argument])
            if tool_call['name']=='answer':
                break
            break
        all_tool_calls.append(cur_tool_calls)

        cache_argument = []
        for t in tool_call_list:
            cache_argument.append(t[1])
        if len(tool_call_list)==0:
            continue
        cur_responses = asyncio.run(run_all(tool_call_list))
        all_tool_responses[f"turn_{step}_response"] = cur_responses
        finish_flag = False
        for cur_response in cur_responses:
            if isinstance(cur_response, dict) and cur_response.get("tool_error"):
                task_status = "error"
                task_error_type = cur_response.get("tool_error_type") or "tool_error"
                finish_flag = True
                break
            if cur_response['tool']=='enhance_reasoning':
                if len(cur_response['exec_result'].strip())>0:
                    code_list.append({'code': cur_response['generated_code'], 'output': cur_response['exec_result']})
            elif cur_response['tool']=='answer':
                final_correct = cur_response['correctness']
                final_answer_model = cur_response['model']
                final_pred = cur_response['pred'].strip()
                finish_flag = True
                break
            elif cur_response['tool']=='search':
                for one_doc in cur_response['search_results_data'][::-1]:
                    if not one_doc in doc_list:
                        doc_list.append(one_doc)
        if finish_flag:
            break

    return_dict = {
        'id': e['id'],
        'all_tool_calls': all_tool_calls,
        'all_tool_responses': all_tool_responses,
        'correct': final_correct,
        'status': task_status,
    }
    if task_error_type:
        return_dict["error_type"] = task_error_type
    with open(os.path.join(my_output_dir, f"{e['id']}.json"), 'w') as f:
        json.dump(return_dict, f, indent=2)

    if task_error_type:
        print(f"[FRAMES_TASK_COMPLETE] id={e['id']} status={task_status} correct={final_correct} error_type={task_error_type}", flush=True)
    else:
        print(f"[FRAMES_TASK_COMPLETE] id={e['id']} status={task_status} correct={final_correct}", flush=True)
    return return_dict

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--example_file_path', type=str, default='frames.jsonl')
    parser.add_argument('--max_rounds', type=int, default=50)
    parser.add_argument('--model_type', type=str, default='Qwen/Qwen3-8B')
    parser.add_argument('--concurrency', type=int, default=int(os.getenv("FRAMES_CONCURRENCY", "2")))
    parser.add_argument('--basic_tools', action='store_true')
    args = parser.parse_args()

    if args.basic_tools:
        keys = list(MODEL_MAPPING.keys())
        for k in keys:
            MODEL_MAPPING[k] = args.model_name

    # global MODEL_NAME
    MODEL_NAME = args.model_name
    # global MODEL_TYPE
    MODEL_TYPE = args.model_type
    # global my_output_dir
    my_output_dir = args.output_dir
    # global MAX_ROUNDS
    MAX_ROUNDS = args.max_rounds
    if not os.path.isdir(os.path.join(my_output_dir, 'answer_cache')):
        os.makedirs(os.path.join(my_output_dir, 'answer_cache'))
    # global vllm_model_configs
    with open(args.model_config) as f:
        vllm_model_configs = json.load(f)
    with open(args.example_file_path) as f:
        lines = f.readlines()
    examples = []
    for eid, l in enumerate(lines):
        raw_example = json.loads(l)
        raw_example['eid'] = eid
        examples.append([run_single, raw_example])

    # Never abort the whole run on a single task exception.
    tool_call_results = asyncio.run(run_all(examples, concurrency=args.concurrency, return_exceptions=True))
