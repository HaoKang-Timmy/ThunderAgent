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
import random
import time
import json
import requests
import asyncio
import subprocess
from tqdm import tqdm
from transformers import AutoTokenizer
import sys
REPO_PATH = os.getenv("REPO_PATH")
sys.path.append(REPO_PATH)
from LLM_CALL import get_llm_response
import multiprocessing as mp
import argparse
import logging
from openai import OpenAI
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
# Provide your api key
oss_client = OpenAI(
  base_url = "https://api.together.xyz/v1",
  api_key = os.getenv("TOGETHER_API_KEY")
)

MODEL_MAPPING = {
    "search-1": "gpt-5",
    "search-2": "gpt-5-mini",
    "search-3": "Qwen/Qwen3-32B",
    "reasoner-1": "gpt-5",
    "reasoner-2": "gpt-5-mini",
    "reasoner-3": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "answer-math-1": "Qwen/Qwen2.5-Math-72B-Instruct",
    "answer-math-2": "Qwen/Qwen2.5-Math-7B-Instruct",
    "answer-1": "gpt-5",
    "answer-2": "gpt-5-mini",
    "answer-3": "meta-llama/Llama-3.3-70B-Instruct",
    "answer-4": "Qwen/Qwen3-32B"
}
TOOL_PRICING = {
    "gpt-5": {
        "input_tokens_per_million": 1.25/10000000,
        "output_tokens_per_million": 10/1000000
    },
    "gpt-5-mini": {
        "input_tokens_per_million": 0.25/10000000,
        "output_tokens_per_million": 2/1000000
    },
    "Qwen/Qwen3-32B": {
        "input_tokens_per_million": 0.8/1000000,
        "output_tokens_per_million": 0.8/1000000
    },
    "Qwen/Qwen2.5-Coder-32B-Instruct": {
        "input_tokens_per_million": 0.8/1000000,
        "output_tokens_per_million": 0.8/1000000
    },
    "Qwen/Qwen2.5-Math-72B-Instruct": {
        "input_tokens_per_million": 0.9/1000000,
        "output_tokens_per_million": 0.9/1000000
    },
    "Qwen/Qwen2.5-Math-7B-Instruct": {
        "input_tokens_per_million": 0.2/1000000,
        "output_tokens_per_million": 0.2/1000000
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "input_tokens_per_million": 0.9/1000000,
        "output_tokens_per_million": 0.9/1000000
    },
    "Qwen/Qwen3-8B": {
        "input_tokens_per_million": 0.2/1000000,
        "output_tokens_per_million": 0.2/1000000
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
    start_time = time.time()
    if arguments['tool']=='enhance_reasoning':
        supported_models = [MODEL_MAPPING[m] for m in ALL_TOOLS['enhance_reasoning']['model']]
        assert arguments['model'] in supported_models,f"Model {arguments['model']} is not supported in enhance_reasoning. Support models: {supported_models}"
        prompt = arguments['context_str'].strip()+'\n\n'
        prompt += f"Question: {arguments['problem']}\nInstead of directly answering the question, please write additional python code that will give intermidiate results after execution. Wrap the code within ```python and ```. The code should be self-contained with all the import and initialization."
        model_name = arguments['model']
        response = ''
        if 'gpt-5' in model_name.lower():
            response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,temperature=1,max_length=40000)
        elif 'qwen2.5-coder' in model_name.lower():
            response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
            if isinstance(response,str):
                response = ''
                while not response:
                    try:
                        response = oss_client.chat.completions.create(
                            model="Qwen/Qwen2.5-Coder-32B-Instruct", 
                            messages=[{"role":"user","content":prompt}],temperature=0.2,
                            top_p=0.7,
                            max_tokens=8000,
                        )
                    except Exception as qwen_error:
                        time.sleep(3)
        if isinstance(response,str):
            arguments['generated_code'] = ''
            arguments['exec_result'] = ''
            return arguments
        try:
            generated_code = response.choices[0].message.content.split('```python')[-1].split('```')[0]
        except:
            generated_code = ''
        if generated_code=='':
            arguments['generated_code'] = ''
            arguments['exec_result'] = ''
            return arguments
        code_path = str(os.path.join(arguments['cur_output_dir'],f'exec_code_{arguments["id"]}.py'))
        with open(code_path,'w') as f:
            f.write(generated_code)
        exec_result = ''
        exec_start = time.time()
        try:
            exec_result = subprocess.run(['python', code_path], timeout=60, capture_output=True, text=True)
            exec_time = time.time()-exec_start
            exec_result = exec_result.stdout
            with open(os.path.join(arguments['cur_output_dir'],f'exec_out_{arguments["id"]}.txt'),'w') as f:
                f.write(exec_result)
        except Exception as e:
            pass
        exec_time = time.time() - exec_start
        arguments['generated_code'] = generated_code
        arguments['exec_result'] = exec_result
        return arguments
    
    elif arguments['tool']=='answer':
        prompt = arguments['context_str'].strip()+'\n\nProblem:\n'+arguments['problem']
        response_str = ''
        pred = ''

        if 'qwen3' in arguments['model'].lower():
            model_name = arguments['model']
            messages = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": prompt}
            ]
            arguments['messages'] = messages
            response = get_llm_response(model=model_name,messages=messages,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
            if isinstance(response,str):
                arguments['response'] = ''
                arguments['pred'] = ''
                arguments['correctness'] = False
                return arguments
            response_str = response.choices[0].message.content
            if not isinstance(response_str,str) or not '\\boxed{' in response_str:
                pred = ''
            else:
                pred_components = response.choices[0].message.content.split('\\boxed{')[-1].split('}')[:-1]
                pred = '}'.join(pred_components).strip()
        elif 'qwen2.5-math' in arguments['model'].lower():
            model_name = arguments['model']
            messages = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": prompt}
            ]
            arguments['messages'] = messages
            response = get_llm_response(model=model_name,messages=messages,return_raw_response=True,model_type='vllm',max_length=2000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
            if isinstance(response,str):
                arguments['response'] = ''
                arguments['pred'] = ''
                arguments['correctness'] = False
                return arguments
            response_str = response.choices[0].message.content
            if not isinstance(response_str,str) or not '\\boxed{' in response_str:
                pred = ''
            else:
                pred_components = response.choices[0].message.content.split('\\boxed{')[-1].split('}')[:-1]
                pred = '}'.join(pred_components).strip()
        elif 'gpt-5' in arguments['model'].lower():
            model_name = arguments['model']
            prompt += ("\n\nTake a deep breath and think hard with high reasoning, wrap the thoughts within <think> and </think>, and wrap only the exact answer without any explanation within <answer> and </answer>."
                        "Output using the following format:\n<think>\n...\n</think>\n<answer>\n...\n</answer>")
            arguments['messages'] = prompt
            response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,max_length=40000)
            if isinstance(response,str):
                arguments['response'] = ''
                arguments['pred'] = ''
                arguments['correctness'] = False
                return arguments
            response_str = response.choices[0].message.content
            if isinstance(response_str,str):
                pred = response.choices[0].message.content.split('<answer>')[-1].split('</answer>')[0].strip()
            else:
                pred = ''
        elif 'llama-3.3' in arguments['model'].lower():
            model_name = arguments['model']
            prompt += "\nWrap the thinking process and explanation between <think> and </think> and wrap only the exact answer without any explanation within <answer> and </answer>."
            arguments['messages'] = prompt
            response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,model_type='vllm',max_length=40000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
            if isinstance(response,str):
                response = ''
                while not response:
                    try:
                        response = client.chat.completions.create(
                            model="meta-llama/Llama-3.3-70B-Instruct", 
                            messages=[{"role":"user","content":prompt}],temperature=0.2,
                            top_p=0.7,
                            max_tokens=40000,
                        )
                    except Exception as llama_error:
                        time.sleep(3)
                if isinstance(response,str):
                    arguments['response'] = ''
                    arguments['pred'] = ''
                    arguments['correctness'] = False
                    return arguments
            response_str = response.choices[0].message.content
            if isinstance(response_str,str):
                pred = response.choices[0].message.content.split('<answer>')[-1].split('</answer>')[0].strip()
            else:
                pred = ''
        
        if pred.strip()=='' or len(pred.split(' '))>500:
            correctness = False
        elif pred.strip().lower()==arguments['answer'].strip().lower():
            correctness = True
        else:
            eval_prompt = (f"Question: {arguments['problem']}\n\n"
                        f"Student answer: {pred}\n\n"
                        f"Reference answer: {arguments['answer']}\n\n"
                        "Assume that the reference answer is correct. Output <correct>True</correct> if the student answer matches the reference answer. Output <correct>False</correct> if the student answer does not match the reference answer.")
            eval_response = get_llm_response(model='gpt-5',messages=eval_prompt,temperature=1)
            eval_result = eval_response.split('<correct>')[-1].split('</correct>')[0]
            if eval_result.lower()=='true':
                correctness = True
            else:
                correctness = False
        arguments['response'] = response_str
        arguments['pred'] = pred
        arguments['correctness'] = correctness
        return arguments

    elif arguments['tool']=='search':
        contents = []
        prompt = arguments['context_str'].strip()+'\n\n'
        prompt += f"Question: {arguments['problem']}\nInstead of directly answering the question, please write a query to search for a piece of relevant and missing information. The query should be a few key words about the information to search or a short sentence. Wrap the query within <query> and </query>."
        cur_query_writer = arguments['model']
        query_to_call = None
        if 'gpt-5' in cur_query_writer.lower():
            response = get_llm_response(model=cur_query_writer,messages=prompt,return_raw_response=True,temperature=1,max_length=40000)
            if isinstance(response,str) or not response:
                query_to_call = arguments['problem']
            else:
                query_to_call = response.choices[0].message.content.split('<query>')[-1].split('</query>')[0]
        elif 'qwen3' in cur_query_writer.lower():
            response = get_llm_response(model=cur_query_writer,messages=prompt,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][cur_query_writer],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
            if isinstance(response,str):
                query_to_call = arguments['problem']
            else:
                query_to_call = response.choices[0].message.content.split('<query>')[-1].split('</query>')[0]
        if query_to_call is None or len(query_to_call)<5:
            pass
        else:
            assert len(query_to_call)>5,f"{query_to_call}"
            payload = {
                "queries": [query_to_call[:390]],
                "topk": 50,
                "return_scores": True,
                "eid": arguments['id']
            }
            results = None
            all_vllm_model_configs = arguments['vllm_model_configs']
            search_try_count = 0
            while not results:
                search_try_count += 1
                try:
                    cur_model_config = random.choice(all_vllm_model_configs['retrieval'])
                    results = requests.post(f'http://{cur_model_config["ip_addr"]}:{cur_model_config["port"]}/retrieve', json=payload).json()
                except Exception as search_error:
                    time.sleep(3)
            if results:
                for r in results[0]:
                    if 'content' in r['document']:
                        contents.append(r['document']['content'])
                    elif 'contents' in r['document']:
                        contents.append(r['document']['contents'])
        arguments['query'] = query_to_call
        arguments['search_results_data'] = contents
        if 'tokenizer' in arguments:
            arguments.pop('tokenizer')
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

    # create the executor sized to your concurrency gate
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # wrap each task so it obeys the semaphore
        async def run_one(idx: int, func: Callable, arg: Any):
            async with sem:
                if asyncio.iscoroutinefunction(func):
                    res = await func(arg)
                else:
                    res = await loop.run_in_executor(executor, func, arg)
                return idx, res, None

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
            # update progress as tasks complete
            for fut in asyncio.as_completed(tasks):
                idx, res, err = await fut
                if err is None:
                    results[idx] = res
                else:
                    if return_exceptions:
                        results[idx] = err
                    else:
                        # cancel remaining, then re-raise the first error
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
    if os.path.isfile(os.path.join(my_output_dir,f"{e['id']}.json")):
        return
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
    all_message_responses = {}
    used_tools = []
    for step in range(MAX_ROUNDS):
        cur_output_dir = os.path.join(my_output_dir,f"step_{step}")
        if not os.path.isdir(os.path.join(cur_output_dir,'tool_return')):
            try:
                os.makedirs(os.path.join(cur_output_dir,'tool_return'))
            except:
                pass
        tools = []
        for t in raw_tools:
            tools.append(t)
        doc_str = ''
        for doc_idx, doc in enumerate(doc_list):
            doc_str += f"Doc {doc_idx+1}: {doc[:1200]} ...\n\n"
        code_str = ''
        for code_idx, code_piece in enumerate(code_list):
            code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
        attempt_str = ''
        for attempt_idx, attempt in enumerate(attempt_list):
            attempt_str += f"Attempt{attempt_idx+1} answer by {attempt['model']}: {attempt['answer']}\n"
        str_cut = cut_seq(seq=attempt_str,l=8000)
        attempt_str = str_cut['string_after_cut']
        if not attempt_str.startswith('Attempt') and len(attempt_str)>0:
            attempt_str = 'Attempt answer: '+attempt_str
        str_cut = cut_seq(seq=code_str+attempt_str,l=12000)
        code_attempt_str = str_cut['string_after_cut']
        code_attempt_str_len = str_cut['effective_length']
        if not code_attempt_str.startswith('```') and len(code_attempt_str)>0:
            code_attempt_str = '```\n'+code_attempt_str
        doc_flag = False
        problem_length = len(tokenizer(problem)['input_ids'])
        if code_attempt_str_len<27000-problem_length:
            if code_attempt_str:
                context_str = cut_seq(seq=doc_str+"\npython code and execution outputs:\n"+code_attempt_str,l=27000-problem_length)
            else:
                context_str = cut_seq(seq=doc_str,l=27000-problem_length)
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
                    {"role": "user", "content": f"Problem: {problem}\n\n{context_str}\n\nChoose an appropriate tool.'"}
                ]
        response = get_llm_response(model=MODEL_NAME,messages=chat,return_raw_response=True,model_type='vllm',model_config=vllm_model_configs[MODEL_NAME],temperature=1,max_length=12000,tools=tools,model_config_path=vllm_model_configs['vllm_model_config_path'],model_config_idx=e['eid'])
        cache_idx = 0
        while os.path.isfile(f"input_output/{cache_idx}.json"):
            cache_idx += 1
        if isinstance(response,str):
            continue
        tool_calls = response.choices[0].message.tool_calls
        cache_tool_calls = []
        for one_tool_call in tool_calls:
            tool_name = one_tool_call.function.name
            try:
                tool_arguments = json.loads(one_tool_call.function.arguments)
            except:
                pass
            cache_tool_calls.append({
                'tool_name': tool_name,
                'tool_arguments': tool_arguments
            })
        message_dict = {
            'content': response.choices[0].message.content,
            'tool_calls': cache_tool_calls
        }
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
                pass
            if not tool_name in ALL_TOOLS:
                cur_tool_calls.append(f'350 invalid tool calls {tool_calls}')
                continue
            func_signature = ALL_TOOLS[tool_name]
            valid_tool_call = True
            for parameter_name,parameter_values in func_signature.items():
                if (not parameter_name in tool_arguments):
                    valid_tool_call = False
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
            cur_tool_calls.append([tool_call])
            expert_model_to_call = MODEL_MAPPING[tool_arguments['model']]
            
            call_tool_argument = None
            used_tools.append(tool_name)
            if tool_name=='enhance_reasoning':
                if 'qwen2.5-coder' in expert_model_to_call.lower():
                    max_code_length = 16000
                    max_context_length = 24000
                elif 'gpt-5' in expert_model_to_call.lower():
                    max_code_length = 40000
                    max_context_length = 120000
                doc_str = ''
                for doc_idx, doc in enumerate(doc_list):
                    if 'qwen2.5-coder' in expert_model_to_call.lower():
                        doc_str += f"Doc {doc_idx+1}: {doc[:1000]}\n\n"
                    else:
                        doc_str += f"Doc {doc_idx+1}: {doc}\n\n"
                code_str = ''
                for code_idx, code_piece in enumerate(code_list):
                    code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                str_cut = cut_seq(seq=code_str,l=max_code_length)
                code_str = str_cut['string_after_cut']
                code_str_len = str_cut['effective_length']
                if not code_str.startswith('```') and len(code_str)>0:
                    code_str = '```\n'+code_str
                problem_len = len(tokenizer(user_problem)['input_ids'])
                context_str = cut_seq(seq=doc_str+code_str,l=max_context_length-problem_len)
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
                if 'qwen2.5-math' in expert_model_to_call.lower():
                    max_code_length = 1000
                    max_context_length = 2000
                elif 'llama-3.3' in expert_model_to_call.lower():
                    max_code_length = 10000
                    max_context_length = 80000
                elif 'qwen3' in expert_model_to_call.lower():
                    max_code_length = 12000
                    max_context_length = 24000
                elif 'gpt-5' in expert_model_to_call.lower():
                    max_code_length = 40000
                    max_context_length = 120000
                doc_str = ''
                for doc_idx, doc in enumerate(doc_list):
                    if 'gpt-5' in expert_model_to_call.lower() or 'llama' in expert_model_to_call.lower():
                        doc_str += f"Doc {doc_idx+1}: {doc}\n\n"
                    else:
                        doc_str += f"Doc {doc_idx+1}: {doc[:1000]}\n\n"
                code_str = ''
                for code_idx, code_piece in enumerate(code_list):
                    code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                str_cut = cut_seq(seq=code_str,l=max_code_length)
                code_str = str_cut['string_after_cut']
                code_str_len = str_cut['effective_length']
                if not code_str.startswith('```') and len(code_str)>0:
                    code_str = '```\n'+code_str
                problem_len = len(tokenizer(user_problem)['input_ids'])
                context_str = cut_seq(seq=doc_str+code_str,l=max_context_length-problem_len)
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
                if 'qwen3' in expert_model_to_call.lower():
                    max_code_length = 12000
                    max_context_length = 24000
                elif 'gpt-5' in expert_model_to_call.lower():
                    max_code_length = 40000
                    max_context_length = 120000
                doc_str = ''
                for doc_idx, doc in enumerate(doc_list):
                    if 'gpt-5' in expert_model_to_call.lower():
                        doc_str += f"Doc {doc_idx+1}: {doc}\n\n"
                    else:
                        doc_str += f"Doc {doc_idx+1}: {doc[:1000]}\n\n"
                code_str = ''
                for code_idx, code_piece in enumerate(code_list):
                    code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                str_cut = cut_seq(seq=code_str,l=max_code_length)
                code_str = str_cut['string_after_cut']
                code_str_len = str_cut['effective_length']
                if not code_str.startswith('```') and len(code_str)>0:
                    code_str = '```\n'+code_str
                problem_len = len(tokenizer(user_problem)['input_ids'])
                context_str = cut_seq(seq=doc_str+code_str,l=max_context_length-problem_len)
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
            tool_call_list.append([call_tool,call_tool_argument])
            break
        all_tool_calls.append(cur_tool_calls)

        cache_argument = []
        for t in tool_call_list:
            cache_argument.append(t[1])
        if len(tool_call_list)==0:
            continue
        cur_responses = asyncio.run(run_all(tool_call_list))
        all_tool_responses[f"turn_{step}_response"] = cur_responses
        all_message_responses[f"turn_{step}_message"] = message_dict
        finish_flag = False
        for cur_response in cur_responses:
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
        'problem': problem,
        'all_tool_calls': all_tool_calls,
        'all_tool_responses': all_tool_responses,
        'answer': answer,
        'all_message_responses': all_message_responses,
        'correct': final_correct
    }
    with open(os.path.join(my_output_dir,f"{e['id']}.json"),'w') as f:
        json.dump(return_dict,f,indent=2)
    return return_dict

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--max_rounds', type=int, default=50)
    parser.add_argument('--model_type', type=str, default='Qwen/Qwen3-8B')
    parser.add_argument('--example_path', type=str)
    args = parser.parse_args()

    # global MODEL_NAME
    MODEL_NAME = args.model_name
    # global MODEL_TYPE
    MODEL_TYPE = args.model_type
    # global my_output_dir
    my_output_dir = args.output_dir
    # global MAX_ROUNDS
    MAX_ROUNDS = args.max_rounds
    if not os.path.isdir(os.path.join(my_output_dir,'answer_cache')):
        os.makedirs(os.path.join(my_output_dir,'answer_cache'))
    # global vllm_model_configs
    with open(args.model_config) as f:
        vllm_model_configs = json.load(f)

    with open(args.example_path) as f:
        lines = f.readlines()
    examples = []
    for eid,l in enumerate(lines):
        raw_example = json.loads(l)
        raw_example['eid'] = eid
        examples.append([run_single,raw_example])

    tool_call_results = asyncio.run(run_all(examples))


    
