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

MODEL_MAPPING = {
    'search': {
        "gpt-5": "search-1",
        "gpt-5-mini": "search-2",
        "Qwen/Qwen3-32B": "search-3"
    },
    'enhance_reasoning': {
        "gpt-5": "reasoner-1",
        "gpt-5-mini": "reasoner-2",
        "Qwen/Qwen2.5-Coder-32B-Instruct": "reasoner-3"
    },
    'answer': {
        "Qwen/Qwen2.5-Math-72B-Instruct": "answer-math-1",
        "Qwen/Qwen2.5-Math-7B-Instruct": "answer-math-2",
        "gpt-5": "answer-1",
        "gpt-5-mini": "answer-2",
        "meta-llama/Llama-3.3-70B-Instruct": "answer-3",
        "Qwen/Qwen3-32B": "answer-4"
    }
}

task_id = '66f5e796acadd55c11fb11f5'
output_path = f'example.json'
output_dir = 'sft_data'
with open('evaluation/hle.jsonl') as f:
    lines = f.readlines()
id2example = {}
for l in lines:
    e = json.loads(l)
    id2example[e['id']] = e
with open(output_path) as f:
    results_data = json.load(f)
problem = id2example[task_id]['question']
messages = [
    {"role": "system", "content": "You are good at using tools.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"code_interpreter\", \"description\": \"python executor to execute code and return outputs\", \"parameters\": {\"properties\": {\"code\": {\"description\": \"The code to execute\", \"type\": \"string\"}}, \"required\": [\"code\"], \"title\": \"parameters\", \"type\": \"object\"}}}\n{\"type\": \"function\", \"function\": {\"name\": \"search\", \"description\": \"Search for missing information\", \"parameters\": {\"properties\": {\"query\": {\"description\": \"The query used to search missing information\", \"type\": \"string\"}}, \"required\": [\"query\"], \"title\": \"parameters\", \"type\": \"object\"}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"},
    {"role": "user", "content": f"Problem: {problem}"}
]
documents = []
for i in range(100):
    if not f"turn_{i}_response" in results_data["tool_responses"]:
        continue
    model_response = results_data["all_tool_calls"][i][0][1]
    cur_dict = results_data["tool_responses"][f"turn_{i}_response"][0]
    
    if "search_results_data" in cur_dict:
        query = cur_dict['query']
        context = ''
        for d in cur_dict["search_results_data"]:
            if not d in documents:
                documents.append(d)
                context += d+'\n\n'
        tool_call_content = '<tool_call>{"name": "search", "arguments": {"query": "QUERY_TO_REPLACE"}}</tool_call>'
        tool_call_content = tool_call_content.replace("QUERY_TO_REPLACE",query)
        messages.append({
            'role': 'assistant',
            'content': model_response+tool_call_content
        })
        messages.append({
            'role': 'user',
            'content': "Search results:\n"+context
        })
    elif "generated_code" in cur_dict and "exec_result" in cur_dict:
        tool_call_content = '<tool_call>{"name": "code_interpreter", "arguments": {"query": "CODE_TO_REPLACE"}}</tool_call>'
        tool_call_content = tool_call_content.replace("CODE_TO_REPLACE",cur_dict["generated_code"])
        messages.append({
            'role': 'assistant',
            'content': model_response+tool_call_content
        })
        messages.append({
            'role': 'user',
            'content': 'Execution results:\n'+cur_dict["exec_result"]
        })
    elif 'answer_response' in cur_dict:
        messages.append({
            'role': 'assistant',
            'content': model_response+cur_dict['answer_response']
        })

data_idx = 0
if not os.path.isdir(output_dir):
    os.makedirs(output_dir,exist_ok=True)
for i in range(3,len(messages)+1,2):
    data_idx += 1
    with open(os.path.join(output_dir,f"{data_idx}.json"),'w') as f:
        json.dump(messages[:i],f,indent=2)