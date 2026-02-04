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
from typing import List, Optional
import argparse
import time
import threading
from concurrent.futures import Future
from queue import Empty, Queue

import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}

def load_corpus(corpus_path: str):
    num_proc = int(os.environ.get("WIKI_CORPUS_NUM_PROC", "16"))
    cache_dir = os.environ.get("WIKI_HF_CACHE_DIR", "cache/huggingface")
    keep_in_memory = _env_flag("WIKI_CORPUS_KEEP_IN_MEMORY", "0")

    load_kwargs = dict(
        data_files=corpus_path,
        split="train",
        num_proc=num_proc,
        cache_dir=cache_dir,
    )
    if keep_in_memory:
        load_kwargs["keep_in_memory"] = True

    try:
        corpus = datasets.load_dataset("json", **load_kwargs)
    except TypeError as e:
        # Backward compatibility for older `datasets` versions.
        if "keep_in_memory" in load_kwargs and "keep_in_memory" in str(e):
            print(
                "[retrieval_wiki] WARNING: datasets.load_dataset() does not support keep_in_memory; "
                "retrying without it.",
                flush=True,
            )
            load_kwargs.pop("keep_in_memory", None)
            corpus = datasets.load_dataset("json", **load_kwargs)
        else:
            raise
    return corpus

def last_token_pool(last_hidden_states,attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_docs(corpus, doc_idxs):
    # `doc_idxs` can be a numpy array; FAISS uses -1 for missing entries.
    results = []
    for idx in doc_idxs:
        ii = int(idx)
        if ii < 0:
            continue
        results.append(corpus[ii])
    return results


def load_model(model_path: str, use_fp16: bool = False):
    if model_path in ['Qwen/Qwen3-Embedding-8B']:
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        # Prefer FlashAttention2 when available, but fall back gracefully if the
        # environment doesn't have `flash_attn` installed.
        try:
            model = AutoModel.from_pretrained(
                model_path,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.float16,
            ).cuda()
        except Exception as e:
            print(
                f"[retrieval_wiki] WARNING: flash_attention_2 unavailable ({type(e).__name__}: {e}); "
                "falling back to SDPA/eager attention.",
                flush=True,
            )
            try:
                model = AutoModel.from_pretrained(
                    model_path,
                    attn_implementation="sdpa",
                    torch_dtype=torch.float16,
                ).cuda()
            except Exception:
                model = AutoModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                ).cuda()
    else:
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        model.cuda()
        if use_fp16:
            model = model.half()
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


def pooling(
        pooler_output,
        last_hidden_state,
        attention_mask=None,
        pooling_method="mean"
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in
                              query_list]

        if 'qwen' in self.model_name.lower():
            if is_query:
                query_list = [f'Instruct: Given a search query, retrieve relevant passages\nQuery:{query}' for query in query_list]

        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                                )
        # inputs = {k: v.cuda() for k, v in inputs.items()}
        inputs.to(self.model.device)

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]
        elif 'qwen' in self.model_name.lower():
            output = self.model(**inputs)
            embeddings = last_token_pool(output.last_hidden_state, inputs['attention_mask'])
            query_emb = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(None,
                                output.last_hidden_state,
                                inputs['attention_mask'],
                                self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
        # print(144,'query_emb.shape',query_emb.shape)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")

        del inputs, output
        # `torch.cuda.empty_cache()` can add significant latency per request; keep it off by default.
        if _env_flag("WIKI_TORCH_EMPTY_CACHE", "0"):
            torch.cuda.empty_cache()

        return query_emb


class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk

        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)


class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            ngpu = faiss.get_num_gpus()
            min_gpus = int(os.environ.get("WIKI_FAISS_GPU_MIN_GPUS", "2"))
            if ngpu < min_gpus:
                print(
                    f"[retrieval_wiki] WARNING: WIKI_FAISS_GPU requested but only {ngpu} GPU(s) visible "
                    f"(<{min_gpus}); keeping CPU FAISS index.",
                    flush=True,
                )
            else:
                co = faiss.GpuMultipleClonerOptions()
                co.useFloat16 = True
                co.shard = True
                print(
                    f"[retrieval_wiki] Moving FAISS index to GPU(s): ngpu={ngpu} shard={co.shard} fp16={co.useFloat16}",
                    flush=True,
                )
                self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        self.corpus = load_corpus(self.corpus_path)
        if _env_flag("WIKI_PRELOAD_CORPUS", "1"):
            print("[retrieval_wiki] Preloading corpus into memory ...", flush=True)
            preload_t0 = time.perf_counter()
            total = None
            try:
                total = len(self.corpus)
            except Exception:
                total = None
            corpus_cache = []
            # Iteration is sequential and much faster than random access via `corpus[idx]`.
            for doc in tqdm(self.corpus, desc="Loading corpus", total=total):
                corpus_cache.append(doc)
            self.corpus = corpus_cache
            print(
                f"[retrieval_wiki] Corpus preloaded: {len(self.corpus)} docs in "
                f"{time.perf_counter() - preload_t0:.1f}s",
                flush=True,
            )

        self.encoder = Encoder(
            model_name=self.retrieval_method,
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores.tolist()
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        results = []
        scores = []
        batch_iter = range(0, len(query_list), self.batch_size)
        if _env_flag("WIKI_RETRIEVAL_TQDM", "0"):
            batch_iter = tqdm(batch_iter, desc="Retrieval process")
        for start_idx in batch_iter:
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            # load_docs is not vectorized, but is a python list approach
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            # chunk them back
            batch_results = [batch_results[i * num: (i + 1) * num] for i in range(len(batch_idxs))]

            results.extend(batch_results)
            scores.extend(batch_scores)

            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            # `torch.cuda.empty_cache()` can add significant latency per request; keep it off by default.
            if _env_flag("WIKI_TORCH_EMPTY_CACHE", "0"):
                torch.cuda.empty_cache()

        if return_score:
            return results, scores
        else:
            return results


#####################################
# FastAPI server below
#####################################

class RequestBatcher:
    """
    Server-side request batching to amortize embedding encode cost across concurrent /retrieve calls.

    Enabled with:
      WIKI_REQUEST_BATCHING=1

    Tunables:
      WIKI_BATCH_TIMEOUT_S (default 0.05): how long to wait to form a batch after the first request arrives
      WIKI_MAX_BATCH_REQUESTS (default 16): max number of HTTP requests per batch
      WIKI_MAX_BATCH_QUERIES (default 16): max total queries per batch across all requests
    """

    def __init__(
        self,
        retriever: DenseRetriever,
        batch_timeout_s: float = 0.05,
        max_batch_requests: int = 16,
        max_batch_queries: int = 16,
    ):
        self.retriever = retriever
        self.batch_timeout_s = float(batch_timeout_s)
        self.max_batch_requests = int(max_batch_requests)
        self.max_batch_queries = int(max_batch_queries)
        self._q: "Queue[dict]" = Queue()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def submit(self, queries: List[str], num: int):
        fut: Future = Future()
        self._q.put({"queries": list(queries), "num": int(num), "future": fut})
        return fut

    def _loop(self):
        while True:
            first = self._q.get()
            batch = [first]
            total_queries = len(first.get("queries") or [])
            deadline = time.monotonic() + self.batch_timeout_s

            while len(batch) < self.max_batch_requests and total_queries < self.max_batch_queries:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    nxt = self._q.get(timeout=timeout)
                except Empty:
                    break
                batch.append(nxt)
                total_queries += len(nxt.get("queries") or [])

            # Partition by `num` so we never mix different K in a single FAISS call.
            by_num: dict[int, list[dict]] = {}
            for item in batch:
                by_num.setdefault(int(item["num"]), []).append(item)

            for num, items in by_num.items():
                flat_queries: List[str] = []
                spans: List[tuple[int, int, dict]] = []
                for item in items:
                    q = item.get("queries") or []
                    start = len(flat_queries)
                    flat_queries.extend(q)
                    spans.append((start, len(flat_queries), item))

                if not flat_queries:
                    for _, _, item in spans:
                        item["future"].set_result(([], []))
                    continue

                try:
                    results, scores = self.retriever.batch_search(flat_queries, num=num, return_score=True)
                    for start, end, item in spans:
                        item["future"].set_result((results[start:end], scores[start:end]))
                except Exception as e:
                    for _, _, item in spans:
                        item["future"].set_exception(e)

class Config:
    """
    Minimal config class (simulating your argparse)
    Replace this with your real arguments or load them dynamically.
    """

    def __init__(
            self,
            retrieval_method: str = "bm25",
            retrieval_topk: int = 10,
            index_path: str = "./index/bm25",
            corpus_path: str = "./data/corpus.jsonl",
            dataset_path: str = "./data",
            data_split: str = "train",
            faiss_gpu: bool = True,
            retrieval_model_path: str = "./model",
            retrieval_pooling_method: str = "mean",
            retrieval_query_max_length: int = 32768,
            retrieval_use_fp16: bool = False,
            retrieval_batch_size: int = 128
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size


# class QueryRequest(BaseModel):
#     queries: List[str]
#     topk: Optional[int] = None
#     return_scores: bool = False

class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False
    eid: str = None
    new_cache_dir: str = None


app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    if not request.topk:
        request.topk = config.retrieval_topk  # fallback to default

    internal_k = int(os.environ.get("WIKI_RETRIEVAL_INTERNAL_TOPK", "1000"))

    # Perform batch retrieval - always get scores to avoid unpacking issues
    if request_batcher is not None:
        fut = request_batcher.submit(queries=request.queries, num=internal_k)
        results, scores = fut.result()
    else:
        results, scores = retriever.batch_search(
            query_list=request.queries,
            num=internal_k,
            return_score=True,
        )

    # Format response
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            # If scores are returned, combine them with results
            combined = []
            for doc, score in zip(single_result, scores[i]):
                if len(doc["contents"])>100:
                    combined.append({"document": doc, "score": score})
                if len(combined)>=request.topk:
                    break
            resp.append(combined)
        else:
            resp.append(single_result)
    return resp


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int)
args = parser.parse_args()

config = Config(
    retrieval_method='qwen',  # or "dense"
    index_path=os.path.join(os.environ.get('INDEX_DIR',None),'wiki.index'),
    corpus_path=os.path.join(os.environ.get('INDEX_DIR',None),'wiki.jsonl'),
    retrieval_topk=3,
    # wiki.index is extremely large (~100GB). Moving it to GPU requires massive GPU memory
    # and will OOM on typical single-GPU jobs. Default to CPU FAISS.
    faiss_gpu=os.environ.get("WIKI_FAISS_GPU", "0").strip().lower() in {"1", "true", "yes"},
    retrieval_model_path='Qwen/Qwen3-Embedding-8B',
    retrieval_pooling_method="mean",
    retrieval_query_max_length=32768,
    retrieval_use_fp16=True,
    retrieval_batch_size=512,
)

retriever = DenseRetriever(config)
request_batcher = None
if _env_flag("WIKI_REQUEST_BATCHING", "0"):
    request_batcher = RequestBatcher(
        retriever,
        batch_timeout_s=float(os.environ.get("WIKI_BATCH_TIMEOUT_S", "0.05")),
        max_batch_requests=int(os.environ.get("WIKI_MAX_BATCH_REQUESTS", "16")),
        max_batch_queries=int(os.environ.get("WIKI_MAX_BATCH_QUERIES", "16")),
    )
    print(
        "[retrieval_wiki] Request batching enabled: "
        f"timeout={request_batcher.batch_timeout_s}s "
        f"max_requests={request_batcher.max_batch_requests} "
        f"max_queries={request_batcher.max_batch_queries}",
        flush=True,
    )

uvicorn.run(app, host="0.0.0.0", port=args.port)

