"""vLLM metrics parsing and storage."""
from dataclasses import dataclass
from typing import Optional, Tuple
import re
import time


@dataclass
class VLLMCacheConfig:
    """Static KV cache configuration from vLLM (fetched once at startup)."""
    block_size: int = 0          # Tokens per block
    num_gpu_blocks: int = 0      # Total GPU blocks
    
    @property
    def total_tokens_capacity(self) -> int:
        """Total KV cache capacity in tokens."""
        return self.block_size * self.num_gpu_blocks
    
    @classmethod
    def from_prometheus_text(cls, text: str) -> "VLLMCacheConfig":
        """Parse cache_config_info from Prometheus metrics text."""
        config = cls()
        
        # Extract block_size and num_gpu_blocks from labels
        # vllm:cache_config_info{block_size="16",...,num_gpu_blocks="27283",...} 1.0
        match = re.search(r'vllm:cache_config_info\{([^}]+)\}', text)
        if match:
            labels = match.group(1)
            
            # Extract block_size
            bs_match = re.search(r'block_size="(\d+)"', labels)
            if bs_match:
                config.block_size = int(bs_match.group(1))
            
            # Extract num_gpu_blocks
            ngb_match = re.search(r'num_gpu_blocks="(\d+)"', labels)
            if ngb_match:
                config.num_gpu_blocks = int(ngb_match.group(1))
        
        return config


@dataclass
class VLLMMetrics:
    """Parsed metrics from vLLM /metrics endpoint."""
    # Request stats
    num_requests_running: int = 0
    num_requests_waiting: int = 0
    
    # KV Cache
    kv_cache_usage_perc: float = 0.0
    
    # Prefix cache (cumulative)
    prefix_cache_queries: int = 0
    prefix_cache_hits: int = 0
    
    # Tokens (cumulative)
    prompt_tokens_total: int = 0
    generation_tokens_total: int = 0
    
    # Preemptions
    num_preemptions: int = 0
    
    # Request success counts
    request_success_stop: int = 0
    request_success_length: int = 0
    request_success_abort: int = 0
    request_success_error: int = 0
    
    # Timestamp when metrics were fetched
    timestamp: float = 0.0
    
    @property
    def prefix_cache_hit_rate(self) -> float:
        """Calculate prefix cache hit rate."""
        if self.prefix_cache_queries == 0:
            return 0.0
        return self.prefix_cache_hits / self.prefix_cache_queries
    
    @property
    def total_requests_completed(self) -> int:
        """Total completed requests."""
        return (self.request_success_stop + self.request_success_length + 
                self.request_success_abort + self.request_success_error)
    
    @classmethod
    def from_prometheus_text(cls, text: str) -> "VLLMMetrics":
        """Parse Prometheus text format into VLLMMetrics."""
        metrics = cls(timestamp=time.time())
        
        # Helper to extract gauge/counter value
        def extract_value(pattern: str) -> Optional[float]:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, IndexError):
                    return None
            return None
        
        # Current request counts
        val = extract_value(r'vllm:num_requests_running\{[^}]*\}\s+([\d.]+)')
        if val is not None:
            metrics.num_requests_running = int(val)
        
        val = extract_value(r'vllm:num_requests_waiting\{[^}]*\}\s+([\d.]+)')
        if val is not None:
            metrics.num_requests_waiting = int(val)
        
        # KV cache usage
        val = extract_value(r'vllm:kv_cache_usage_perc\{[^}]*\}\s+([\d.]+)')
        if val is not None:
            metrics.kv_cache_usage_perc = val
        
        # Prefix cache (counters)
        val = extract_value(r'vllm:prefix_cache_queries_total\{[^}]*\}\s+([\d.eE+]+)')
        if val is not None:
            metrics.prefix_cache_queries = int(val)
        
        val = extract_value(r'vllm:prefix_cache_hits_total\{[^}]*\}\s+([\d.eE+]+)')
        if val is not None:
            metrics.prefix_cache_hits = int(val)
        
        # Token counts
        val = extract_value(r'vllm:prompt_tokens_total\{[^}]*\}\s+([\d.eE+]+)')
        if val is not None:
            metrics.prompt_tokens_total = int(val)
        
        val = extract_value(r'vllm:generation_tokens_total\{[^}]*\}\s+([\d.eE+]+)')
        if val is not None:
            metrics.generation_tokens_total = int(val)
        
        # Preemptions
        val = extract_value(r'vllm:num_preemptions_total\{[^}]*\}\s+([\d.eE+]+)')
        if val is not None:
            metrics.num_preemptions = int(val)
        
        # Request success counts by finish reason
        for reason in ['stop', 'length', 'abort', 'error']:
            val = extract_value(rf'vllm:request_success_total\{{[^}}]*finished_reason="{reason}"[^}}]*\}}\s+([\d.]+)')
            if val is not None:
                setattr(metrics, f'request_success_{reason}', int(val))
        
        return metrics

