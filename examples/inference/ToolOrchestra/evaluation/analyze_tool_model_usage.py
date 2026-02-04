#!/usr/bin/env python3
"""
Analyze tool call model usage from HLE logs.
Reports how many times each tool (enhance_reasoning, answer, search) called
GPT-5/GPT-5-mini vs open-source models.
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

def extract_tool_model(line: str) -> tuple[str, str] | None:
    """Extract (tool, model) from a log line with type=tool_call."""
    tool_match = re.search(r'tool=(\S+)', line)
    model_match = re.search(r'model=(\S+)', line)
    if tool_match and model_match:
        return (tool_match.group(1), model_match.group(1))
    return None

def is_openai_model(model: str) -> bool:
    """Check if a model is GPT-5/GPT-5-mini."""
    return 'gpt-5' in model.lower()

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_tool_model_usage.py <log_file1> [log_file2] ...")
        return 1
    
    log_files = [Path(p) for p in sys.argv[1:]]
    
    # Stats: tool_name -> {openai: count, opensource: count, models: {model_name: count}}
    stats = defaultdict(lambda: {'openai': 0, 'opensource': 0, 'models': defaultdict(int)})
    
    total_lines = 0
    for log_file in log_files:
        if not log_file.exists():
            print(f"Warning: {log_file} not found, skipping")
            continue
        
        print(f"Reading {log_file}...", file=sys.stderr)
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                total_lines += 1
                if 'type=tool_call' not in line:
                    continue
                
                result = extract_tool_model(line)
                if not result:
                    continue
                
                tool, model = result
                stats[tool]['models'][model] += 1
                
                if is_openai_model(model):
                    stats[tool]['openai'] += 1
                else:
                    stats[tool]['opensource'] += 1
    
    print(f"\nProcessed {total_lines} lines from {len(log_files)} log file(s)\n")
    print("=" * 80)
    print("HLE Tool Call Model Usage Summary")
    print("=" * 80)
    
    # Print per-tool breakdown
    for tool in ['enhance_reasoning', 'search', 'answer']:
        if tool not in stats:
            continue
        
        s = stats[tool]
        total = s['openai'] + s['opensource']
        if total == 0:
            continue
        
        opensource_pct = (s['opensource'] / total) * 100.0 if total > 0 else 0.0
        
        print(f"\nTool: {tool}")
        print(f"  Total calls: {total}")
        print(f"  GPT-5/GPT-5-mini: {s['openai']} ({(s['openai']/total*100.0):.1f}%)")
        print(f"  Open-source: {s['opensource']} ({opensource_pct:.1f}%)")
        print(f"  Breakdown by model:")
        
        # Sort models by count (descending)
        sorted_models = sorted(s['models'].items(), key=lambda x: x[1], reverse=True)
        for model, count in sorted_models:
            pct = (count / total) * 100.0
            model_type = "OpenAI" if is_openai_model(model) else "Open-source"
            print(f"    {model:50s} {count:5d} ({pct:5.1f}%) [{model_type}]")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("Overall Summary (all tools)")
    print("=" * 80)
    
    total_openai = sum(s['openai'] for s in stats.values())
    total_opensource = sum(s['opensource'] for s in stats.values())
    grand_total = total_openai + total_opensource
    
    if grand_total > 0:
        print(f"Total tool calls: {grand_total}")
        print(f"GPT-5/GPT-5-mini: {total_openai} ({(total_openai/grand_total*100.0):.1f}%)")
        print(f"Open-source: {total_opensource} ({(total_opensource/grand_total*100.0):.1f}%)")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

