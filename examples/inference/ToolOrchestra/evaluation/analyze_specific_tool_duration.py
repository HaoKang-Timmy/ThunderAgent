#!/usr/bin/env python3
"""
Analyze duration statistics for a specific tool+model combination.
"""

import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def extract_duration_for_tool_model(log_files, tool_name, model_name):
    """Extract all duration_ms values for a specific tool+model combination."""
    durations = []
    
    for log_file in log_files:
        if not log_file.exists():
            print(f"Warning: {log_file} not found, skipping", file=sys.stderr)
            continue
        
        print(f"Reading {log_file}...", file=sys.stderr)
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if 'type=tool_call' not in line:
                    continue
                
                # Check if this line matches our tool+model
                if f'tool={tool_name}' not in line:
                    continue
                if f'model={model_name}' not in line:
                    continue
                
                # Extract duration_ms
                match = re.search(r'duration_ms=([0-9]+(?:\.[0-9]+)?)', line)
                if match:
                    durations.append(float(match.group(1)))
    
    return durations

def compute_stats(values):
    """Compute comprehensive statistics."""
    if len(values) == 0:
        return None
    
    arr = np.array(values)
    
    stats = {
        'n': len(values),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)) if len(values) > 1 else 0.0,
        'percentiles': {
            'p1': float(np.percentile(arr, 1)),
            'p10': float(np.percentile(arr, 10)),
            'p20': float(np.percentile(arr, 20)),
            'p25': float(np.percentile(arr, 25)),
            'p30': float(np.percentile(arr, 30)),
            'p40': float(np.percentile(arr, 40)),
            'p50': float(np.percentile(arr, 50)),
            'p75': float(np.percentile(arr, 75)),
            'p90': float(np.percentile(arr, 90)),
            'p95': float(np.percentile(arr, 95)),
            'p99': float(np.percentile(arr, 99)),
        }
    }
    
    return stats

def plot_histogram(values, bins, title, xlabel, output_path):
    """Plot a histogram with specified number of bins."""
    if len(values) == 0:
        print("No data to plot")
        return
    
    arr = np.array(values)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
    
    # Create histogram
    counts, edges, patches = ax.hist(arr, bins=bins, edgecolor='black', alpha=0.7)
    
    # Add percentage labels on top of bars
    total = len(arr)
    for count, patch in zip(counts, patches):
        if count > 0:
            height = patch.get_height()
            pct = (count / total) * 100.0
            ax.text(patch.get_x() + patch.get_width() / 2.0, height,
                   f'{pct:.1f}%\n({int(count)})',
                   ha='center', va='bottom', fontsize=9)
    
    # Add statistics to title
    stats = compute_stats(values)
    title_with_stats = f"{title}\n(n={stats['n']}, mean={stats['mean']:.1f}ms, std={stats['std']:.1f}ms, median={stats['percentiles']['p50']:.1f}ms)"
    
    ax.set_title(title_with_stats, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nHistogram saved to: {output_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_specific_tool_duration.py <log_file1> [log_file2] ...")
        print("Example: python analyze_specific_tool_duration.py logs/hle.log")
        return 1
    
    log_files = [Path(p) for p in sys.argv[1:]]
    
    # Target tool and model
    tool_name = "enhance_reasoning"
    model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
    
    print(f"\nExtracting duration data for:", file=sys.stderr)
    print(f"  Tool: {tool_name}", file=sys.stderr)
    print(f"  Model: {model_name}", file=sys.stderr)
    print(file=sys.stderr)
    
    durations = extract_duration_for_tool_model(log_files, tool_name, model_name)
    
    if len(durations) == 0:
        print(f"\nNo data found for tool={tool_name} model={model_name}")
        return 1
    
    # Compute statistics
    stats = compute_stats(durations)
    
    # Print results
    print("\n" + "=" * 80)
    print(f"Duration Statistics: {tool_name} with {model_name}")
    print("=" * 80)
    print(f"\nSample size: {stats['n']}")
    print(f"\nBasic statistics (milliseconds):")
    print(f"  Min:  {stats['min']:10.2f} ms")
    print(f"  Max:  {stats['max']:10.2f} ms")
    print(f"  Mean: {stats['mean']:10.2f} ms")
    print(f"  Std:  {stats['std']:10.2f} ms")
    
    print(f"\nPercentiles (milliseconds):")
    for pct_name, pct_value in sorted(stats['percentiles'].items()):
        pct_num = pct_name[1:]  # Remove 'p' prefix
        print(f"  {pct_num:>3s}th: {pct_value:10.2f} ms")
    
    print("\n" + "=" * 80)
    
    # Convert to seconds for easier reading
    durations_sec = [d / 1000.0 for d in durations]
    print(f"\nDuration range: {min(durations_sec):.2f}s - {max(durations_sec):.2f}s")
    
    # Plot histogram
    output_path = Path("/tmp/enhance_reasoning_qwen_coder_duration_hist.png")
    plot_histogram(
        durations,
        bins=10,
        title=f"{tool_name} Duration Distribution ({model_name})",
        xlabel="Duration (milliseconds)",
        output_path=output_path
    )
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

