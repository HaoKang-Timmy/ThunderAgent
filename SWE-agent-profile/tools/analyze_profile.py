#!/usr/bin/env python3
"""Analyze step_profiles.csv and compute statistics."""
import argparse
import csv
import math
import sys
from pathlib import Path


def analyze_csv(csv_path: str):
    """Analyze the step profiles CSV file."""
    prefill_times = []
    decode_times = []
    tool_call_times = []
    kv_hit_rates = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prefill_times.append(float(row['prefill_s']))
            decode_times.append(float(row['decode_s']))
            tool_call_times.append(float(row['tool_call_s']))
            kv_rate = row.get('kv_hit_rate')
            if kv_rate:
                kv_hit_rates.append(float(kv_rate))
    
    n = len(prefill_times)
    if n == 0:
        print("No data found in CSV.")
        return
    
    # Averages
    avg_prefill = sum(prefill_times) / n
    avg_decode = sum(decode_times) / n
    avg_tool_call = sum(tool_call_times) / n
    avg_kv_hit_rate = sum(kv_hit_rates) / len(kv_hit_rates) if kv_hit_rates else 0
    
    # Variance of tool_call (population variance)
    if n > 1:
        mean_tc = avg_tool_call
        var_tool_call = sum((x - mean_tc) ** 2 for x in tool_call_times) / n
        std_tool_call = math.sqrt(var_tool_call)
    else:
        var_tool_call = 0
        std_tool_call = 0
    
    # Print results
    print(f"=== Profile Analysis: {csv_path} ===")
    print(f"Total steps: {n}")
    print()
    print(f"Average prefill time:    {avg_prefill:.4f} s")
    print(f"Average decode time:     {avg_decode:.4f} s")
    print(f"Average tool_call time:  {avg_tool_call:.4f} s")
    print(f"Average KV hit rate:     {avg_kv_hit_rate:.4f} ({avg_kv_hit_rate*100:.2f}%)")
    print()
    print(f"Tool call variance:      {var_tool_call:.4f}")
    print(f"Tool call std dev:       {std_tool_call:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze step_profiles.csv")
    parser.add_argument(
        "csv_file",
        nargs="?",
        default="step_profiles.csv",
        help="Path to the CSV file (default: step_profiles.csv)",
    )
    args = parser.parse_args()
    
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    
    analyze_csv(str(csv_path))


if __name__ == "__main__":
    main()

