#!/usr/bin/env python3
"""
Download eval.index and eval.jsonl from HuggingFace dataset multi-train/index
Usage: python download_index.py [--local-dir PATH]
"""

from huggingface_hub import hf_hub_download
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Download index files from multi-train/index dataset')
    parser.add_argument(
        '--local-dir',
        type=str,
        default='/workspace/dataset/multi-train/index',
        help='Local directory to save downloaded files (default: /workspace/dataset/multi-train/index)'
    )
    args = parser.parse_args()

    # Get HF token from environment
    token = os.environ.get('HF_TOKEN')
    if not token:
        print("Warning: HF_TOKEN not found in environment. You may need to:")
        print("  source ToolOrchestra/set_up_hf_env.sh")
        print("or set HF_TOKEN manually")
    else:
        print(f"Using HF_TOKEN: {token[:10]}...")

    # Create local directory if it doesn't exist
    os.makedirs(args.local_dir, exist_ok=True)
    print(f"Target directory: {args.local_dir}\n")

    # Download the files
    files_to_download = ['eval.index', 'eval.jsonl']
    repo_id = 'multi-train/index'

    for filename in files_to_download:
        print(f"Downloading {filename}...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type='dataset',
                token=token,
                local_dir=args.local_dir,
                local_dir_use_symlinks=False
            )
            print(f"✓ Successfully downloaded {filename} to {downloaded_path}\n")
        except Exception as e:
            print(f"✗ Error downloading {filename}: {e}\n")
            return 1

    print("✓ Download complete!")
    return 0


if __name__ == '__main__':
    exit(main())

