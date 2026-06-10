#!/usr/bin/env python3
"""
scripts/download_data.py

Downloads the ShareGPT dataset used for benchmark prompts.
This ensures all benchmarks use real conversational prompts,
not synthetic or hardcoded inputs.

Usage:
  python scripts/download_data.py
"""

import os
import urllib.request
from pathlib import Path

DATA_DIR = Path("./data")
SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered"
    "/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
)
SHAREGPT_PATH = DATA_DIR / "ShareGPT_V3_unfiltered_cleaned_split.json"


def download_sharegpt():
    DATA_DIR.mkdir(exist_ok=True)

    if SHAREGPT_PATH.exists():
        size_mb = SHAREGPT_PATH.stat().st_size / 1024**2
        print(f"ShareGPT already downloaded ({size_mb:.1f}MB) → {SHAREGPT_PATH}")
        return

    print(f"Downloading ShareGPT dataset...")
    print(f"  URL:  {SHAREGPT_URL}")
    print(f"  Dest: {SHAREGPT_PATH}")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
        print(f"\r  Progress: {pct:.1f}% ({downloaded // 1024**2}MB)", end="", flush=True)

    urllib.request.urlretrieve(SHAREGPT_URL, SHAREGPT_PATH, reporthook=progress)
    print(f"\n  Done. {SHAREGPT_PATH.stat().st_size / 1024**2:.1f}MB")


def verify():
    import json
    print("Verifying dataset...")
    with open(SHAREGPT_PATH) as f:
        data = json.load(f)

    human_turns = sum(
        1 for conv in data
        for turn in conv.get("conversations", [])
        if turn.get("from") == "human" and turn.get("value", "").strip()
    )
    print(f"  Conversations: {len(data):,}")
    print(f"  Human turns:   {human_turns:,}")
    print("  Verification: PASS")


if __name__ == "__main__":
    download_sharegpt()
    verify()
