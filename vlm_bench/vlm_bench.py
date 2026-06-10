#!/usr/bin/env python3
"""
VLM (Qwen2-VL-7B-Instruct) inference benchmark via vLLM.
Measures TTFT and decode throughput for image+text inputs.
Hardware: NVIDIA A30 (SM80, 24GB), vLLM 0.17.1
"""
import time
import json
import statistics
import os
os.environ["TRANSFORMERS_CACHE"] = "/storage/gxg8313/hf/hub"

import numpy as np
from PIL import Image
from vllm import LLM, SamplingParams

MODEL_PATH = "/storage/gxg8313/hf/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac"

def make_dummy_image(width=336, height=336):
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")

def run_vlm_benchmark():
    print("=" * 60)
    print("VLM Benchmark: Qwen2-VL-7B-Instruct on A30")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────
    print("\n[1/3] Loading model (fp16)...")
    t0 = time.time()
    llm = LLM(
        model=MODEL_PATH,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        limit_mm_per_prompt={"image": 1},
    )
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    dummy_img = make_dummy_image(336, 336)

    # vLLM Qwen2-VL prompt format
    PROMPT = ("<|im_start|>user\n"
              "<|vision_start|><|image_pad|><|vision_end|>"
              "Describe this image.<|im_end|>\n"
              "<|im_start|>assistant\n")

    # ── TTFT benchmark ─────────────────────────────────────────────
    print("\n[2/3] TTFT benchmark (single-request, 5 warmup + 20 measured)...")
    sampling_ttft = SamplingParams(max_tokens=1, temperature=0)

    # warmup
    for _ in range(5):
        llm.generate(
            {"prompt": PROMPT, "multi_modal_data": {"image": dummy_img}},
            sampling_ttft,
        )

    ttft_ms = []
    for _ in range(20):
        t0 = time.perf_counter()
        llm.generate(
            {"prompt": PROMPT, "multi_modal_data": {"image": dummy_img}},
            sampling_ttft,
        )
        ttft_ms.append((time.perf_counter() - t0) * 1000)

    p50 = statistics.median(ttft_ms)
    p95 = sorted(ttft_ms)[int(0.95 * len(ttft_ms))]
    p99 = max(ttft_ms)
    print(f"  TTFT  p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms")

    # ── Decode throughput benchmark ────────────────────────────────
    print("\n[3/3] Decode throughput (batch=8, max_tokens=128, 5 runs)...")
    sampling_decode = SamplingParams(max_tokens=128, temperature=0)
    batch_size = 8
    batch_prompts = [
        {"prompt": PROMPT, "multi_modal_data": {"image": dummy_img}}
        for _ in range(batch_size)
    ]

    # warmup
    llm.generate(batch_prompts, sampling_decode)

    throughput_runs = []
    for i in range(5):
        t0 = time.perf_counter()
        outputs = llm.generate(batch_prompts, sampling_decode)
        elapsed = time.perf_counter() - t0
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        tok_s = total_tokens / elapsed
        throughput_runs.append(tok_s)
        print(f"  Run {i+1}: {tok_s:.1f} tok/s ({total_tokens} tokens, {elapsed:.2f}s)")

    avg_tok_s = statistics.mean(throughput_runs)
    print(f"\n  Avg throughput: {avg_tok_s:.1f} tok/s (B={batch_size}, max_tokens=128)")

    results = {
        "model": "Qwen2-VL-7B-Instruct",
        "hardware": "NVIDIA A30 SM80 24GB",
        "dtype": "fp16",
        "vllm_version": "0.17.1",
        "image_size": "336x336",
        "ttft_ms": {"p50": round(p50, 1), "p95": round(p95, 1), "p99": round(p99, 1), "n": len(ttft_ms)},
        "decode_throughput": {"batch": batch_size, "max_tokens": 128, "avg_tok_s": round(avg_tok_s, 1), "runs": [round(x, 1) for x in throughput_runs]},
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  TTFT p50/p99: {p50:.1f}ms / {p99:.1f}ms")
    print(f"  Decode throughput (B={batch_size}): {avg_tok_s:.1f} tok/s")
    print("  Results saved to results.json")


if __name__ == "__main__":
    run_vlm_benchmark()
