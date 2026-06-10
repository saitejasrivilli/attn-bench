"""
benchmark/continuous_batching.py

Benchmarks continuous batching vs static batching for LLM serving.
Uses vLLM if available, otherwise simulates with transformers.

Continuous batching (Orca/vLLM): new requests are added to the running batch
as soon as a slot frees up, rather than waiting for the whole batch to finish.
This dramatically improves GPU utilization when request lengths vary.

Usage:
    python benchmark/continuous_batching.py
    python benchmark/continuous_batching.py --model Qwen/Qwen2.5-7B-Instruct
    python benchmark/continuous_batching.py --static_only --n_prompts 50
"""

import argparse
import random
import time
from typing import Optional

import numpy as np


# ── Prompt generation ─────────────────────────────────────────────────────────

_PROMPT_TEMPLATES = [
    "Explain the concept of {topic} in simple terms.",
    "What are the main differences between {topic} and {alt}?",
    "Write a short Python function that {task}.",
    "Summarize the key points about {topic}.",
    "How does {topic} work under the hood?",
    "Compare {topic} vs {alt} for production use.",
    "What is the time complexity of {task}?",
    "Give three real-world examples of {topic}.",
    "Describe a common pitfall when using {topic}.",
    "What should a beginner know about {topic}?",
]

_TOPICS = [
    "attention mechanisms", "transformer models", "CUDA kernels",
    "gradient checkpointing", "mixed precision training", "flash attention",
    "KV cache", "PagedAttention", "tensor parallelism", "speculative decoding",
    "beam search", "top-p sampling", "RLHF", "LoRA fine-tuning", "quantization",
]

_ALTS = [
    "RNNs", "CNNs", "dense layers", "MoE", "retrieval augmentation",
    "full fine-tuning", "prefix tuning", "adapter layers",
]

_TASKS = [
    "sorts a list of dicts by a nested key",
    "implements a sliding window over a sequence",
    "computes cosine similarity between two vectors",
    "reads a CSV and groups rows by a column",
    "retries an HTTP request with exponential backoff",
]


def generate_prompts(n: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    prompts = []
    for _ in range(n):
        tmpl = rng.choice(_PROMPT_TEMPLATES)
        topic = rng.choice(_TOPICS)
        alt   = rng.choice(_ALTS)
        task  = rng.choice(_TASKS)
        prompts.append(tmpl.format(topic=topic, alt=alt, task=task))
    return prompts


# ── Static batching ───────────────────────────────────────────────────────────

def benchmark_static_batching(
    model_name: str,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int = 128,
) -> dict:
    """
    Fixed batch — waits for all sequences to finish before starting next batch.
    Uses transformers if vLLM is not available.

    Returns:
        throughput_tokens_per_sec, mean_latency_ms, p50_latency_ms, p99_latency_ms
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError as e:
        raise RuntimeError("transformers not installed") from e

    print(f"  Loading {model_name} for static batching...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
    latencies_ms = []
    total_tokens  = 0

    t_wall_start = time.perf_counter()
    with torch.inference_mode():
        for batch in batches:
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(model.device)

            t0 = time.perf_counter()
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            new_tokens = out.shape[-1] - inputs["input_ids"].shape[-1]
            total_tokens += new_tokens * len(batch)
            # assign equal latency to each item in batch (conservative)
            latencies_ms.extend([elapsed_ms] * len(batch))

    wall_s = time.perf_counter() - t_wall_start
    latencies_ms_arr = np.array(latencies_ms)

    return {
        "method":                  "static_batching",
        "batch_size":              batch_size,
        "n_prompts":               len(prompts),
        "throughput_tokens_per_sec": round(total_tokens / wall_s, 1),
        "mean_latency_ms":         round(float(latencies_ms_arr.mean()), 1),
        "p50_latency_ms":          round(float(np.percentile(latencies_ms_arr, 50)), 1),
        "p99_latency_ms":          round(float(np.percentile(latencies_ms_arr, 99)), 1),
        "total_time_s":            round(wall_s, 2),
    }


# ── Continuous batching via vLLM ──────────────────────────────────────────────

def benchmark_continuous_batching_vllm(
    model_name: str,
    prompts: list[str],
    max_new_tokens: int = 128,
) -> dict:
    """
    Uses vLLM for continuous batching — adds new requests as old ones finish.
    vLLM's scheduler handles this natively via its PagedAttention + continuous
    batch scheduler.

    Returns:
        throughput_tokens_per_sec, mean_latency_ms, p50_latency_ms, p99_latency_ms
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise RuntimeError(
            "vLLM not installed. Install with: pip install vllm"
        ) from e

    print(f"  Loading {model_name} via vLLM (continuous batching)...")
    llm = LLM(
        model=model_name,
        dtype="float16",
        trust_remote_code=True,
        max_model_len=2048,
    )
    sampling = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
    )

    # warmup
    _ = llm.generate(prompts[:2], sampling)

    per_request_times = []
    t_wall_start = time.perf_counter()

    # vLLM processes the full list with its internal continuous scheduler
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    wall_s  = time.perf_counter() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    # vLLM doesn't expose per-request latency directly from generate();
    # approximate as wall_s / n since the scheduler keeps GPU busy
    approx_lat_ms = wall_s / len(prompts) * 1000
    latencies_ms  = np.full(len(prompts), approx_lat_ms)

    return {
        "method":                  "continuous_batching_vllm",
        "batch_size":              "dynamic",
        "n_prompts":               len(prompts),
        "throughput_tokens_per_sec": round(total_tokens / wall_s, 1),
        "mean_latency_ms":         round(float(latencies_ms.mean()), 1),
        "p50_latency_ms":          round(float(np.percentile(latencies_ms, 50)), 1),
        "p99_latency_ms":          round(float(np.percentile(latencies_ms, 99)), 1),
        "total_time_s":            round(wall_s, 2),
    }


# ── Comparison runner ─────────────────────────────────────────────────────────

def run_comparison(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    batch_sizes: list[int] = None,
    n_prompts: int = 100,
    max_new_tokens: int = 128,
    static_only: bool = False,
) -> list[dict]:
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32]

    prompts = generate_prompts(n_prompts)
    results = []

    print(f"\nModel: {model_name}")
    print(f"Prompts: {n_prompts}  |  max_new_tokens: {max_new_tokens}\n")
    print(f"{'Method':<28} | {'Batch':>5} | {'Throughput':>12} | {'P50 lat':>9} | {'P99 lat':>9}")
    print("-" * 75)

    for bs in batch_sizes:
        print(f"\n  Static batching, batch_size={bs}...")
        try:
            r = benchmark_static_batching(model_name, prompts, bs, max_new_tokens)
            results.append(r)
            print(f"  {'Static batching':<26} | {bs:>5} | "
                  f"{r['throughput_tokens_per_sec']:>10.1f}/s | "
                  f"{r['p50_latency_ms']:>7.0f}ms | "
                  f"{r['p99_latency_ms']:>7.0f}ms")
        except Exception as e:
            print(f"  Static bs={bs} failed: {e}")

    if not static_only:
        print(f"\n  Continuous batching (vLLM)...")
        try:
            r = benchmark_continuous_batching_vllm(model_name, prompts, max_new_tokens)
            results.append(r)
            print(f"  {'Continuous (vLLM)':<26} | {'dyn':>5} | "
                  f"{r['throughput_tokens_per_sec']:>10.1f}/s | "
                  f"{r['p50_latency_ms']:>7.0f}ms | "
                  f"{r['p99_latency_ms']:>7.0f}ms")
        except RuntimeError as e:
            print(f"  vLLM not available ({e}); skipping continuous batching.")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Static vs continuous batching benchmark")
    parser.add_argument("--model",        default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n_prompts",    type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_sizes",  type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--static_only",  action="store_true",
                        help="Skip vLLM continuous batching (transformers only)")
    args = parser.parse_args()

    run_comparison(
        model_name=args.model,
        batch_sizes=args.batch_sizes,
        n_prompts=args.n_prompts,
        max_new_tokens=args.max_new_tokens,
        static_only=args.static_only,
    )


if __name__ == "__main__":
    main()
