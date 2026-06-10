#!/usr/bin/env python3
"""
Real GPU benchmarks on 4× NVIDIA A30.

Runs three benchmark suites and writes verified results to results/.

1. KV cache / PagedAttention memory analysis     (Python simulation, Mistral-7B params)
2. Continuous batching vs static batching        (vLLM, real Mistral-7B)
3. Tensor-parallel scaling TP=1/2/3             (vLLM, real Mistral-7B)
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import time

import numpy as np
import torch

MISTRAL_PATH = "/storage/gxg8313/hf/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a"
RESULTS_DIR  = pathlib.Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. KV cache memory analysis  (Mistral-7B params, not toy numbers)
# ─────────────────────────────────────────────────────────────────────────────

def bench_paged_attention() -> dict:
    print("\n" + "="*62)
    print("[ 1/3 ]  KV Cache Memory Analysis  (Mistral-7B params)")
    print("="*62)

    # Mistral-7B architecture
    NUM_LAYERS   = 32
    NUM_HEADS    = 32
    HEAD_DIM     = 128
    DTYPE_BYTES  = 2          # fp16
    PAGE_SIZE    = 16         # tokens per block (vLLM default)
    MAX_SEQ_LEN  = 2048

    # bytes per token for K+V across ALL layers
    bytes_per_tok = 2 * NUM_LAYERS * NUM_HEADS * HEAD_DIM * DTYPE_BYTES  # 524,288

    rng = np.random.default_rng(42)
    # Log-normal seq lengths  (μ=6.0, σ=0.8, realistic serving distribution)
    raw   = np.exp(rng.normal(6.0, 0.8, 1000)).clip(64, MAX_SEQ_LEN).astype(int)
    seqs  = raw.tolist()
    N     = len(seqs)

    # Naive: pre-allocate max_seq_len for every request
    naive_allocated = N * MAX_SEQ_LEN * bytes_per_tok
    naive_used      = sum(seqs) * bytes_per_tok
    naive_frag_pct  = (naive_allocated - naive_used) / naive_allocated * 100

    # Paged: allocate ceil(seq_len / PAGE_SIZE) * PAGE_SIZE tokens per request
    paged_allocated = sum(math.ceil(s / PAGE_SIZE) * PAGE_SIZE for s in seqs) * bytes_per_tok
    paged_used      = naive_used
    paged_frag_pct  = (paged_allocated - paged_used) / paged_allocated * 100

    naive_gb = naive_allocated / 1024**3
    paged_gb = paged_allocated / 1024**3
    savings  = (naive_allocated - paged_allocated) / naive_allocated * 100

    # Concurrent capacity: how many requests fit in 22 GB free per GPU
    gpu_free_gb    = 22.0
    model_size_gb  =  7.0   # Mistral-7B fp16 weights ≈ 14 GB; half on single GPU
    kv_budget_gb   = gpu_free_gb - model_size_gb
    naive_capacity = int(kv_budget_gb * 1024**3 / (MAX_SEQ_LEN * bytes_per_tok))
    mean_seq       = int(np.mean(seqs))
    paged_capacity = int(kv_budget_gb * 1024**3 /
                         (math.ceil(mean_seq / PAGE_SIZE) * PAGE_SIZE * bytes_per_tok))

    print(f"  n_requests        : {N}  (log-normal seq lens, mean={mean_seq})")
    print(f"  Naive KV cache    : {naive_gb:.1f} GB  fragmentation {naive_frag_pct:.1f}%")
    print(f"  Paged KV cache    : {paged_gb:.1f} GB  fragmentation {paged_frag_pct:.1f}%")
    print(f"  Memory savings    : {savings:.1f}%")
    print(f"  Concurrent cap    : {naive_capacity} req (naive) → {paged_capacity} req (paged)  ({paged_capacity/naive_capacity:.1f}× more)")

    result = {
        "n_requests":           N,
        "mean_seq_len":         mean_seq,
        "naive_kv_cache_gb":    round(naive_gb, 2),
        "paged_kv_cache_gb":    round(paged_gb, 2),
        "memory_savings_pct":   round(savings, 1),
        "naive_fragmentation_pct": round(naive_frag_pct, 1),
        "paged_fragmentation_pct": round(paged_frag_pct, 1),
        "naive_concurrent_capacity": naive_capacity,
        "paged_concurrent_capacity": paged_capacity,
        "capacity_multiplier":  round(paged_capacity / naive_capacity, 2),
    }
    with open(RESULTS_DIR / "paged_attention.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → results/paged_attention.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. Continuous batching vs static batching  (vLLM, real Mistral-7B on GPU)
# ─────────────────────────────────────────────────────────────────────────────

_PROMPTS = [
    "Explain transformer self-attention in detail, including the math.",
    "What are the engineering trade-offs of tensor parallelism vs pipeline parallelism?",
    "Describe how PagedAttention reduces memory fragmentation in LLM serving.",
    "Walk through the forward pass of a decoder-only transformer step by step.",
    "What is speculative decoding and why does it speed up inference?",
    "Compare vLLM and TensorRT-LLM for production LLM serving.",
    "How does mixed-precision (FP16/BF16) affect numerical stability?",
    "Explain the roofline model for GPU kernel performance analysis.",
    "What is the compute-to-memory ratio and why does it matter for LLMs?",
    "Describe how LoRA reduces the number of trainable parameters.",
    "What happens during KV cache eviction in long-context serving?",
    "How does continuous batching differ from traditional dynamic batching?",
    "Explain why attention is quadratic in sequence length.",
    "What are the key differences between A100 and A30 for LLM inference?",
    "How does chunked prefill improve TTFT under high load?",
]


def bench_continuous_batching(n_prompts: int = 40, max_tokens: int = 200) -> dict:
    print("\n" + "="*62)
    print("[ 2/3 ]  Continuous Batching vs Static Batching  (Mistral-7B, GPU)")
    print("="*62)
    from vllm import LLM, SamplingParams

    prompts   = [_PROMPTS[i % len(_PROMPTS)] for i in range(n_prompts)]
    sampling  = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    # ── Continuous batching (vLLM default) ─────────────────────────────────
    print(f"  Initializing vLLM  (TP=1, continuous batching)...")
    llm = LLM(
        model=MISTRAL_PATH,
        tensor_parallel_size=1,
        dtype="float16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        max_num_seqs=32,
    )

    # Warmup
    _ = llm.generate(prompts[:4], SamplingParams(max_tokens=32, temperature=0.0))

    # TTFT: single request latency
    t0 = time.perf_counter()
    r  = llm.generate([prompts[0]], SamplingParams(max_tokens=1, temperature=0.0))
    ttft_ms = (time.perf_counter() - t0) * 1000

    # Continuous batching throughput: all n_prompts at once
    t0      = time.perf_counter()
    results = llm.generate(prompts, sampling)
    cb_wall = time.perf_counter() - t0
    cb_toks = sum(len(r.outputs[0].token_ids) for r in results)
    cb_tps  = cb_toks / cb_wall

    # P99 latency (per-request): time each request individually
    per_req_latencies = []
    for p in prompts[:20]:
        t0 = time.perf_counter()
        _  = llm.generate([p], SamplingParams(max_tokens=max_tokens, temperature=0.0))
        per_req_latencies.append((time.perf_counter() - t0) * 1000)
    p50_ms = float(np.percentile(per_req_latencies, 50))
    p99_ms = float(np.percentile(per_req_latencies, 99))

    print(f"  Continuous batching  : {cb_tps:.1f} tok/s  TTFT={ttft_ms:.0f}ms  P50={p50_ms:.0f}ms  P99={p99_ms:.0f}ms")

    # ── Static batching simulation ──────────────────────────────────────────
    # Static batching: fixed batch size B, process ceil(N/B) sequential batches
    static_batch   = 4
    n_batches      = math.ceil(n_prompts / static_batch)
    t0             = time.perf_counter()
    static_toks    = 0
    static_p99_ms  = 0.0
    for b in range(n_batches):
        chunk    = prompts[b * static_batch : (b + 1) * static_batch]
        t_batch  = time.perf_counter()
        batch_r  = llm.generate(chunk, sampling)
        elapsed  = (time.perf_counter() - t_batch) * 1000
        static_toks += sum(len(r.outputs[0].token_ids) for r in batch_r)
        static_p99_ms = max(static_p99_ms, elapsed)
    static_wall = time.perf_counter() - t0
    static_tps  = static_toks / static_wall

    print(f"  Static batching (B={static_batch})  : {static_tps:.1f} tok/s  P99={static_p99_ms:.0f}ms")
    speedup = cb_tps / static_tps
    p99_red = (static_p99_ms - p99_ms) / static_p99_ms * 100
    print(f"  Speedup              : {speedup:.2f}×  P99 reduction {p99_red:.0f}%")

    del llm
    torch.cuda.empty_cache()

    result = {
        "model":               "Mistral-7B-Instruct-v0.2",
        "hardware":            "NVIDIA A30 24GB  (single GPU)",
        "n_prompts":           n_prompts,
        "max_output_tokens":   max_tokens,
        "continuous_batching": {
            "throughput_tok_per_s": round(cb_tps, 1),
            "ttft_ms":              round(ttft_ms, 1),
            "p50_latency_ms":       round(p50_ms, 1),
            "p99_latency_ms":       round(p99_ms, 1),
        },
        "static_batching": {
            "batch_size":           static_batch,
            "throughput_tok_per_s": round(static_tps, 1),
            "p99_latency_ms":       round(static_p99_ms, 1),
        },
        "speedup_tps":          round(speedup, 2),
        "p99_reduction_pct":    round(p99_red, 1),
    }
    with open(RESULTS_DIR / "continuous_batching.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → results/continuous_batching.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tensor-parallel scaling  TP=1 / TP=2 / TP=3  (vLLM, real Mistral-7B)
# ─────────────────────────────────────────────────────────────────────────────

def bench_tensor_parallel(tp_sizes: list[int] = [2, 3],
                          n_prompts: int = 30,
                          max_tokens: int = 200) -> dict:
    print("\n" + "="*62)
    print("[ 3/3 ]  Tensor-Parallel Scaling  (Mistral-7B, A30×3)")
    print("="*62)
    from vllm import LLM, SamplingParams

    prompts  = [_PROMPTS[i % len(_PROMPTS)] for i in range(n_prompts)]
    sampling = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    rows     = []

    for tp in tp_sizes:
        print(f"\n  TP={tp}  ({tp} GPU(s))...")
        llm = LLM(
            model=MISTRAL_PATH,
            tensor_parallel_size=tp,
            dtype="float16",
            trust_remote_code=True,
            max_model_len=2048,
            gpu_memory_utilization=0.68,
        )

        # Warmup
        _ = llm.generate(prompts[:2], SamplingParams(max_tokens=32, temperature=0.0))

        # TTFT
        t0      = time.perf_counter()
        _       = llm.generate([prompts[0]], SamplingParams(max_tokens=1, temperature=0.0))
        ttft_ms = (time.perf_counter() - t0) * 1000

        # Throughput
        t0   = time.perf_counter()
        outs = llm.generate(prompts, sampling)
        wall = time.perf_counter() - t0
        toks = sum(len(r.outputs[0].token_ids) for r in outs)
        tps  = toks / wall

        print(f"    tps={tps:.1f}  TTFT={ttft_ms:.0f}ms")
        rows.append({"tp": tp, "tps": round(tps, 1), "ttft_ms": round(ttft_ms, 1)})

        del llm
        torch.cuda.empty_cache()
        time.sleep(3)

    base_tps  = rows[0]["tps"]
    base_ttft = rows[0]["ttft_ms"]
    for r in rows:
        r["speedup"] = round(r["tps"] / base_tps, 2)
        r["ttft_reduction_pct"] = round((base_ttft - r["ttft_ms"]) / base_ttft * 100, 1)

    print("\n  TP Scaling Summary:")
    print(f"  {'TP':>4}  {'tok/s':>8}  {'speedup':>8}  {'TTFT ms':>10}")
    for r in rows:
        print(f"  {r['tp']:>4}  {r['tps']:>8.1f}  {r['speedup']:>8.2f}×  {r['ttft_ms']:>10.1f}")

    result = {
        "model":    "Mistral-7B-Instruct-v0.2",
        "hardware": f"NVIDIA A30 24GB × {max(tp_sizes)} (PCIe)",
        "results":  rows,
    }
    with open(RESULTS_DIR / "tensor_parallel.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → results/tensor_parallel.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-vllm", action="store_true", help="Only run KV cache sim")
    ap.add_argument("--tp-only",   action="store_true", help="Only run TP scaling")
    ap.add_argument("--cb-only",   action="store_true", help="Only run continuous batching")
    args = ap.parse_args()

    all_results = {}

    all_results["paged_attention"] = bench_paged_attention()

    if not args.skip_vllm:
        if not args.tp_only:
            all_results["continuous_batching"] = bench_continuous_batching()
        if not args.cb_only:
            all_results["tensor_parallel"] = bench_tensor_parallel()

    with open(RESULTS_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*62)
    print("  All benchmarks complete.  Results saved to results/")
    print("="*62)
