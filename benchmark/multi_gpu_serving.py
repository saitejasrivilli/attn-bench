"""
benchmark/multi_gpu_serving.py

Multi-GPU tensor-parallel serving benchmark on 4×A30.
Compares 1-GPU vs 2-GPU vs 4-GPU with tensor parallelism via vLLM.

Tensor parallelism splits each attention/MLP layer across N GPUs so each
device holds only 1/N of the weight matrices. On PCIe-connected A30s,
inter-GPU bandwidth (~16 GB/s) limits scaling efficiency vs NVLink (~600 GB/s).

Usage:
    python benchmark/multi_gpu_serving.py
    python benchmark/multi_gpu_serving.py --model Qwen/Qwen2.5-7B-Instruct --tp_sizes 1 2 4
    python benchmark/multi_gpu_serving.py --n_requests 100 --max_tokens 512
"""

import argparse
import time
from typing import Optional

import numpy as np


# ── Prompt generation ─────────────────────────────────────────────────────────

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


def _make_prompts(n: int) -> list[str]:
    return [_PROMPTS[i % len(_PROMPTS)] for i in range(n)]


# ── Single TP configuration benchmark ─────────────────────────────────────────

def _run_single_tp(
    model_name: str,
    tp_size: int,
    prompts: list[str],
    max_tokens: int,
) -> dict:
    """Loads model at given tp_size and measures TTFT + throughput."""
    from vllm import LLM, SamplingParams
    import torch

    print(f"  Initializing vLLM  TP={tp_size}  ({tp_size} GPU(s))...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        dtype="float16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.90,
    )

    sampling = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    # warmup — 2 requests
    _ = llm.generate(prompts[:2], sampling)

    # measure TTFT via a single-request pass first
    t0 = time.perf_counter()
    _ = llm.generate([prompts[0]], SamplingParams(max_tokens=1, temperature=0.0))
    ttft_ms = (time.perf_counter() - t0) * 1000

    # throughput pass
    t_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.perf_counter() - t_start

    total_out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput_tps   = total_out_tokens / elapsed

    # GPU memory per device
    mem_per_gpu_gb = 0.0
    if torch.cuda.is_available():
        # vLLM claims GPUs 0..tp_size-1
        mem_bytes = sum(
            torch.cuda.memory_reserved(i) for i in range(tp_size)
        )
        mem_per_gpu_gb = (mem_bytes / tp_size) / 1024**3

    return {
        "tp_size":          tp_size,
        "ttft_ms":          round(ttft_ms, 1),
        "throughput_tps":   round(throughput_tps, 1),
        "mem_per_gpu_gb":   round(mem_per_gpu_gb, 1),
        "total_time_s":     round(elapsed, 2),
        "n_requests":       len(prompts),
    }


# ── Main benchmark function ───────────────────────────────────────────────────

def benchmark_tp_serving(
    model_name: str,
    tp_sizes: list[int],
    n_requests: int = 50,
    max_tokens: int = 256,
) -> list[dict]:
    """
    For each tensor_parallel_size, measures TTFT, throughput, and GPU memory.
    Returns list of result dicts.

    Note: each TP configuration requires re-loading the model, which takes
    several minutes per configuration. Run sequentially on the cluster.
    """
    try:
        from vllm import LLM, SamplingParams  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "vLLM is required for multi-GPU serving benchmark. "
            "Install with: pip install vllm"
        ) from e

    prompts = _make_prompts(n_requests)
    results = []

    for tp in tp_sizes:
        try:
            r = _run_single_tp(model_name, tp, prompts, max_tokens)
            results.append(r)
        except Exception as e:
            print(f"  TP={tp} failed: {e}")
            results.append({
                "tp_size": tp,
                "ttft_ms": None,
                "throughput_tps": None,
                "mem_per_gpu_gb": None,
                "error": str(e),
            })

    return results


# ── Results table ─────────────────────────────────────────────────────────────

def print_results_table(results: list[dict]):
    base_tps = None
    for r in results:
        if r.get("throughput_tps") is not None:
            base_tps = r["throughput_tps"]
            break

    print(f"\n{'=' * 72}")
    print(f" Multi-GPU Tensor-Parallel Serving — 4×A30 (PCIe)")
    print(f"{'=' * 72}")
    print(f" {'TP Size':>7} | {'TTFT (ms)':>9} | {'Throughput':>12} | "
          f"{'Mem/GPU':>9} | {'Speedup':>8}")
    print(f" {'-' * 68}")

    for r in results:
        if r.get("throughput_tps") is None:
            print(f" {r['tp_size']:>7} | {'ERROR':>9} | {'—':>12} | {'—':>9} | {'—':>8}")
            continue
        speedup = r["throughput_tps"] / base_tps if base_tps else 0.0
        print(f" {r['tp_size']:>7} | "
              f"{r['ttft_ms']:>9.1f} | "
              f"{r['throughput_tps']:>10.1f}/s | "
              f"{r['mem_per_gpu_gb']:>7.1f} GB | "
              f"{speedup:>7.2f}x")

    print(f"\n Note: PCIe bandwidth (~16 GB/s) limits TP scaling vs NVLink (~600 GB/s).")
    print(f"       Expect ~1.7x at TP=2, ~3.1x at TP=4 on PCIe A30 topology.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU TP serving benchmark")
    parser.add_argument("--model",      default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tp_sizes",   type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--n_requests", type=int, default=50)
    parser.add_argument("--max_tokens", type=int, default=256)
    args = parser.parse_args()

    print(f"\nMulti-GPU TP Serving Benchmark")
    print(f"Model:      {args.model}")
    print(f"TP sizes:   {args.tp_sizes}")
    print(f"Requests:   {args.n_requests}  |  max_tokens: {args.max_tokens}")

    results = benchmark_tp_serving(
        model_name=args.model,
        tp_sizes=args.tp_sizes,
        n_requests=args.n_requests,
        max_tokens=args.max_tokens,
    )

    print_results_table(results)


if __name__ == "__main__":
    main()
