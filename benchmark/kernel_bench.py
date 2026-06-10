"""
benchmark/kernel_bench.py

Standalone kernel benchmark — measures raw attention kernel performance
without inference engine overhead (no vLLM, no SGLang).

Outputs:
  results/kernel_bench.json  — timing, throughput, memory per config
  results/kernel_bench.csv   — same data in tabular form

Run:
  python benchmark/kernel_bench.py --config configs/bench_config.yaml
"""

import json
import csv
import time
import argparse
from pathlib import Path
from typing import Dict, List

import torch
import yaml

# ── ensure project root is on path ───────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from kernels.triton_attention import TritonAttention, verify_correctness
from kernels.cuda_baseline import FlashAttentionBaseline


# ── timing utility ────────────────────────────────────────────────────────────

def cuda_time_ms(fn, warmup: int, iters: int) -> float:
    """
    Returns median wall-clock time in ms using CUDA events.
    More accurate than time.perf_counter for GPU kernels.
    """
    # warmup — JIT compile, caches
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)
    times = []

    for _ in range(iters):
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    times.sort()
    return times[len(times) // 2]   # median


def measure_memory_gb(fn) -> float:
    """Peak GPU memory allocated during forward pass (GB)."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**3


# ── benchmark single config ───────────────────────────────────────────────────

def bench_config(
    kernel_name: str,
    kernel,
    batch: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    warmup: int,
    iters: int,
    device: str = "cuda",
) -> Dict:

    q = torch.randn(batch, num_heads,    seq_len, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.float16, device=device)

    def fwd():
        with torch.no_grad():
            kernel(q, k, v)

    latency_ms  = cuda_time_ms(fwd, warmup, iters)
    memory_gb   = measure_memory_gb(fwd)

    # tokens processed per second
    total_tokens   = batch * seq_len
    throughput_tps = total_tokens / (latency_ms / 1000.0)

    # FLOP estimate: 4 * batch * num_heads * seq^2 * head_dim (QK^T + AV)
    flops = 4 * batch * num_heads * seq_len * seq_len * head_dim
    tflops = flops / (latency_ms / 1000.0) / 1e12

    return {
        "kernel":        kernel_name,
        "batch":         batch,
        "seq_len":       seq_len,
        "num_heads":     num_heads,
        "num_kv_heads":  num_kv_heads,
        "head_dim":      head_dim,
        "latency_ms":    round(latency_ms, 4),
        "throughput_tps": round(throughput_tps, 1),
        "memory_gb":     round(memory_gb, 4),
        "tflops":        round(tflops, 3),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bench_config.yaml")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    with open(args.config)       as f: cfg   = yaml.safe_load(f)
    with open(args.model_config) as f: mcfg  = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    print(f"\n=== Kernel Benchmark on {torch.cuda.get_device_name(0)} ===\n")

    # ── correctness first — fail early if kernel is wrong ──────────────────
    print("Running correctness check...")
    passed = verify_correctness(device=device)
    if not passed:
        raise RuntimeError("Triton kernel failed correctness check. Fix before benchmarking.")
    print()

    # ── build kernels ─────────────────────────────────────────────────────
    num_heads    = mcfg["attention"]["num_heads"]
    num_kv_heads = mcfg["attention"]["num_kv_heads"]
    head_dim     = mcfg["attention"]["head_dim"]

    kernels = {
        "flashattn2_cuda": FlashAttentionBaseline(num_heads, num_kv_heads, head_dim).to(device),
        "triton_custom":   TritonAttention(num_heads, num_kv_heads, head_dim).to(device),
    }

    kcfg    = cfg["kernel_bench"]
    warmup  = kcfg["warmup_iters"]
    iters   = kcfg["bench_iters"]

    results = []
    total   = len(kernels) * len(kcfg["batch_sizes"]) * len(kcfg["seq_lengths"])
    done    = 0

    for kernel_name, kernel in kernels.items():
        for batch in kcfg["batch_sizes"]:
            for seq_len in kcfg["seq_lengths"]:
                done += 1
                print(f"[{done}/{total}] {kernel_name} | batch={batch} | seq_len={seq_len}")
                row = bench_config(
                    kernel_name=kernel_name,
                    kernel=kernel,
                    batch=batch,
                    seq_len=seq_len,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    warmup=warmup,
                    iters=iters,
                    device=device,
                )
                results.append(row)
                print(f"  latency={row['latency_ms']:.2f}ms | "
                      f"throughput={row['throughput_tps']:.0f} tok/s | "
                      f"mem={row['memory_gb']:.3f}GB | "
                      f"tflops={row['tflops']:.2f}")

    # ── save results ──────────────────────────────────────────────────────
    json_path = out_dir / "kernel_bench.json"
    csv_path  = out_dir / "kernel_bench.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved → {json_path}, {csv_path}")

    # ── print speedup summary ─────────────────────────────────────────────
    print("\n=== Speedup Summary (Triton vs FlashAttn-2) ===")
    fa2     = {(r["batch"], r["seq_len"]): r for r in results if r["kernel"] == "flashattn2_cuda"}
    triton_ = {(r["batch"], r["seq_len"]): r for r in results if r["kernel"] == "triton_custom"}
    for key in fa2:
        b, s = key
        speedup = fa2[key]["latency_ms"] / triton_[key]["latency_ms"]
        print(f"  batch={b} seq={s}: {speedup:.2f}x speedup | "
              f"FA2={fa2[key]['latency_ms']:.2f}ms | "
              f"Triton={triton_[key]['latency_ms']:.2f}ms")


if __name__ == "__main__":
    main()
