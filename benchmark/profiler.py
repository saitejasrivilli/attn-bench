"""
benchmark/profiler.py

torch.profiler integration and roofline analysis for attention kernels.

Outputs:
  - Chrome trace JSON (open in chrome://tracing or Perfetto)
  - Roofline plot (compute vs memory bandwidth bound)
  - Per-kernel CUDA stats (SM util, memory BW, occupancy)

Usage:
  from benchmark.profiler import profile_kernel, roofline_analysis
"""

import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.profiler as profiler


# ── A30 hardware limits (actual specs) ────────────────────────────────────────
A30_SPECS = {
    "peak_tflops_fp16":   165.0,   # TFLOPS (tensor core fp16)
    "peak_bandwidth_tbps": 0.933,   # TB/s HBM2 bandwidth
    "sm_count":           56,
    "cuda_cores":         3584,
}


# ── torch.profiler wrapper ────────────────────────────────────────────────────

def profile_kernel(
    fn,
    label: str,
    output_dir: str = "results/profiles",
    wait: int = 2,
    warmup: int = 3,
    active: int = 5,
    record_shapes: bool = True,
    with_flops: bool = True,
) -> Dict:
    """
    Profiles `fn` with torch.profiler and saves Chrome trace.

    Returns dict with aggregated CUDA kernel stats.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    trace_file = str(out_path / f"{label}_trace.json")

    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        schedule=profiler.schedule(wait=wait, warmup=warmup, active=active),
        on_trace_ready=profiler.tensorboard_trace_handler(str(out_path)),
        record_shapes=record_shapes,
        with_flops=with_flops,
        with_stack=False,
    ) as prof:
        for step in range(wait + warmup + active):
            fn()
            torch.cuda.synchronize()
            prof.step()

    # ── export chrome trace ───────────────────────────────────────────────
    prof.export_chrome_trace(trace_file)

    # ── aggregate top CUDA kernels ────────────────────────────────────────
    key_avgs = prof.key_averages()
    cuda_events = [
        e for e in key_avgs
        if e.device_time_total > 0
    ]
    cuda_events.sort(key=lambda e: e.device_time_total, reverse=True)

    stats = {
        "label":       label,
        "trace_file":  trace_file,
        "top_kernels": [
            {
                "name":          e.key,
                "cuda_time_us":  round(e.device_time_total / active, 1),
                "cpu_time_us":   round(e.cpu_time_total    / active, 1),
                "calls":         e.count // active,
                "flops":         getattr(e, "flops", 0),
            }
            for e in cuda_events[:10]
        ],
        "total_cuda_time_us": round(
            sum(e.device_time_total for e in cuda_events) / active, 1
        ),
    }

    stats_file = out_path / f"{label}_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Profile saved → {trace_file}")
    print(f"Stats saved   → {stats_file}")
    return stats


# ── Roofline analysis ─────────────────────────────────────────────────────────

def compute_roofline_point(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    latency_ms: float,
    dtype_bytes: int = 2,   # float16 = 2 bytes
) -> Dict:
    """
    Computes the roofline operating point for one attention kernel config.

    Arithmetic intensity (FLOP/byte) determines whether the kernel is
    compute-bound or memory-bound relative to the A30 ridge point.

    Ridge point = peak_tflops / peak_bandwidth
                = 165 TFLOPS / 0.933 TB/s ≈ 177 FLOP/byte
    """
    # ── FLOPs: 2 × B × H × S² × D  (QK^T) + 2 × B × H × S² × D  (AV)
    flops = 4 * batch * num_heads * seq_len * seq_len * head_dim

    # ── Bytes: load Q, K, V  +  store O  (4 tensors × B × H × S × D × dtype)
    bytes_accessed = 4 * batch * num_heads * seq_len * head_dim * dtype_bytes

    arith_intensity = flops / bytes_accessed       # FLOP/byte

    ridge_point     = (A30_SPECS["peak_tflops_fp16"] * 1e12) / \
                      (A30_SPECS["peak_bandwidth_tbps"] * 1e12)

    achieved_tflops = flops / (latency_ms / 1000.0) / 1e12
    peak_bandwidth_tbps_achieved = bytes_accessed / (latency_ms / 1000.0) / 1e12

    bound = "compute" if arith_intensity >= ridge_point else "memory"

    return {
        "batch":                       batch,
        "seq_len":                     seq_len,
        "flops":                       flops,
        "bytes_accessed":              bytes_accessed,
        "arithmetic_intensity":        round(arith_intensity, 2),
        "ridge_point":                 round(ridge_point, 2),
        "achieved_tflops":             round(achieved_tflops, 3),
        "peak_bandwidth_tbps":         A30_SPECS["peak_bandwidth_tbps"],
        "achieved_bandwidth_tbps":     round(peak_bandwidth_tbps_achieved, 4),
        "pct_peak_compute":            round(achieved_tflops / A30_SPECS["peak_tflops_fp16"] * 100, 2),
        "pct_peak_bandwidth":          round(peak_bandwidth_tbps_achieved / A30_SPECS["peak_bandwidth_tbps"] * 100, 2),
        "bound":                       bound,
    }


def roofline_analysis(
    kernel_bench_results_path: str,
    output_path: str = "results/roofline.json",
) -> None:
    """
    Reads kernel_bench.json and annotates each row with roofline data.
    Saves augmented results to roofline.json.
    """
    with open(kernel_bench_results_path) as f:
        rows = json.load(f)

    augmented = []
    for row in rows:
        roofline = compute_roofline_point(
            batch=row["batch"],
            seq_len=row["seq_len"],
            num_heads=row["num_heads"],
            head_dim=row["head_dim"],
            latency_ms=row["latency_ms"],
        )
        augmented.append({**row, "roofline": roofline})

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(augmented, f, indent=2)

    print(f"Roofline analysis → {output_path}")

    # ── summary ──────────────────────────────────────────────────────────
    print("\n=== Roofline Summary ===")
    for r in augmented:
        rf = r["roofline"]
        print(f"  {r['kernel']} | seq={r['seq_len']} | "
              f"AI={rf['arithmetic_intensity']:.1f} FLOP/byte | "
              f"{rf['bound']}-bound | "
              f"{rf['pct_peak_compute']:.1f}% peak compute | "
              f"{rf['pct_peak_bandwidth']:.1f}% peak BW")
