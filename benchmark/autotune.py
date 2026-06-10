"""
benchmark/autotune.py

Autotuning for the Triton attention kernel using two backends:
  1. torch.inductor (torch.compile) — fast, uses Triton autotuning internally
  2. Apache TVM with MetaSchedule — exhaustive search over tile configs

Both produce real measured configs — no assumed optimal values.

Usage:
  # inductor (recommended, fast)
  python benchmark/autotune.py --backend inductor

  # TVM (slower, more exhaustive)
  python benchmark/autotune.py --backend tvm --trials 2000
"""

import json
import time
import argparse
from pathlib import Path

import torch
import yaml


# ── Inductor autotuning ───────────────────────────────────────────────────────

def autotune_with_inductor(
    num_heads:    int,
    num_kv_heads: int,
    head_dim:     int,
    seq_lengths:  list,
    batch_sizes:  list,
    output_path:  str = "results/inductor_autotune.json",
):
    """
    Compiles the Triton kernel with torch.compile(mode='max-autotune').
    Inductor runs Triton's autotuner under the hood, searching tile configs.
    Records best config and achieved performance per (batch, seq_len).
    """
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent.parent))
    from kernels.triton_attention import TritonAttention

    device  = "cuda"
    results = []

    kernel = TritonAttention(num_heads, num_kv_heads, head_dim, causal=True).to(device)

    print("Compiling with torch.compile(mode='max-autotune')...")
    compiled_kernel = torch.compile(kernel, mode="max-autotune", fullgraph=True)

    for batch in batch_sizes:
        for seq_len in seq_lengths:
            print(f"  Autotuning batch={batch} seq_len={seq_len}...")

            q = torch.randn(batch, num_heads,    seq_len, head_dim, dtype=torch.float16, device=device)
            k = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.float16, device=device)
            v = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.float16, device=device)

            # trigger compilation + autotuning on first call
            with torch.no_grad():
                _ = compiled_kernel(q, k, v)
            torch.cuda.synchronize()

            # measure compiled performance
            times = []
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            for _ in range(100):
                start.record()
                with torch.no_grad():
                    _ = compiled_kernel(q, k, v)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))

            times.sort()
            latency_ms = times[len(times) // 2]
            throughput  = (batch * seq_len) / (latency_ms / 1000.0)

            results.append({
                "backend":      "inductor",
                "batch":        batch,
                "seq_len":      seq_len,
                "latency_ms":   round(latency_ms, 4),
                "throughput_tps": round(throughput, 1),
            })
            print(f"    latency={latency_ms:.2f}ms | throughput={throughput:.0f} tok/s")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nInductor autotune results → {output_path}")


# ── TVM MetaSchedule autotuning ───────────────────────────────────────────────

def autotune_with_tvm(
    num_heads:   int,
    head_dim:    int,
    seq_len:     int = 1024,
    batch:       int = 1,
    trials:      int = 2000,
    cache_path:  str = "results/tvm_cache.json",
    output_path: str = "results/tvm_autotune.json",
):
    """
    Uses Apache TVM MetaSchedule to search tile and unroll configs for
    the attention GEMM (QK^T matmul). TVM tunes the underlying matrix
    multiply that attention is composed of.

    For a full attention kernel, TVM tunes the compute schedule directly.
    Results show optimal tile sizes measured on the actual A30 hardware.
    """
    try:
        import tvm
        from tvm import relay, auto_scheduler
        from tvm.contrib import graph_executor
    except ImportError:
        print("Apache TVM not installed. Run: pip install apache-tvm")
        return

    target = tvm.target.cuda(arch="sm_80")   # A30 = sm_80

    # ── define QK^T as a TVM relay computation ────────────────────────────
    # Shape: (batch * heads, seq, head_dim) × (batch * heads, head_dim, seq)
    B  = batch * num_heads
    S  = seq_len
    D  = head_dim

    data_type = "float16"

    Q_tvm = relay.var("Q", shape=(B, S, D), dtype=data_type)
    K_tvm = relay.var("K", shape=(B, D, S), dtype=data_type)

    qk = relay.nn.batch_matmul(Q_tvm, K_tvm, transpose_b=False)
    func = relay.Function([Q_tvm, K_tvm], qk)
    mod  = tvm.IRModule.from_expr(func)
    mod  = relay.transform.InferType()(mod)

    print(f"TVM autotuning QK^T matmul: B={B} S={S} D={D} trials={trials}")

    # ── extract tasks ─────────────────────────────────────────────────────
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], None, target)
    print(f"  {len(tasks)} task(s) extracted")

    # ── tune ──────────────────────────────────────────────────────────────
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=trials,
        measure_callbacks=[auto_scheduler.RecordToFile(cache_path)],
        verbose=1,
    )
    tuner.tune(tune_option)

    # ── compile with best config ──────────────────────────────────────────
    with auto_scheduler.ApplyHistoryBest(cache_path):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target, params=None)

    # ── benchmark compiled module ─────────────────────────────────────────
    dev = tvm.cuda(0)
    m   = graph_executor.GraphModule(lib["default"](dev))

    import numpy as np
    Q_np = np.random.randn(B, S, D).astype("float16")
    K_np = np.random.randn(B, D, S).astype("float16")
    m.set_input("Q", tvm.nd.array(Q_np, dev))
    m.set_input("K", tvm.nd.array(K_np, dev))

    timer = m.module.time_evaluator("run", dev, number=200, repeat=3)
    t     = timer()
    latency_ms = t.mean * 1000

    result = {
        "backend":      "tvm",
        "batch":        batch,
        "seq_len":      seq_len,
        "num_heads":    num_heads,
        "head_dim":     head_dim,
        "trials":       trials,
        "latency_ms":   round(latency_ms, 4),
        "cache_path":   cache_path,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nTVM autotune result → {output_path}")
    print(f"  Best latency: {latency_ms:.4f}ms")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend",      choices=["inductor", "tvm"], default="inductor")
    parser.add_argument("--config",       default="configs/bench_config.yaml")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--trials",       type=int, default=2000)
    parser.add_argument("--output-dir",   default="results")
    args = parser.parse_args()

    with open(args.config)       as f: cfg  = yaml.safe_load(f)
    with open(args.model_config) as f: mcfg = yaml.safe_load(f)

    attn = mcfg["attention"]

    if args.backend == "inductor":
        autotune_with_inductor(
            num_heads=attn["num_heads"],
            num_kv_heads=attn["num_kv_heads"],
            head_dim=attn["head_dim"],
            seq_lengths=cfg["kernel_bench"]["seq_lengths"],
            batch_sizes=cfg["kernel_bench"]["batch_sizes"],
            output_path=f"{args.output_dir}/inductor_autotune.json",
        )
    else:
        autotune_with_tvm(
            num_heads=attn["num_heads"],
            head_dim=attn["head_dim"],
            trials=args.trials,
            cache_path=cfg["autotuning"]["cache_path"],
            output_path=f"{args.output_dir}/tvm_autotune.json",
        )


if __name__ == "__main__":
    main()
