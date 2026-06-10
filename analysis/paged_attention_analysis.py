"""
analysis/paged_attention_analysis.py

Analyzes KV cache memory efficiency: naive KV cache vs PagedAttention-style
paged allocation.

Simulates a serving workload with variable-length sequences and shows:
- Memory fragmentation in naive allocation
- How paging reduces fragmentation
- KV cache utilization (bytes used / bytes allocated)

Usage:
    python analysis/paged_attention_analysis.py
    python analysis/paged_attention_analysis.py --n_requests 2000 --max_seq_len 4096
"""

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class NaiveKVCache:
    """Pre-allocates max_seq_len * num_heads * head_dim per sequence."""

    def __init__(
        self,
        max_seq_len: int,
        block_size: int = 1,       # unused, kept for API symmetry
        num_heads: int = 32,
        head_dim: int = 128,
        dtype_bytes: int = 2,
    ):
        self.max_seq_len = max_seq_len
        self.num_heads   = num_heads
        self.head_dim    = head_dim
        self.dtype_bytes = dtype_bytes
        # bytes per token per KV layer (K + V, 2 tensors)
        self._bytes_per_token = 2 * num_heads * head_dim * dtype_bytes

    def _seq_allocated_bytes(self) -> int:
        return self.max_seq_len * self._bytes_per_token

    def allocate(self, batch_size: int, actual_lengths: list[int]) -> dict:
        """
        Returns:
            allocated_mb  – total memory pre-allocated for the batch
            used_mb       – memory actually used by real tokens
            fragmentation_pct – wasted fraction as a percentage
        """
        allocated = batch_size * self._seq_allocated_bytes()
        used = sum(min(l, self.max_seq_len) for l in actual_lengths) * self._bytes_per_token
        frag = (allocated - used) / allocated * 100 if allocated > 0 else 0.0
        return {
            "allocated_mb":      allocated / 1024**2,
            "used_mb":           used / 1024**2,
            "fragmentation_pct": frag,
        }


class PagedKVCache:
    """Allocates in fixed-size pages; only allocates pages as needed."""

    def __init__(
        self,
        page_size: int = 16,
        num_heads: int = 32,
        head_dim: int = 128,
        dtype_bytes: int = 2,
    ):
        self.page_size   = page_size
        self.num_heads   = num_heads
        self.head_dim    = head_dim
        self.dtype_bytes = dtype_bytes
        self._bytes_per_token = 2 * num_heads * head_dim * dtype_bytes
        self._bytes_per_page  = page_size * self._bytes_per_token

    def _pages_needed(self, seq_len: int) -> int:
        return math.ceil(seq_len / self.page_size)

    def allocate(self, actual_lengths: list[int]) -> dict:
        """
        Returns:
            allocated_mb  – pages allocated (rounded up per sequence)
            used_mb       – memory used by actual tokens
            fragmentation_pct – intra-page waste as a percentage
        """
        total_pages = sum(self._pages_needed(l) for l in actual_lengths)
        allocated   = total_pages * self._bytes_per_page
        used        = sum(actual_lengths) * self._bytes_per_token
        frag = (allocated - used) / allocated * 100 if allocated > 0 else 0.0
        return {
            "allocated_mb":      allocated / 1024**2,
            "used_mb":           used / 1024**2,
            "fragmentation_pct": frag,
        }


def simulate_serving_workload(
    n_requests: int = 1000,
    max_seq_len: int = 2048,
    batch_size: int = 32,
    page_size: int = 16,
    num_heads: int = 32,
    head_dim: int = 128,
    dtype_bytes: int = 2,
    seed: int = 42,
) -> dict:
    """
    Simulates requests with realistic length distribution (log-normal).
    Compares naive vs paged allocation across the workload.
    Returns memory savings statistics.

    Log-normal parameters chosen to produce a mean around 512 tokens
    with a long tail up to max_seq_len, matching real LLM serving traces.
    """
    rng = np.random.default_rng(seed)

    # log-normal: mean~512, std~400 tokens, clipped to [16, max_seq_len]
    mu    = math.log(512)
    sigma = 0.8
    lengths = rng.lognormal(mu, sigma, size=n_requests).astype(int)
    lengths = np.clip(lengths, 16, max_seq_len).tolist()

    naive = NaiveKVCache(max_seq_len, num_heads=num_heads,
                         head_dim=head_dim, dtype_bytes=dtype_bytes)
    paged = PagedKVCache(page_size, num_heads=num_heads,
                         head_dim=head_dim, dtype_bytes=dtype_bytes)

    naive_frags, paged_frags = [], []
    naive_alloc, paged_alloc = [], []

    # process in batches to model real serving behaviour
    for i in range(0, n_requests, batch_size):
        batch = lengths[i : i + batch_size]
        if not batch:
            break
        n_res = naive.allocate(len(batch), batch)
        p_res = paged.allocate(batch)
        naive_frags.append(n_res["fragmentation_pct"])
        paged_frags.append(p_res["fragmentation_pct"])
        naive_alloc.append(n_res["allocated_mb"])
        paged_alloc.append(p_res["allocated_mb"])

    naive_peak = max(naive_alloc)
    paged_peak = max(paged_alloc)
    savings_pct = (naive_peak - paged_peak) / naive_peak * 100

    return {
        "n_requests":         n_requests,
        "max_seq_len":        max_seq_len,
        "mean_length":        float(np.mean(lengths)),
        "lengths":            lengths,
        "naive_frags":        naive_frags,
        "paged_frags":        paged_frags,
        "naive_alloc_mb":     naive_alloc,
        "paged_alloc_mb":     paged_alloc,
        "naive_peak_gb":      naive_peak / 1024,
        "paged_peak_gb":      paged_peak / 1024,
        "naive_avg_frag_pct": float(np.mean(naive_frags)),
        "paged_avg_frag_pct": float(np.mean(paged_frags)),
        "memory_savings_pct": savings_pct,
    }


def plot_fragmentation_comparison(
    stats: dict,
    output_path: str = "results/paged_attention_analysis.png",
):
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_batches   = len(stats["naive_frags"])
    batch_idxs  = list(range(n_batches))
    lengths     = stats["lengths"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"KV Cache Memory Analysis — {stats['n_requests']} requests, "
        f"max_len={stats['max_seq_len']}",
        fontsize=13,
    )

    # 1 — Fragmentation over time
    ax = axes[0, 0]
    ax.plot(batch_idxs, stats["naive_frags"], label="Naive (pre-allocated)", alpha=0.8)
    ax.plot(batch_idxs, stats["paged_frags"], label="Paged (PagedAttention)", alpha=0.8)
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Fragmentation (%)")
    ax.set_title("KV Cache Fragmentation per Batch")
    ax.legend()

    # 2 — Memory allocated over time
    ax = axes[0, 1]
    naive_gb = [v / 1024 for v in stats["naive_alloc_mb"]]
    paged_gb = [v / 1024 for v in stats["paged_alloc_mb"]]
    ax.plot(batch_idxs, naive_gb, label="Naive", alpha=0.8)
    ax.plot(batch_idxs, paged_gb, label="Paged", alpha=0.8)
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Allocated KV cache (GB)")
    ax.set_title("Peak Memory Allocation per Batch")
    ax.legend()

    # 3 — Sequence length distribution
    ax = axes[1, 0]
    ax.hist(lengths, bins=50, edgecolor="none", alpha=0.75)
    ax.axvline(np.mean(lengths), color="red", linestyle="--", label=f"mean={np.mean(lengths):.0f}")
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("Count")
    ax.set_title("Request Length Distribution (log-normal)")
    ax.legend()

    # 4 — Bar summary
    ax = axes[1, 1]
    categories = ["Avg Fragmentation (%)", "Peak Memory (GB)"]
    naive_vals = [stats["naive_avg_frag_pct"], stats["naive_peak_gb"]]
    paged_vals = [stats["paged_avg_frag_pct"], stats["paged_peak_gb"]]
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width / 2, naive_vals, width, label="Naive")
    ax.bar(x + width / 2, paged_vals, width, label="Paged")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_title("Summary: Naive vs Paged")
    ax.legend()
    for i, (nv, pv) in enumerate(zip(naive_vals, paged_vals)):
        ax.text(i - width / 2, nv + 0.3, f"{nv:.1f}", ha="center", fontsize=8)
        ax.text(i + width / 2, pv + 0.3, f"{pv:.1f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="KV cache fragmentation analysis")
    parser.add_argument("--n_requests",  type=int, default=1000)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--page_size",   type=int, default=16,
                        help="PagedAttention block size (tokens per page)")
    parser.add_argument("--output",      default="results/paged_attention_analysis.png")
    args = parser.parse_args()

    print(f"\nRunning KV cache simulation: {args.n_requests} requests, "
          f"max_len={args.max_seq_len}, page_size={args.page_size}")

    stats = simulate_serving_workload(
        n_requests=args.n_requests,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        page_size=args.page_size,
    )

    print(f"\n{'=' * 60}")
    print(f" KV Cache Analysis ({stats['n_requests']} requests, max_len={stats['max_seq_len']})")
    print(f"{'=' * 60}")
    print(f" Mean request length:   {stats['mean_length']:.1f} tokens")
    print(f" Naive allocation:  avg fragmentation {stats['naive_avg_frag_pct']:5.1f}%, "
          f"peak {stats['naive_peak_gb']:5.1f} GB")
    print(f" Paged allocation:  avg fragmentation {stats['paged_avg_frag_pct']:5.1f}%, "
          f"peak {stats['paged_peak_gb']:5.1f} GB")
    print(f" Memory savings:    {stats['memory_savings_pct']:.1f}%")
    print(f"{'=' * 60}\n")

    plot_fragmentation_comparison(stats, args.output)


if __name__ == "__main__":
    main()
