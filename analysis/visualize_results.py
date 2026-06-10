"""
analysis/visualize_results.py

Reads all results JSON files and produces:
  1. Console summary table (copy-paste ready for resume/writeup)
  2. results/summary.json — merged all-results table
  3. results/plots/     — matplotlib figures

Usage:
  python analysis/visualize_results.py --results-dir results/
"""

import json
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")   # no display needed on cluster
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


# ── Load all throughput results ───────────────────────────────────────────────

def load_results(results_dir: str) -> pd.DataFrame:
    rdir = Path(results_dir)
    rows = []

    for f in rdir.glob("throughput_*.json"):
        with open(f) as fh:
            data = json.load(fh)
        rows.extend(data)

    if not rows:
        raise FileNotFoundError(f"No throughput_*.json files found in {results_dir}")

    return pd.DataFrame(rows)


# ── Console summary table ─────────────────────────────────────────────────────

def print_summary_table(df: pd.DataFrame):
    print("\n" + "=" * 90)
    print(" BENCHMARK SUMMARY — Actual measured results on 4× NVIDIA A30")
    print("=" * 90)

    combos = [
        ("vllm",   "flashattn2", "vLLM + FlashAttention-2 (CUDA baseline)"),
        ("vllm",   "triton",     "vLLM + Triton custom kernel"),
        ("sglang", "flashattn2", "SGLang + FlashInfer (CUDA baseline)"),
        ("sglang", "triton",     "SGLang + Triton custom kernel"),
    ]

    for engine, kernel, label in combos:
        subset = df[(df["engine"] == engine) & (df["kernel"] == kernel)]
        if subset.empty:
            print(f"\n{label}: NO DATA")
            continue

        print(f"\n{label}")
        print(f"  {'Concurrency':>11} | {'Throughput (tok/s)':>18} | {'TTFT p50 (ms)':>13} | "
              f"{'TTFT p95 (ms)':>13} | {'P99 lat (ms)':>12} | {'Success':>7}")
        print("  " + "-" * 85)

        for _, row in subset.sort_values("concurrency").iterrows():
            print(f"  {int(row['concurrency']):>11} | "
                  f"{row['throughput_tps']:>18.1f} | "
                  f"{row['ttft_p50_ms']:>13.1f} | "
                  f"{row['ttft_p95_ms']:>13.1f} | "
                  f"{row['latency_p99_ms']:>12.1f} | "
                  f"{row['success_rate']:>7.1%}")

    # ── speedup table ─────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(" SPEEDUP: Triton vs FlashAttn-2 baseline (by concurrency)")
    print("=" * 90)

    for engine in ["vllm", "sglang"]:
        base   = df[(df["engine"] == engine) & (df["kernel"] == "flashattn2")]
        triton = df[(df["engine"] == engine) & (df["kernel"] == "triton")]

        if base.empty or triton.empty:
            continue

        print(f"\n  Engine: {engine.upper()}")
        print(f"  {'Concurrency':>11} | {'Base tok/s':>10} | {'Triton tok/s':>12} | "
              f"{'Speedup':>8} | {'TTFT improvement':>18}")
        print("  " + "-" * 70)

        for c in sorted(df["concurrency"].unique()):
            b = base[base["concurrency"]   == c]
            t = triton[triton["concurrency"] == c]
            if b.empty or t.empty:
                continue
            b, t = b.iloc[0], t.iloc[0]
            speedup = t["throughput_tps"] / b["throughput_tps"]
            ttft_imp = (b["ttft_p50_ms"] - t["ttft_p50_ms"]) / b["ttft_p50_ms"] * 100
            print(f"  {int(c):>11} | {b['throughput_tps']:>10.1f} | "
                  f"{t['throughput_tps']:>12.1f} | "
                  f"{speedup:>7.2f}x | "
                  f"{ttft_imp:>17.1f}%")

    print()


# ── Plots ─────────────────────────────────────────────────────────────────────

def make_plots(df: pd.DataFrame, out_dir: str):
    plots_dir = Path(out_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", palette="colorblind")
    df["label"] = df["engine"].str.upper() + " + " + df["kernel"].str.replace("flashattn2", "FlashAttn2").str.replace("triton", "Triton")

    # ── 1. Throughput vs concurrency ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, group in df.groupby("label"):
        group = group.sort_values("concurrency")
        ax.plot(group["concurrency"], group["throughput_tps"], marker="o", label=label)
    ax.set_xlabel("Concurrent users")
    ax.set_ylabel("Throughput (tokens / second)")
    ax.set_title("Throughput vs Concurrency — All 4 Combinations")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mticker.FixedLocator(sorted(df["concurrency"].unique())))
    fig.tight_layout()
    fig.savefig(plots_dir / "throughput_vs_concurrency.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {plots_dir}/throughput_vs_concurrency.png")

    # ── 2. TTFT P95 vs concurrency ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, group in df.groupby("label"):
        group = group.sort_values("concurrency")
        ax.plot(group["concurrency"], group["ttft_p95_ms"], marker="s", label=label)
    ax.set_xlabel("Concurrent users")
    ax.set_ylabel("TTFT P95 (ms)")
    ax.set_title("Time to First Token (P95) vs Concurrency")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / "ttft_p95_vs_concurrency.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {plots_dir}/ttft_p95_vs_concurrency.png")

    # ── 3. P99 latency heatmap (engine × kernel × concurrency) ───────────
    pivot = df.pivot_table(
        index=["engine", "kernel"], columns="concurrency", values="latency_p99_ms"
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("P99 End-to-End Latency (ms) by Engine × Kernel × Concurrency")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(plots_dir / "p99_latency_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {plots_dir}/p99_latency_heatmap.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    print(f"Loading results from {args.results_dir}...")
    df = load_results(args.results_dir)

    summary_path = Path(args.results_dir) / "summary.json"
    df.to_json(summary_path, orient="records", indent=2)
    print(f"Merged summary → {summary_path}")

    print_summary_table(df)

    print("\nGenerating plots...")
    make_plots(df, args.results_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
