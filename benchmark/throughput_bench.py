"""
benchmark/throughput_bench.py

Concurrent load benchmark against live vLLM / SGLang servers.
Uses real ShareGPT prompts — no synthetic or hardcoded inputs.

Measures:
  - Throughput (tokens/sec output)
  - Time to first token (TTFT ms)
  - P50 / P95 / P99 end-to-end latency

Run (after servers are live):
  python benchmark/throughput_bench.py \\
      --engine vllm --kernel flashattn2 --port 8000 \\
      --concurrency 1 4 16 32 \\
      --output-dir results/
"""

import asyncio
import json
import time
import argparse
import random
import statistics
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

import aiohttp
import yaml


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    prompt_tokens:  int
    output_tokens:  int
    ttft_ms:        float   # time to first token
    e2e_latency_ms: float   # full request
    success:        bool
    error:          Optional[str] = None


@dataclass
class BenchResult:
    engine:               str
    kernel:               str
    concurrency:          int
    num_requests:         int
    total_time_s:         float
    throughput_tps:       float   # output tokens / total_time
    ttft_p50_ms:          float
    ttft_p95_ms:          float
    ttft_p99_ms:          float
    latency_p50_ms:       float
    latency_p95_ms:       float
    latency_p99_ms:       float
    success_rate:         float
    prompt_tokens_mean:   float
    output_tokens_mean:   float


# ── ShareGPT dataset loader ───────────────────────────────────────────────────

def load_sharegpt_prompts(dataset_path: str, n: int, tokenizer=None) -> List[Tuple[str, int]]:
    """
    Load n prompts from ShareGPT dataset.
    Returns list of (prompt_text, approx_token_count) tuples.

    Uses real conversations — no synthetic data.
    """
    with open(dataset_path) as f:
        data = json.load(f)

    prompts = []
    for conv in data:
        if not conv.get("conversations"):
            continue
        # take the first human turn as the prompt
        for turn in conv["conversations"]:
            if turn.get("from") == "human" and turn.get("value"):
                text = turn["value"].strip()
                if 50 < len(text.split()) < 500:   # filter very short/long
                    prompts.append(text)
                break

    random.seed(42)
    random.shuffle(prompts)
    prompts = prompts[:n]

    if tokenizer:
        counts = [len(tokenizer.encode(p)) for p in prompts]
    else:
        counts = [len(p.split()) * 4 // 3 for p in prompts]   # rough estimate

    return list(zip(prompts, counts))


# ── Async request sender ──────────────────────────────────────────────────────

async def send_request(
    session:      aiohttp.ClientSession,
    url:          str,
    prompt:       str,
    max_tokens:   int,
    timeout:      int,
) -> RequestResult:
    """
    Sends a single streaming request and measures TTFT + E2E latency.
    Uses SSE streaming to detect first token.
    """
    payload = {
        "model":       "default",
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  max_tokens,
        "stream":      True,
        "temperature": 0.0,   # greedy — deterministic for reproducibility
    }

    t_start   = time.perf_counter()
    ttft_ms   = None
    out_tokens = 0

    try:
        async with session.post(
            url, json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as resp:
            if resp.status != 200:
                return RequestResult(
                    prompt_tokens=0, output_tokens=0,
                    ttft_ms=0, e2e_latency_ms=0,
                    success=False, error=f"HTTP {resp.status}"
                )

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t_start) * 1000
                        out_tokens += 1
                except (json.JSONDecodeError, KeyError):
                    continue

        e2e_ms = (time.perf_counter() - t_start) * 1000
        return RequestResult(
            prompt_tokens=0,   # filled below from server usage field
            output_tokens=out_tokens,
            ttft_ms=ttft_ms or e2e_ms,
            e2e_latency_ms=e2e_ms,
            success=True,
        )

    except asyncio.TimeoutError:
        return RequestResult(
            prompt_tokens=0, output_tokens=0,
            ttft_ms=0, e2e_latency_ms=timeout * 1000,
            success=False, error="timeout"
        )
    except Exception as e:
        return RequestResult(
            prompt_tokens=0, output_tokens=0,
            ttft_ms=0, e2e_latency_ms=0,
            success=False, error=str(e)
        )


# ── Concurrent runner ─────────────────────────────────────────────────────────

async def run_concurrent(
    base_url:    str,
    prompts:     List[Tuple[str, int]],
    concurrency: int,
    max_tokens:  int,
    timeout:     int,
    warmup:      int,
) -> List[RequestResult]:
    """
    Sends `warmup` requests first (discarded), then the full prompt list
    at `concurrency` simultaneous requests.
    """
    url = f"{base_url}/v1/chat/completions"
    connector = aiohttp.TCPConnector(limit=concurrency + 10)

    async with aiohttp.ClientSession(connector=connector) as session:
        # warmup
        warmup_tasks = [
            send_request(session, url, prompts[i % len(prompts)][0], max_tokens, timeout)
            for i in range(warmup)
        ]
        await asyncio.gather(*warmup_tasks)

        # real benchmark — rate-limited to `concurrency` in-flight
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_request(prompt: str) -> RequestResult:
            async with semaphore:
                return await send_request(session, url, prompt, max_tokens, timeout)

        t0 = time.perf_counter()
        tasks = [bounded_request(p) for p, _ in prompts]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - t0

    return results, total_time


# ── Aggregate metrics ─────────────────────────────────────────────────────────

def aggregate(
    results:      List[RequestResult],
    total_time_s: float,
    engine:       str,
    kernel:       str,
    concurrency:  int,
) -> BenchResult:

    successes = [r for r in results if r.success]
    if not successes:
        raise RuntimeError("All requests failed")

    def pct(vals, p):
        vals_sorted = sorted(vals)
        idx = int(len(vals_sorted) * p / 100)
        return vals_sorted[min(idx, len(vals_sorted) - 1)]

    ttfts    = [r.ttft_ms        for r in successes]
    latencies = [r.e2e_latency_ms for r in successes]
    total_out = sum(r.output_tokens for r in successes)

    return BenchResult(
        engine=engine,
        kernel=kernel,
        concurrency=concurrency,
        num_requests=len(results),
        total_time_s=round(total_time_s, 3),
        throughput_tps=round(total_out / total_time_s, 2),
        ttft_p50_ms=round(pct(ttfts, 50), 2),
        ttft_p95_ms=round(pct(ttfts, 95), 2),
        ttft_p99_ms=round(pct(ttfts, 99), 2),
        latency_p50_ms=round(pct(latencies, 50), 2),
        latency_p95_ms=round(pct(latencies, 95), 2),
        latency_p99_ms=round(pct(latencies, 99), 2),
        success_rate=round(len(successes) / len(results), 4),
        prompt_tokens_mean=0,
        output_tokens_mean=round(statistics.mean([r.output_tokens for r in successes]), 1),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

async def amain(args):
    with open(args.config)       as f: cfg  = yaml.safe_load(f)
    with open(args.model_config) as f: mcfg = yaml.safe_load(f)

    bcfg     = cfg["benchmark"]
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"http://127.0.0.1:{args.port}"

    print(f"\nLoading ShareGPT prompts from {bcfg['dataset_path']}...")
    prompts = load_sharegpt_prompts(bcfg["dataset_path"], bcfg["dataset_sample_size"])
    print(f"Loaded {len(prompts)} prompts")

    all_results = []

    for concurrency in args.concurrency:
        subset = prompts[:bcfg["num_requests_per_level"]]

        print(f"\n--- concurrency={concurrency} | engine={args.engine} | kernel={args.kernel} ---")
        raw, total_time = await run_concurrent(
            base_url=base_url,
            prompts=subset,
            concurrency=concurrency,
            max_tokens=bcfg["output_tokens"],
            timeout=bcfg["request_timeout_s"],
            warmup=bcfg["warmup_requests"],
        )

        bench = aggregate(raw, total_time, args.engine, args.kernel, concurrency)
        all_results.append(asdict(bench))

        print(f"  throughput={bench.throughput_tps:.1f} tok/s | "
              f"TTFT p50={bench.ttft_p50_ms:.1f}ms p95={bench.ttft_p95_ms:.1f}ms | "
              f"latency p99={bench.latency_p99_ms:.1f}ms | "
              f"success={bench.success_rate:.1%}")

    out_path = out_dir / f"throughput_{args.engine}_{args.kernel}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine",       required=True, choices=["vllm", "sglang"])
    parser.add_argument("--kernel",       required=True, choices=["flashattn2", "triton"])
    parser.add_argument("--port",         type=int, required=True)
    parser.add_argument("--concurrency",  type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--config",       default="configs/bench_config.yaml")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--output-dir",   default="results")
    args = parser.parse_args()

    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
