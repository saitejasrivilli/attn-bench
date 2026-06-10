#!/bin/bash
# scripts/run_all.sh
# Full benchmark pipeline — runs all 4 combinations end-to-end
# Must be run from project root with conda env active

set -e
cd "$(dirname "$0")/.."

MODEL_DIR="./weights/mistral-7b"
RESULTS_DIR="./results"
LOG_DIR="./logs"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

echo "============================================================"
echo " Attention Kernel Benchmark — Full Pipeline"
echo "============================================================"
echo ""

# ── Step 0: Correctness check ─────────────────────────────────────────────
echo "[0/6] Running Triton kernel correctness check..."
python -c "
import sys; sys.path.insert(0, '.')
from kernels.triton_attention import verify_correctness
ok = verify_correctness()
sys.exit(0 if ok else 1)
"
echo "  Correctness: PASS"
echo ""

# ── Step 1: Standalone kernel benchmark ──────────────────────────────────
echo "[1/6] Standalone kernel benchmark (no engine overhead)..."
python benchmark/kernel_bench.py \
    --config configs/bench_config.yaml \
    --model-config configs/model_config.yaml \
    --output-dir "$RESULTS_DIR"
echo ""

# ── Step 2: Roofline analysis ─────────────────────────────────────────────
echo "[2/6] Roofline analysis..."
python -c "
import sys; sys.path.insert(0, '.')
from benchmark.profiler import roofline_analysis
roofline_analysis('results/kernel_bench.json', 'results/roofline.json')
"
echo ""

# ── Step 3: Autotune (inductor) ────────────────────────────────────────────
echo "[3/6] Autotuning with torch.inductor..."
python benchmark/autotune.py \
    --backend inductor \
    --config configs/bench_config.yaml \
    --model-config configs/model_config.yaml \
    --output-dir "$RESULTS_DIR"
echo ""

# ── Step 4: Engine benchmarks — 4 combinations ────────────────────────────
echo "[4/6] Engine benchmark: vLLM + FlashAttention-2 (baseline)..."
VLLM_PORT=8000

# launch vLLM with FlashAttention-2 baseline
VLLM_ATTENTION_BACKEND=FLASH_ATTN python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --port $VLLM_PORT \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --dtype float16 \
    --max-model-len 4096 \
    --disable-log-requests \
    > "$LOG_DIR/vllm_flashattn2.log" 2>&1 &
VLLM_PID=$!

# wait for server
echo "  Waiting for vLLM (FlashAttn2) to be ready..."
until curl -s http://127.0.0.1:${VLLM_PORT}/health > /dev/null 2>&1; do sleep 5; done
echo "  vLLM ready."

python benchmark/throughput_bench.py \
    --engine vllm --kernel flashattn2 \
    --port $VLLM_PORT \
    --concurrency 1 4 16 32 \
    --output-dir "$RESULTS_DIR"

kill $VLLM_PID && wait $VLLM_PID 2>/dev/null || true
sleep 10   # allow GPU memory to release
echo ""

echo "[4b/6] Engine benchmark: vLLM + Triton custom kernel..."
VLLM_ATTENTION_BACKEND=TRITON_CUSTOM python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --port $VLLM_PORT \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --dtype float16 \
    --max-model-len 4096 \
    --disable-log-requests \
    > "$LOG_DIR/vllm_triton.log" 2>&1 &
VLLM_PID=$!

until curl -s http://127.0.0.1:${VLLM_PORT}/health > /dev/null 2>&1; do sleep 5; done
echo "  vLLM (Triton) ready."

python benchmark/throughput_bench.py \
    --engine vllm --kernel triton \
    --port $VLLM_PORT \
    --concurrency 1 4 16 32 \
    --output-dir "$RESULTS_DIR"

kill $VLLM_PID && wait $VLLM_PID 2>/dev/null || true
sleep 10
echo ""

echo "[5/6] Engine benchmark: SGLang + FlashInfer (baseline)..."
SGLANG_PORT=30000

python -m sglang.launch_server \
    --model-path "$MODEL_DIR" \
    --port $SGLANG_PORT \
    --tp 2 \
    --mem-fraction-static 0.88 \
    --dtype float16 \
    --max-prefill-tokens 8192 \
    --chunked-prefill-size 512 \
    > "$LOG_DIR/sglang_flashinfer.log" 2>&1 &
SGLANG_PID=$!

echo "  Waiting for SGLang to be ready..."
until curl -s http://127.0.0.1:${SGLANG_PORT}/health > /dev/null 2>&1; do sleep 5; done
echo "  SGLang ready."

python benchmark/throughput_bench.py \
    --engine sglang --kernel flashattn2 \
    --port $SGLANG_PORT \
    --concurrency 1 4 16 32 \
    --output-dir "$RESULTS_DIR"

kill $SGLANG_PID && wait $SGLANG_PID 2>/dev/null || true
sleep 10
echo ""

echo "[5b/6] Engine benchmark: SGLang + Triton custom kernel..."
SGLANG_CUSTOM_ATTENTION_BACKEND="./engines/sglang_triton_hook.py" \
python -m sglang.launch_server \
    --model-path "$MODEL_DIR" \
    --port $SGLANG_PORT \
    --tp 2 \
    --mem-fraction-static 0.88 \
    --dtype float16 \
    --max-prefill-tokens 8192 \
    > "$LOG_DIR/sglang_triton.log" 2>&1 &
SGLANG_PID=$!

until curl -s http://127.0.0.1:${SGLANG_PORT}/health > /dev/null 2>&1; do sleep 5; done
echo "  SGLang (Triton) ready."

python benchmark/throughput_bench.py \
    --engine sglang --kernel triton \
    --port $SGLANG_PORT \
    --concurrency 1 4 16 32 \
    --output-dir "$RESULTS_DIR"

kill $SGLANG_PID && wait $SGLANG_PID 2>/dev/null || true
echo ""

# ── Step 6: Merge results and print summary ───────────────────────────────
echo "[6/6] Merging results..."
python analysis/visualize_results.py --results-dir "$RESULTS_DIR"

echo ""
echo "============================================================"
echo " Benchmark complete. Results in: $RESULTS_DIR"
echo "============================================================"
