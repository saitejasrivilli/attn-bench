#!/bin/bash
# setup.sh — A30 cluster environment setup
# Run once per node before benchmarking

set -e

echo "=== Setting up attention_benchmark environment ==="

# ── 1. Conda environment ──────────────────────────────────────────────────────
conda create -n attn_bench python=3.11 -y
conda activate attn_bench

# ── 2. Core PyTorch (CUDA 12.1, matches A30 driver) ──────────────────────────
pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ── 3. Triton (ships with PyTorch 2.3 but pin explicitly) ────────────────────
pip install triton==2.3.0

# ── 4. FlashAttention-2 ───────────────────────────────────────────────────────
pip install flash-attn==2.5.9.post1 --no-build-isolation

# ── 5. vLLM ──────────────────────────────────────────────────────────────────
pip install vllm==0.4.3

# ── 6. SGLang ─────────────────────────────────────────────────────────────────
pip install "sglang[all]==0.2.15"

# ── 7. Profiling and benchmark utilities ──────────────────────────────────────
pip install \
    numpy==1.26.4 \
    pandas==2.2.2 \
    matplotlib==3.9.0 \
    seaborn==0.13.2 \
    pyyaml==6.0.1 \
    tqdm==4.66.4 \
    aiohttp==3.9.5 \
    httpx==0.27.0 \
    transformers==4.41.2 \
    accelerate==0.30.1 \
    huggingface_hub==0.23.2

# ── 8. Apache TVM (for autotuning experiments) ───────────────────────────────
pip install apache-tvm==0.16.0

# ── 9. Verify installs ────────────────────────────────────────────────────────
python - <<'EOF'
import torch, triton, flash_attn, vllm
print(f"PyTorch:       {torch.__version__}")
print(f"CUDA:          {torch.version.cuda}")
print(f"Triton:        {triton.__version__}")
print(f"FlashAttn:     {flash_attn.__version__}")
print(f"vLLM:          {vllm.__version__}")
print(f"GPUs:          {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} | {props.total_memory // 1024**3}GB VRAM")
EOF

echo ""
echo "=== Setup complete. Activate with: conda activate attn_bench ==="
