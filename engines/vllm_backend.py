"""
engines/vllm_backend.py

Registers the custom Triton attention kernel as a vLLM attention backend.
vLLM's attention layer is pluggable via its AttentionBackend interface.

Usage:
  Set VLLM_ATTENTION_BACKEND=TRITON_CUSTOM before launching vLLM server,
  or pass --attention-backend triton_custom when using the OpenAI-compatible server.

  The vllm_server_launcher() function below handles this automatically.
"""

import os
import sys
import subprocess
import time
import signal
import requests
from pathlib import Path
from typing import Optional


# ── vLLM AttentionBackend interface ──────────────────────────────────────────
# vLLM dynamically loads backends via the VLLM_ATTENTION_BACKEND env var.
# We implement the required interface so our Triton kernel is called instead
# of the default FlashAttention-2 CUDA backend.

try:
    import torch
    from vllm.attention.backends.abstract import (
        AttentionBackend,
        AttentionImpl,
        AttentionMetadata,
        AttentionType,
    )

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from kernels.triton_attention import TritonAttention

    class TritonCustomBackend(AttentionBackend):
        """
        vLLM attention backend that routes prefill + decode through the
        custom Triton kernel instead of FlashAttention-2.
        """

        @staticmethod
        def get_name() -> str:
            return "TRITON_CUSTOM"

        @staticmethod
        def get_impl_cls():
            return TritonCustomImpl

        @staticmethod
        def get_metadata_cls():
            from vllm.attention.backends.flash_attn import FlashAttentionMetadata
            return FlashAttentionMetadata

        @staticmethod
        def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
            return (2, num_blocks, block_size, num_kv_heads, head_size)

        @staticmethod
        def swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst):
            from vllm.attention.backends.flash_attn import FlashAttentionBackend
            FlashAttentionBackend.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

        @staticmethod
        def copy_blocks(kv_caches, src_to_dists):
            from vllm.attention.backends.flash_attn import FlashAttentionBackend
            FlashAttentionBackend.copy_blocks(kv_caches, src_to_dists)

    class TritonCustomImpl(AttentionImpl):

        def __init__(self, num_heads, head_size, scale, num_kv_heads=None,
                     alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto",
                     blocksparse_params=None, logits_soft_cap=None):
            self.num_heads    = num_heads
            self.num_kv_heads = num_kv_heads or num_heads
            self.head_size    = head_size
            self.scale        = scale
            self._kernel = TritonAttention(
                num_heads=num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=head_size,
                causal=True,
            )

        def forward(self, query, key, value, kv_cache, attn_metadata,
                    kv_scale=1.0, attn_type=AttentionType.DECODER):
            # vLLM passes (num_tokens, num_heads * head_size) shaped tensors
            # Reshape to (batch, heads, seq, head_dim) for our kernel
            num_tokens = query.shape[0]

            q = query.view(1, num_tokens, self.num_heads,    self.head_size).transpose(1, 2)
            k = key.view  (1, num_tokens, self.num_kv_heads, self.head_size).transpose(1, 2)
            v = value.view (1, num_tokens, self.num_kv_heads, self.head_size).transpose(1, 2)

            out = self._kernel(q, k, v)
            return out.transpose(1, 2).contiguous().view(num_tokens, -1)

except ImportError:
    # vLLM not installed in this environment — backend classes won't be available
    # but server launcher functions still work
    pass


# ── Server launcher ───────────────────────────────────────────────────────────

def launch_vllm_server(
    model_path: str,
    port: int = 8000,
    tensor_parallel: int = 2,
    gpu_memory_utilization: float = 0.90,
    dtype: str = "float16",
    max_model_len: int = 4096,
    attention_backend: str = "FLASH_ATTN",   # or "TRITON_CUSTOM"
    log_file: Optional[str] = None,
) -> subprocess.Popen:
    """
    Launches vLLM OpenAI-compatible server as a subprocess.
    Returns the Popen handle — caller is responsible for termination.

    attention_backend:
      "FLASH_ATTN"     → stock FlashAttention-2 CUDA kernel (baseline)
      "TRITON_CUSTOM"  → our custom Triton kernel
    """
    env = os.environ.copy()
    env["VLLM_ATTENTION_BACKEND"] = attention_backend

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model",                   model_path,
        "--port",                    str(port),
        "--tensor-parallel-size",    str(tensor_parallel),
        "--gpu-memory-utilization",  str(gpu_memory_utilization),
        "--dtype",                   dtype,
        "--max-model-len",           str(max_model_len),
        "--disable-log-requests",
    ]

    print(f"Launching vLLM [{attention_backend}] on port {port}...")
    stdout = open(log_file, "w") if log_file else subprocess.DEVNULL
    proc   = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=subprocess.STDOUT)
    _wait_for_server(f"http://127.0.0.1:{port}/health", timeout=180)
    print(f"vLLM [{attention_backend}] ready on port {port}")
    return proc


def stop_server(proc: subprocess.Popen, timeout: int = 30):
    """Gracefully terminate a server process."""
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()


def _wait_for_server(url: str, timeout: int = 180, interval: float = 3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(interval)
    raise TimeoutError(f"Server at {url} did not become ready within {timeout}s")
